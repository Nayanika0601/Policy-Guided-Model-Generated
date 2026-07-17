
import os

import cv2
import time
import json
import threading
import signal

from mediapipe_processor import (
    face_landmarker, pose_landmarker,
    MovementTracker, ObservationBuilder,
    process_frame_data, draw_overlay,
    OBSERVATION_INTERVAL_SEC, NO_MOVEMENT_TIMEOUT_SEC,
    CAMERA_INDEX, SHOW_WINDOW,
)
from speech_output import speak, try_speak
import sounddevice as sd
from speech_input import listen_for_answer, SAMPLE_RATE, CHANNELS, BLOCK_SAMPLES
from quiz_manager import QuizManager
from engagement_manager import EngagementManager
from logger import SessionLogger
from llm_responder import generate as llm_generate, LLM_MODE as _LLM_MODE
from pedagogical_policy import select_strategy, STRATEGY_DESCRIPTIONS
from prompt_assembler import assemble_prompt
from llm_responder import _history as _llm_history
from hand_raise_manager import handle_hand_raise as build_hand_raise_help_event

_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
with open(os.path.join(_CFG_DIR, "settings.json"), "r") as f:
    _settings = json.load(f)

_thresholds = _settings["thresholds"]
_COOLDOWN_SEC        = float(_settings["cooldown_sec"])
_HAND_RAISE_SEC      = 3.0
_FACE_ABSENT_SEC     = float(_thresholds["face_absent_sec"])
_NO_MOVEMENT_SEC     = float(_thresholds["no_movement_sec"])
_GAZE_AWAY_THRESH    = float(_thresholds["gaze_away_thresh"])

MAX_QUESTIONS = 10

_YES_WORDS = {"yes", "yeah", "yep", "yup", "sure", "uh huh", "ok", "okay"}
_NO_WORDS  = {"no", "nope", "nah", "not really", "never mind", "nevermind"}


def _is_yes(text: str) -> bool:
    t = text.strip().lower().rstrip(".!?,")
    return any(w in t for w in _YES_WORDS)


def _is_no(text: str) -> bool:
    t = text.strip().lower().rstrip(".!?,")
    return any(w in t for w in _NO_WORDS)


def build_engagement_state(trigger: str | None) -> dict:
    state = {
        "hand_raised": False,
        "face_absent": False,
        "gaze_away": False,
        "no_movement": False,
    }

    if trigger in {
        "hand_raised",
        "hand_raised_opening",
        "hand_raised_followup",
        "hand_raised_closing_asked",
        "hand_raised_closing_no",
        "hand_raised_closing_silence",
    }:
        state["hand_raised"] = True
    elif trigger == "face_absent":
        state["face_absent"] = True
    elif trigger == "head_turned":
        state["gaze_away"] = True
    elif trigger == "no_movement":
        state["no_movement"] = True

    return state


def _strategy_log_fields(llm_result: dict) -> dict:
    decision = llm_result.get("strategy_decision", {})
    return {
        "selected_strategy": decision.get("strategy"),
        "strategy_reason": decision.get("reason"),
        "strategy_description": decision.get("strategy_description"),
        "strategy_concept": decision.get("concept"),
        "strategy_weak_concept": decision.get("weak_concept"),
    }


def _learner_state_dict(learner_state) -> dict:
    if learner_state is None:
        return {}
    if hasattr(learner_state, "to_dict"):
        return learner_state.to_dict()
    if hasattr(learner_state, "get_summary"):
        return learner_state.get_summary()
    return {}


def _fire_llm_with_strategy(obs: dict, trigger: str, mode: str,
                             quiz_manager=None,
                             quiz_context: dict | None = None,
                             turn_state_override: dict | None = None,
                             learner_state_override=None) -> dict:
    interaction_dict = {}
    learner_dict = {}
    if quiz_manager is not None:
        try:
            interaction_dict = quiz_manager.interaction_state.to_dict()
            learner_dict = _learner_state_dict(quiz_manager.learner_state)
        except Exception:
            pass

    HARDCODED_RESPONSES = {
        "hand_raised":  "I see your hand. Do you have a question?",
        "face_absent":  "I cannot see you. Please come back into view.",
        "head_turned":  "Please look back at the screen.",
        "no_movement":  "You have been still for a while. Take a stretch.",
        "no_answer":    "I have not heard your answer. Please give it a try.",
    }

    import json as _json
    with open("config/settings.json") as _f:
        _settings = _json.load(_f)
    condition = _settings.get("condition", "C4")

    if condition == "C1":
        response_text = HARDCODED_RESPONSES.get(trigger,
                        "I am here to help.")
        return {
            "response_text":    response_text,
            "strategy":         "scripted",
            "interaction_state": interaction_dict,
            "learner_state":    learner_dict,
            "llm_prompt_sent":  "SCRIPTED",
            "llm_latency_sec":  0.0,
            "llm_used":         False,
            "condition":        "C1"
        }

    elif condition == "C2":
        llm_result = llm_generate(obs, trigger, mode=mode)
        llm_result["strategy"]          = "none"
        llm_result["interaction_state"] = interaction_dict
        llm_result["learner_state"]     = learner_dict
        llm_result["condition"]         = "C2"
        llm_result["llm_used"]          = True
        return llm_result

    elif condition == "C3":
        system_prompt, user_prompt = assemble_prompt(
            obs=obs,
            trigger=trigger,
            interaction=interaction_dict,
            learner=learner_dict,
            strategy="",
            strategy_description="",
            history=list(_llm_history[-3:]),
            quiz_context={}
        )
        llm_result = llm_generate(obs, trigger, mode=mode,
                        assembled_prompt=(system_prompt, user_prompt))
        llm_result["strategy"]          = "none"
        llm_result["interaction_state"] = interaction_dict
        llm_result["learner_state"]     = learner_dict
        llm_result["condition"]         = "C3"
        return llm_result

    if turn_state_override is not None:
        turn_state = turn_state_override
    elif quiz_manager is not None and hasattr(quiz_manager, "last_turn_state"):
        turn_state = quiz_manager.last_turn_state or {}
    else:
        turn_state = interaction_dict if isinstance(interaction_dict, dict) else {}

    if learner_state_override is not None:
        learner_state = learner_state_override
    else:
        learner_state = (
            quiz_manager.learner_state
            if quiz_manager is not None and hasattr(quiz_manager, "learner_state")
            else learner_dict
        )
    engagement_state = build_engagement_state(trigger)

    strategy_decision = select_strategy(turn_state, learner_state, engagement_state)
    strategy = strategy_decision["strategy"]
    strategy_desc = strategy_decision.get(
        "strategy_description",
        STRATEGY_DESCRIPTIONS.get(strategy, ""),
    )

    system_prompt, user_prompt = assemble_prompt(
        obs=obs,
        trigger=trigger,
        interaction=interaction_dict,
        learner=learner_dict,
        strategy=strategy,
        strategy_description=strategy_desc,
        history=list(_llm_history[-3:]),
        quiz_context=quiz_context or {}
    )

    llm_result = llm_generate(
        obs,
        trigger,
        mode=mode,
        assembled_prompt=(system_prompt, user_prompt)
    )
    llm_result["strategy"] = strategy
    llm_result["strategy_decision"] = strategy_decision
    llm_result["selected_strategy"] = strategy_decision.get("strategy")
    llm_result["strategy_reason"] = strategy_decision.get("reason")
    llm_result["strategy_description"] = strategy_decision.get("strategy_description")
    llm_result["strategy_concept"] = strategy_decision.get("concept")
    llm_result["strategy_weak_concept"] = strategy_decision.get("weak_concept")
    llm_result["interaction_state"] = interaction_dict
    llm_result["learner_state"] = learner_dict
    llm_result["condition"] = "C4"
    return llm_result


def _handle_post_question_strategy(log, state, quiz, result: dict,
                                   fallback_obs: dict) -> None:
    answer_status = result.get("answer_status")
    if answer_status not in {"correct", "incorrect", "no_answer"}:
        return

    trigger = "no_answer" if answer_status == "no_answer" else "quiz_answer"
    turn_state = {
        "trigger": trigger,
        "concept": result.get("concept"),
        "answer_status": answer_status,
        "response_behavior": result.get("response_behavior"),
        "response_time_sec": result.get("response_time_sec"),
        "question_text": result.get("question_text"),
        "student_answer": result.get("student_answer"),
        "is_correct": result.get("is_correct"),
    }
    learner_dict = _learner_state_dict(quiz.learner_state) if quiz is not None else {}
    obs = state.get_latest() or fallback_obs or {}
    quiz_context = {
        "concept": result.get("concept", ""),
        "question_text": result.get("question_text", ""),
    }

    llm_result = _fire_llm_with_strategy(
        obs,
        trigger,
        mode="quiz",
        quiz_manager=quiz,
        quiz_context=quiz_context,
        turn_state_override=turn_state,
        learner_state_override=learner_dict,
    )
    response = llm_result["response_text"]
    speak(response)
    log.log_engagement_event({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trigger": trigger,
        "response_text": response,
        "llm_used": llm_result.get("llm_used"),
        "llm_latency_sec": llm_result.get("llm_latency_sec"),
        "llm_prompt_sent": llm_result.get("llm_prompt_sent", ""),
        "llm_raw_response": llm_result.get("llm_raw_response", ""),
        "llm_retry_log": llm_result.get("llm_retry_log", []),
        "llm_backend": llm_result.get("llm_backend", "gpu_ssh"),
        "strategy": llm_result.get("strategy", ""),
        **_strategy_log_fields(llm_result),
        "interaction_state": llm_result.get("interaction_state", {}),
        "learner_state": llm_result.get("learner_state", {}),
        "condition": llm_result.get("condition"),
    })


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_obs: dict | None = None
        self._obs_count: int = 0
        self._new_obs_ready = threading.Event()
        self.running = True
        self.latest_frame = None  # latest rendered frame for main-thread display

        self.interrupt_event = threading.Event()
        self.interrupt_reason = ""
        self.pending_alert_text = ""

        self._alert_cooldown: dict[str, float] = {}

        self.detection_time: float = 0.0
        self.obs_snapshot: dict | None = None


        self._hand_raise_start: float = 0.0
        self._hand_last_seen: float = 0.0
        self._hand_tracking: bool = False

        self._face_absent_start: float = 0.0
        self._face_tracking_absent: bool = False

    def push_observation(self, obs_dict: dict) -> None:
        with self._lock:
            self._latest_obs = obs_dict
            self._obs_count += 1
        self._new_obs_ready.set()

    def pop_observation(self, timeout: float = 1.0) -> dict | None:
        if self._new_obs_ready.wait(timeout=timeout):
            self._new_obs_ready.clear()
            with self._lock:
                return self._latest_obs
        return None

    def get_latest(self) -> dict | None:
        with self._lock:
            return self._latest_obs

    @property
    def obs_count(self) -> int:
        with self._lock:
            return self._obs_count

    def cooldown_ok(self, trigger: str) -> bool:
        last = self._alert_cooldown.get(trigger, 0.0)
        return (time.time() - last) >= _COOLDOWN_SEC

    def fire_cooldown(self, trigger: str) -> None:
        self._alert_cooldown[trigger] = time.time()

    def request_interrupt(self, reason: str, alert_text: str = "",
                          obs_snapshot: dict | None = None) -> None:
        self.interrupt_reason = reason
        self.pending_alert_text = alert_text
        self.obs_snapshot = obs_snapshot or {}
        self.interrupt_event.set()


    def update_hand_raise(self, hand_raised: bool) -> float:
        now = time.time()
        if hand_raised:
            self._hand_last_seen = now
            if not self._hand_tracking:
                self._hand_raise_start = now
                self._hand_tracking = True
            return now - self._hand_raise_start
        else:
            if self._hand_tracking:
                gap = now - self._hand_last_seen
                if gap > 1.0:
                    self._hand_tracking = False
                    self._hand_raise_start = 0.0
                    return 0.0
                else:
                    return now - self._hand_raise_start
            return 0.0


    def update_face_absent(self, face_detected: bool) -> float:
        now = time.time()
        if face_detected:
            self._face_tracking_absent = False
            self._face_absent_start = 0.0
            return 0.0
        else:
            if not self._face_tracking_absent:
                self._face_absent_start = now
                self._face_tracking_absent = True
            return now - self._face_absent_start


def camera_thread(cap, tracker, builder, log, state: SharedState):
    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_data = process_frame_data(rgb, w, h, tracker)
        builder.add_frame(frame_data)

        face = frame_data.get("face_detected", False)
        hand = frame_data.get("hand_raised", False)


        hand_dur = state.update_hand_raise(hand)
        if hand_dur >= _HAND_RAISE_SEC and state.cooldown_ok("hand_raised"):
            state.fire_cooldown("hand_raised")
            state.detection_time = time.time()
            print(f"  [camera {time.strftime('%H:%M:%S')}] "
                  f"Hand raised detected ({hand_dur:.1f}s sustained)")
            state.request_interrupt("hand_raised", obs_snapshot={
                "face_detected":    face,
                "gaze_on_robot":    frame_data.get("gaze", 0.0),
                "head_yaw_deg":     frame_data.get("yaw", 0.0),
                "head_pitch_deg":   frame_data.get("pitch", 0.0),
                "head_moving":      frame_data.get("head_moving", False),
                "body_detected":    frame_data.get("body_detected", False),
                "no_movement_sec":  frame_data.get("no_movement_sec", 0.0),
                "speech_energy":    "low",
                "hand_raised":      True,
                "still_there_prompt": False,
                "hand_raise_side":  frame_data.get("hand_raise_side", "none"),
            })
            state._hand_tracking = False
            state._hand_raise_start = 0.0

        face_absent_dur = state.update_face_absent(face)
        if face_absent_dur >= _FACE_ABSENT_SEC and state.cooldown_ok("face_absent"):
            state.fire_cooldown("face_absent")
            state.detection_time = time.time()
            print(f"  [camera {time.strftime('%H:%M:%S')}] "
                  f"Face absent detected ({face_absent_dur:.1f}s)")
            state.request_interrupt("alert", "face_absent", obs_snapshot={
                "face_detected": False, "gaze_on_robot": 0.0,
                "head_yaw_deg": 0.0, "head_pitch_deg": 0.0,
                "head_moving": False, "body_detected": False,
                "no_movement_sec": 0.0, "speech_energy": "low",
                "hand_raised": False, "still_there_prompt": True,
                "hand_raise_side": "none",
            })

        if builder.ready():
            obs = builder.build()
            obs_dict = obs.to_dict()
            log.log_observation(obs_dict)
            state.push_observation(obs_dict)

            gaze   = obs_dict.get("gaze_on_robot", 0.0)
            no_mov = obs_dict.get("no_movement_sec", 0.0)
            obs_face = obs_dict.get("face_detected", False)

            if obs_face and no_mov >= _NO_MOVEMENT_SEC and state.cooldown_ok("no_movement"):
                state.fire_cooldown("no_movement")
                state.detection_time = time.time()
                print(f"  [camera {time.strftime('%H:%M:%S')}] "
                      f"No movement detected ({no_mov:.1f}s)")
                state.request_interrupt("alert", "no_movement", obs_snapshot=obs_dict)

            elif obs_face and gaze < _GAZE_AWAY_THRESH and state.cooldown_ok("head_turned"):
                state.fire_cooldown("head_turned")
                state.detection_time = time.time()
                print(f"  [camera {time.strftime('%H:%M:%S')}] "
                      f"Head turned detected (gaze={gaze:.2f})")
                state.request_interrupt("alert", "head_turned", obs_snapshot=obs_dict)

        if SHOW_WINDOW:
            countdown = max(0.0, OBSERVATION_INTERVAL_SEC -
                            (time.time() - builder._last_obs_time))
            draw_overlay(frame, frame_data, countdown)
            state.latest_frame = frame.copy()


_WINDOW_NAME = "Marty — Edu Robot"


def phase1_camera_check(cap, tracker, builder, log, state: SharedState) -> str:
    """Runs in pipeline thread. Writes frames to state.latest_frame so the
    main-thread display loop stays alive throughout."""
    print("\n" + "=" * 55)
    print("  PHASE 1 — Camera check (10 seconds)")
    print("=" * 55)

    saw_face = False
    saw_hand = False
    saw_head_move = False
    start = time.time()

    while time.time() - start < 10.0:
        if not state.running:
            return "quit"
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame.")
            return "quit"

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_data = process_frame_data(rgb, w, h, tracker)
        builder.add_frame(frame_data)

        if frame_data["face_detected"]:
            saw_face = True
        if frame_data["hand_raised"]:
            saw_hand = True
        if frame_data["head_moving"]:
            saw_head_move = True

        if builder.ready():
            obs = builder.build()
            log.log_observation(obs.to_dict())
            print(f"  [phase1] Obs: face={obs.face_detected} "
                  f"hand={obs.hand_raised} head_moving={obs.head_moving}")

        if SHOW_WINDOW:
            countdown = max(0.0, OBSERVATION_INTERVAL_SEC -
                            (time.time() - builder._last_obs_time))
            draw_overlay(frame, frame_data, countdown)
            state.latest_frame = frame.copy()

    print("\n  Phase 1 results:")
    print(f"    Face detected    : {'YES' if saw_face else 'NO'}")
    print(f"    Hand raise       : {'YES' if saw_hand else 'NO'}")
    print(f"    Head movement    : {'YES' if saw_head_move else 'NO'}")

    if saw_face:
        print("\n  Camera OK — face detected.")
    else:
        print("\n  WARNING: No face detected during warm-up.")

    speak("Camera check complete. Are you ready to start the quiz? Say yes or no.")
    result = listen_for_answer(timeout_sec=10)
    answer = result["transcript"].strip().lower()
    print(f"  Heard: '{answer}'")

    if _is_yes(answer):
        speak("Great!")
        return "quiz"
    elif answer == "":
        speak("I didn't hear you. Say yes to start or no to just watch.")
        result = listen_for_answer(timeout_sec=10)
        answer = result["transcript"].strip().lower()
        return "quiz" if _is_yes(answer) else "monitor"
    else:
        speak("No problem! I'll just keep watching for now.")
        return "monitor"


def handle_hand_raise(log, obs: dict | None = None,
                      mode: str = "quiz",
                      state: "SharedState | None" = None,
                      interaction_dict: dict | None = None,
                      learner_dict: dict | None = None,
                      quiz_context: dict | None = None,
                      correct_answer: str = "",
                      accepted_answers: list = None) -> None:
    if obs is None:
        obs = {}
    interaction_dict = interaction_dict or {}
    learner_dict = learner_dict or {}
    quiz_context = quiz_context or {
        "question_text": "",
        "difficulty_level": "",
        "concept": ""
    }
    accepted_answers = accepted_answers or []

    import json as _json_hr
    with open("config/settings.json") as _f_hr:
        _settings_hr = _json_hr.load(_f_hr)
    _condition = _settings_hr.get("condition", "C4")

    def _hand_help_event(spoken_request: str = "") -> dict:
        return build_hand_raise_help_event(
            spoken_request=spoken_request,
            question_text=quiz_context.get("question_text", ""),
            concept=quiz_context.get("concept", ""),
        )

    def _hand_strategy_fields(spoken_request: str = "") -> dict:
        help_event = _hand_help_event(spoken_request)
        return {
            "strategy": help_event.get("response_policy"),
            "selected_strategy": help_event.get("response_policy"),
            "strategy_reason": "learner raised a hand",
            "strategy_description": "Learner-initiated hand-raise help request.",
            "strategy_concept": quiz_context.get("concept") or None,
            "strategy_weak_concept": learner_dict.get("weak_concept"),
            "spoken_request": help_event.get("spoken_request", spoken_request),
            "help_request_type": help_event.get("help_request_type"),
            "response_policy": help_event.get("response_policy"),
            "routes_to_strategy": help_event.get("routes_to_strategy"),
            "condition": _condition,
        }

    if _condition == "C1":
        speak("I see your hand. Do you have a question?")
        log.log_engagement_event({
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
            "trigger":           "hand_raised",
            "strategy":          "scripted",
            "student_question":  "",
            "response_text":     "I see your hand. Do you have a question?",
            "llm_prompt_sent":   "SCRIPTED",
            "llm_used":          False,
            "llm_latency_sec":   0.0,
            "interaction_state": interaction_dict or {},
            "learner_state":     learner_dict or {},
            "condition":         "C1"
        })
        return

    elif _condition in ("C2", "C3"):
        speak("I see your hand. Do you have a question?")
        listen_result = listen_for_answer(timeout_sec=10)
        student_question = listen_result.get("transcript", "").strip()

        if student_question:
            llm_ans = llm_generate(obs, "hand_raised", mode=mode)
            answer_text = llm_ans.get("response_text",
                          "Good question! Keep exploring Python.")
        else:
            answer_text = "No worries, feel free to ask whenever ready."

        speak(answer_text)
        log.log_engagement_event({
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
            "trigger":           "hand_raised",
            "strategy":          "none",
            "student_question":  student_question,
            "response_text":     answer_text,
            "llm_prompt_sent":   "hand_raised basic",
            "llm_used":          True if student_question else False,
            "llm_latency_sec":   0.0,
            "interaction_state": interaction_dict or {},
            "learner_state":     learner_dict or {},
            "condition":         _condition
        })
        return

    def _is_no_or_empty(text):
        text = text.strip().lower()
        if not text:
            return True
        no_words = ["no", "nope", "nothing", "never mind", "nevermind",
                    "i'm fine", "im fine", "all good", "forget it",
                    "not really", "no thanks", "nah"]
        return any(text.startswith(w) for w in no_words)

    def _is_direct_question(text):
        text = text.strip().lower()
        question_words = ["what", "why", "how", "when", "where", "which",
                          "who", "can", "is", "are", "does", "do",
                          "explain", "tell me", "show me"]
        return any(text.startswith(w) for w in question_words)

    def _suppress_alerts():
        if state is not None:
            state.interrupt_event.clear()
            state.fire_cooldown("no_movement")
            state.fire_cooldown("head_turned")
            state.fire_cooldown("face_absent")

    # STEP 1 — Opening
    opening_system = (
        "You are an encouraging programming tutor robot. The student has raised "
        "their hand. Generate a single warm opening line that acknowledges their "
        "hand raise and invites them to share a question or thought. "
        "Vary the phrasing each time. Respond in 5 to 15 words only."
    )
    opening_user = "Generate the opening line now."
    llm_open = llm_generate(obs, "hand_raised", mode=mode,
                            assembled_prompt=(opening_system, opening_user))
    opening_text = llm_open["response_text"]
    if opening_text.strip().upper() == "NONE":
        opening_text = "I see your hand is up — what's on your mind?"
    speak(opening_text)

    # STEP 2 — Listen
    listen1 = listen_for_answer(timeout_sec=10)
    transcript = listen1["transcript"]
    print(f"  [hand raise] Heard: '{transcript}'")

    student_question = ""

    # STEP 3 — Student said no / nothing
    if _is_no_or_empty(transcript):
        closing_system = "You are an encouraging programming tutor robot."
        closing_user = (
            "The student said they have no question. Generate a brief "
            "encouraging line to continue. 5 to 15 words."
        )
        llm_close = llm_generate(obs, "hand_raised", mode=mode,
                                 assembled_prompt=(closing_system, closing_user))
        closing_text = llm_close["response_text"]
        if closing_text.strip().upper() == "NONE":
            closing_text = "All good — let's keep going!"
        speak(closing_text)

        _suppress_alerts()
        log.log_engagement_event({
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
            "trigger":           "hand_raised",
            **_hand_strategy_fields(transcript),
            "student_question":  "",
            "response_text":     closing_text,
            "interaction_state": interaction_dict,
            "learner_state":     learner_dict,
            "obs":               obs,
        })
        return

    # STEP 4 — Direct question or confirmed yes?
    if _is_direct_question(transcript):
        student_question = transcript
        print(f"  [hand raise] Student asked directly: '{student_question}'")
    else:
        # STEP 5 — Ask them to go ahead, then listen again
        followup_system = "You are an encouraging programming tutor robot."
        followup_user = (
            "The student confirmed they have a question. "
            "Invite them to go ahead and ask it. 5 to 15 words."
        )
        llm_fu = llm_generate(obs, "hand_raised", mode=mode,
                              assembled_prompt=(followup_system, followup_user))
        followup_text = llm_fu["response_text"]
        if followup_text.strip().upper() == "NONE":
            followup_text = "Go ahead — what's your question?"
        speak(followup_text)

        listen2 = listen_for_answer(timeout_sec=15)
        student_question = listen2["transcript"]
        print(f"  [hand raise] Student question: '{student_question}'")

        if not student_question.strip():
            fallback_text = "No worries, let's continue!"
            speak(fallback_text)

            _suppress_alerts()
            log.log_engagement_event({
                "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
                "trigger":           "hand_raised",
                **_hand_strategy_fields(student_question),
                "student_question":  "",
                "response_text":     fallback_text,
                "interaction_state": interaction_dict,
                "learner_state":     learner_dict,
                "obs":               obs,
            })
            return

    # STEP 6 — Answer the student question
    answer_system = (
        "You are an encouraging Python programming tutor helping an adult CS "
        "student. The student has asked a question during a programming quiz. "
        "Answer their question clearly and concisely. "
        "The quiz covers these topics: variables, data types, loops, "
        "conditionals, functions, recursion. "
        "The student may ask about these topics or general Python concepts. "
        "Keep your answer between 20 and 60 words. "
        "Be accurate, encouraging, and peer-level in tone. "
        "Do not mention sensor data or technical pipeline details. "
        "You must never reveal or hint at the correct answer to the "
        "current quiz question even if the student asks about the same topic."
    )
    banned = ", ".join(accepted_answers) if accepted_answers else correct_answer
    current_q = ""
    if quiz_context.get("question_text"):
        current_q = (
            f"Note: The student is currently working on this quiz question: "
            f"{quiz_context.get('question_text')} "
            f"(concept: {quiz_context.get('concept', '')}). "
            f"DO NOT mention or hint at any of these words or symbols in "
            f"your reply: {banned}. "
            f"If their question is related, explain the concept generally "
            f"without using those specific terms."
        )

    answer_user = f"{current_q}\nStudent question: {student_question}"
    llm_answer = llm_generate(obs, "hand_raised", mode=mode,
                              assembled_prompt=(answer_system, answer_user))
    answer_text = llm_answer["response_text"]
    if answer_text.strip().upper() == "NONE":
        answer_text = "Great question — let's tackle it together!"
    speak(answer_text)

    _suppress_alerts()

    # STEP 7 — Log
    log.log_engagement_event({
        "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trigger":           "hand_raised",
        **_hand_strategy_fields(student_question),
        "student_question":  student_question,
        "response_text":     answer_text,
        "llm_prompt_sent":   answer_user,
        "llm_latency_sec":   llm_answer.get("llm_latency_sec"),
        "llm_used":          llm_answer.get("llm_used"),
        "interaction_state": interaction_dict,
        "learner_state":     learner_dict,
        "obs":               obs,
    })


def phase234_main_loop(cap, tracker, builder, log, state: SharedState):
    _quiz_ref = None
    quiz  = QuizManager(
        start_level=1,
        max_questions=MAX_QUESTIONS,
        enable_builtin_feedback=False,
    )
    _quiz_ref = quiz
    engagement_manager = EngagementManager()
    engagement_manager.set_state_refs(
        quiz.interaction_state,
        quiz.learner_state
    )

    cam_thread = threading.Thread(
        target=camera_thread,
        args=(cap, tracker, builder, log, state),
        daemon=True,
        name="camera-thread",
    )
    cam_thread.start()

    def _quiz_worker():
        resume_question = False

        print("\n" + "=" * 60)
        print(f"  PHASE 2+3 — Quiz ({MAX_QUESTIONS} questions) + Engagement")
        print("  Camera runs continuously in background")
        print("  Hand raise 3s → interrupt | Face absent 3s → alert")
        print("  Press Q in camera window to quit early")
        print("=" * 60)

        speak(f"Let's begin. I'll ask you {MAX_QUESTIONS} programming questions.")

        try:
            while state.running and not quiz.is_done:
                obs_dict = state.get_latest() or {}
                face_now = obs_dict.get("face_detected", False)

                print(f"\n  [quiz {quiz.questions_asked + 1}/{MAX_QUESTIONS}]")

                if state.interrupt_event.is_set():
                    reason = state.interrupt_reason
                    alert_text = state.pending_alert_text
                    state.interrupt_event.clear()

                    if reason == "hand_raised":
                        gap = time.time() - state.detection_time
                        print(f"  [pending {time.strftime('%H:%M:%S')}] "
                              f"Hand raised — LLM triggered "
                              f"({gap:.1f}s after detection)")
                        latest_obs = state.obs_snapshot or state.get_latest() or {}
                        handle_hand_raise(
                            log,
                            latest_obs,
                            mode="quiz",
                            state=state,
                            interaction_dict=_quiz_ref.interaction_state.to_dict() if _quiz_ref else {},
                            learner_dict=_quiz_ref.learner_state.to_dict() if _quiz_ref else {},
                            quiz_context={
                                "question_text": _quiz_ref._current_question.get("question_text", "") if _quiz_ref and _quiz_ref._current_question else "",
                                "difficulty_level": _quiz_ref.level if _quiz_ref else "",
                                "concept": _quiz_ref._current_question.get("concept", "") if _quiz_ref and _quiz_ref._current_question else ""
                            },
                            correct_answer=_quiz_ref._current_question.get("correct_answer", "") if _quiz_ref and _quiz_ref._current_question else "",
                            accepted_answers=_quiz_ref._current_question.get("accepted_answers", []) if _quiz_ref and _quiz_ref._current_question else []
                        )
                        state.obs_snapshot = None
                        resume_question = True
                        continue
                    elif reason == "alert" and alert_text:
                        gap = time.time() - state.detection_time
                        print(f"  [pending {time.strftime('%H:%M:%S')}] "
                              f"Alert ({alert_text}) — LLM triggered "
                              f"({gap:.1f}s after detection)")
                        latest_obs = state.obs_snapshot or state.get_latest() or {}
                        llm_result = _fire_llm_with_strategy(latest_obs, alert_text,
                                                              mode="quiz",
                                                              quiz_manager=_quiz_ref)
                        response = llm_result["response_text"]
                        print(f"  [llm->speak {time.strftime('%H:%M:%S')}] "
                              f"'{response}'")
                        speak(response)
                        log.log_engagement_event({
                            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "trigger":         alert_text,
                            "response_text":   response,
                            "llm_used":        llm_result["llm_used"],
                            "llm_latency_sec": llm_result["llm_latency_sec"],
                            "llm_prompt_sent":  llm_result.get("llm_prompt_sent", ""),
                            "llm_raw_response": llm_result.get("llm_raw_response", ""),
                            "llm_retry_log":    llm_result.get("llm_retry_log", []),
                            "llm_backend":      llm_result.get("llm_backend", "gpu_ssh"),
                            "strategy":          llm_result.get("strategy", ""),
                            **_strategy_log_fields(llm_result),
                            "interaction_state": llm_result.get("interaction_state", {}),
                            "learner_state":     llm_result.get("learner_state", {}),
                        })
                        state.obs_snapshot = None
                        resume_question = True
                        continue

                state.interrupt_event.clear()
                state.interrupt_reason = ""

                result = quiz.ask_next_question(
                    listen_timeout=15,
                    face_detected=face_now,
                    resume=resume_question,
                    stop_event=state.interrupt_event,
                )
                resume_question = False

                if result.get("interrupted"):
                    reason = state.interrupt_reason
                    alert_text = state.pending_alert_text
                    state.interrupt_event.clear()

                    if reason == "hand_raised":
                        gap = time.time() - state.detection_time
                        print(f"  [interrupt {time.strftime('%H:%M:%S')}] "
                              f"Hand raised — LLM triggered "
                              f"({gap:.1f}s after detection)")
                        latest_obs = state.obs_snapshot or state.get_latest() or {}
                        handle_hand_raise(
                            log,
                            latest_obs,
                            mode="quiz",
                            state=state,
                            interaction_dict=_quiz_ref.interaction_state.to_dict() if _quiz_ref else {},
                            learner_dict=_quiz_ref.learner_state.to_dict() if _quiz_ref else {},
                            quiz_context={
                                "question_text": _quiz_ref._current_question.get("question_text", "") if _quiz_ref and _quiz_ref._current_question else "",
                                "difficulty_level": _quiz_ref.level if _quiz_ref else "",
                                "concept": _quiz_ref._current_question.get("concept", "") if _quiz_ref and _quiz_ref._current_question else ""
                            },
                            correct_answer=_quiz_ref._current_question.get("correct_answer", "") if _quiz_ref and _quiz_ref._current_question else "",
                            accepted_answers=_quiz_ref._current_question.get("accepted_answers", []) if _quiz_ref and _quiz_ref._current_question else []
                        )
                        state.obs_snapshot = None
                        resume_question = True
                        continue

                    elif reason == "alert":
                        gap = time.time() - state.detection_time
                        print(f"  [interrupt {time.strftime('%H:%M:%S')}] "
                              f"Alert ({alert_text}) — LLM triggered "
                              f"({gap:.1f}s after detection)")
                        latest_obs = state.obs_snapshot or state.get_latest() or {}
                        llm_result = _fire_llm_with_strategy(latest_obs, alert_text,
                                                              mode="quiz",
                                                              quiz_manager=_quiz_ref)
                        response = llm_result["response_text"]
                        print(f"  [llm→speak {time.strftime('%H:%M:%S')}] "
                              f"'{response}'")
                        speak(response)
                        log.log_engagement_event({
                            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "trigger":         alert_text,
                            "response_text":   response,
                            "llm_used":        llm_result["llm_used"],
                            "llm_latency_sec": llm_result["llm_latency_sec"],
                            "llm_prompt_sent":  llm_result.get("llm_prompt_sent", ""),
                            "llm_raw_response": llm_result.get("llm_raw_response", ""),
                            "llm_retry_log":    llm_result.get("llm_retry_log", []),
                            "llm_backend":      llm_result.get("llm_backend", "gpu_ssh"),
                            "strategy":          llm_result.get("strategy", ""),
                            **_strategy_log_fields(llm_result),
                            "interaction_state": llm_result.get("interaction_state", {}),
                            "learner_state":     llm_result.get("learner_state", {}),
                        })
                        state.obs_snapshot = None
                        resume_question = True
                        continue

                log.log_quiz_result(result)
                _handle_post_question_strategy(log, state, _quiz_ref, result, obs_dict)

                latest = state.get_latest() or obs_dict
                if not latest.get("face_detected", False):
                    if result["student_answer"]:
                        face_absent_obs = {
                            "face_detected": False, "gaze_on_robot": 0.0,
                            "head_yaw_deg": 0.0, "head_pitch_deg": 0.0,
                            "head_moving": False, "body_detected": False,
                            "no_movement_sec": 0.0, "speech_energy": "low",
                            "hand_raised": False, "still_there_prompt": True,
                            "hand_raise_side": "none",
                        }
                        llm_fa = _fire_llm_with_strategy(face_absent_obs, "face_absent",
                                                          mode="quiz", quiz_manager=_quiz_ref)
                        speak(llm_fa["response_text"])
                        log.log_engagement_event({
                            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "trigger":         "face_absent_answer",
                            "response_text":   llm_fa["response_text"],
                            "answer_still_checked": True,
                            "llm_used":        llm_fa["llm_used"],
                            "llm_latency_sec": llm_fa["llm_latency_sec"],
                            "llm_prompt_sent":  llm_fa.get("llm_prompt_sent", ""),
                            "llm_raw_response": llm_fa.get("llm_raw_response", ""),
                            "llm_retry_log":    llm_fa.get("llm_retry_log", []),
                            "llm_backend":      llm_fa.get("llm_backend", "gpu_ssh"),
                            "strategy":          llm_fa.get("strategy", ""),
                            **_strategy_log_fields(llm_fa),
                            "interaction_state": llm_fa.get("interaction_state", {}),
                            "learner_state":     llm_fa.get("learner_state", {}),
                        })

                log.save()

        except KeyboardInterrupt:
            print("\nInterrupted by user.")

        if quiz.is_done:
            speak(f"That's all {MAX_QUESTIONS} questions done! Great effort!")
        else:
            speak("Quiz ended. Well done!")

        log.summarise()
        log.save()

        print("\n" + "=" * 60)
        print("  PHASE 5 — Post-quiz camera monitoring")
        print("  Camera continues observing every 5s")
        print("  Engagement alerts still active")
        print("  Press Q in camera window to exit")
        print("=" * 60)

        speak("Quiz is done. I'll keep watching. Press Q to exit.")

        try:
            while state.running:
                obs_dict = state.pop_observation(timeout=2.0)
                if obs_dict is None:
                    continue

                face = obs_dict.get("face_detected", False)
                no_mov = obs_dict.get("no_movement_sec", 0.0)
                print(f"  [monitor] face={face} no_mov={no_mov:.1f}s "
                      f"obs={state.obs_count}")

                if state.interrupt_event.is_set():
                    reason = state.interrupt_reason
                    alert_text = state.pending_alert_text
                    state.interrupt_event.clear()

                    if reason == "alert" and alert_text:
                        gap = time.time() - state.detection_time
                        print(f"  [post-quiz {time.strftime('%H:%M:%S')}] "
                              f"Alert ({alert_text}) — LLM triggered "
                              f"({gap:.1f}s after detection)")
                        latest_obs = state.obs_snapshot or state.get_latest() or {}
                        llm_result = _fire_llm_with_strategy(latest_obs, alert_text,
                                                              mode="monitor",
                                                              quiz_manager=_quiz_ref)
                        response = llm_result["response_text"]
                        print(f"  [llm→speak {time.strftime('%H:%M:%S')}] "
                              f"'{response}'")
                        speak(response)
                        log.log_engagement_event({
                            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "trigger":         "post_quiz_" + alert_text,
                            "response_text":   response,
                            "llm_used":        llm_result["llm_used"],
                            "llm_latency_sec": llm_result["llm_latency_sec"],
                            "llm_prompt_sent":  llm_result.get("llm_prompt_sent", ""),
                            "llm_raw_response": llm_result.get("llm_raw_response", ""),
                            "llm_retry_log":    llm_result.get("llm_retry_log", []),
                            "llm_backend":      llm_result.get("llm_backend", "gpu_ssh"),
                            "strategy":          llm_result.get("strategy", ""),
                            **_strategy_log_fields(llm_result),
                            "interaction_state": llm_result.get("interaction_state", {}),
                            "learner_state":     llm_result.get("learner_state", {}),
                        })
                        state.obs_snapshot = None
                    elif reason == "hand_raised":
                        gap = time.time() - state.detection_time
                        print(f"  [post-quiz {time.strftime('%H:%M:%S')}] "
                              f"Hand raised — LLM triggered "
                              f"({gap:.1f}s after detection)")
                        latest_obs = state.obs_snapshot or state.get_latest() or {}
                        handle_hand_raise(
                        log,
                        latest_obs,
                        mode="monitor",
                        state=state,
                        interaction_dict=_quiz_ref.interaction_state.to_dict() if _quiz_ref else {},
                        learner_dict=_quiz_ref.learner_state.to_dict() if _quiz_ref else {},
                        quiz_context={
                            "question_text": _quiz_ref._current_question.get("question_text", "") if _quiz_ref and _quiz_ref._current_question else "",
                            "difficulty_level": _quiz_ref.level if _quiz_ref else "",
                            "concept": _quiz_ref._current_question.get("concept", "") if _quiz_ref and _quiz_ref._current_question else ""
                        },
                        correct_answer=_quiz_ref._current_question.get("correct_answer", "") if _quiz_ref and _quiz_ref._current_question else "",
                        accepted_answers=_quiz_ref._current_question.get("accepted_answers", []) if _quiz_ref and _quiz_ref._current_question else []
                    )
                        state.obs_snapshot = None

                    log.save()

        except KeyboardInterrupt:
            print("\nExiting.")

        state.running = False
        log.save()

    quiz_thread = threading.Thread(target=_quiz_worker, daemon=True, name="quiz-thread")
    quiz_thread.start()
    quiz_thread.join()   # display loop is in main(); just wait here
    state.running = False
    cam_thread.join(timeout=3.0)
    log.save()


def phase5_monitor_only(cap, tracker, builder, log, state: SharedState):
    _quiz_ref = None
    cam_thread = threading.Thread(
        target=camera_thread,
        args=(cap, tracker, builder, log, state),
        daemon=True,
        name="camera-thread",
    )
    cam_thread.start()

    print("\n" + "=" * 60)
    print("  MONITOR MODE — Camera monitoring only (no quiz)")
    print("  Engagement alerts active")
    print("  Press Q in camera window to exit")
    print("=" * 60)

    speak("I'll just keep watching for now. Press Q whenever you want to stop.")

    try:
        while state.running:
            obs_dict = state.pop_observation(timeout=2.0)
            if obs_dict is None:
                continue

            face = obs_dict.get("face_detected", False)
            no_mov = obs_dict.get("no_movement_sec", 0.0)
            print(f"  [monitor] face={face} no_mov={no_mov:.1f}s "
                  f"obs={state.obs_count}")

            if state.interrupt_event.is_set():
                reason = state.interrupt_reason
                alert_text = state.pending_alert_text
                state.interrupt_event.clear()

                if reason == "alert" and alert_text:
                    gap = time.time() - state.detection_time
                    print(f"  [monitor {time.strftime('%H:%M:%S')}] "
                          f"Alert ({alert_text}) — LLM triggered "
                          f"({gap:.1f}s after detection)")
                    latest_obs = state.obs_snapshot or state.get_latest() or {}
                    llm_result = _fire_llm_with_strategy(latest_obs, alert_text,
                                                          mode="monitor",
                                                          quiz_manager=_quiz_ref)
                    response = llm_result["response_text"]
                    print(f"  [llm→speak {time.strftime('%H:%M:%S')}] "
                          f"'{response}'")
                    speak(response)
                    log.log_engagement_event({
                        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "trigger":         "monitor_" + alert_text,
                        "response_text":   response,
                        "llm_used":        llm_result["llm_used"],
                        "llm_latency_sec": llm_result["llm_latency_sec"],
                        "llm_prompt_sent":  llm_result.get("llm_prompt_sent", ""),
                        "llm_raw_response": llm_result.get("llm_raw_response", ""),
                        "llm_retry_log":    llm_result.get("llm_retry_log", []),
                        "llm_backend":      llm_result.get("llm_backend", "gpu_ssh"),
                        "strategy":          llm_result.get("strategy", ""),
                        **_strategy_log_fields(llm_result),
                        "interaction_state": llm_result.get("interaction_state", {}),
                        "learner_state":     llm_result.get("learner_state", {}),
                    })
                    state.obs_snapshot = None
                elif reason == "hand_raised":
                    gap = time.time() - state.detection_time
                    print(f"  [monitor {time.strftime('%H:%M:%S')}] "
                          f"Hand raised — LLM triggered "
                          f"({gap:.1f}s after detection)")
                    latest_obs = state.obs_snapshot or state.get_latest() or {}
                    handle_hand_raise(
                        log,
                        latest_obs,
                        mode="monitor",
                        state=state,
                        interaction_dict=_quiz_ref.interaction_state.to_dict() if _quiz_ref else {},
                        learner_dict=_quiz_ref.learner_state.to_dict() if _quiz_ref else {},
                        quiz_context={
                            "question_text": _quiz_ref._current_question.get("question_text", "") if _quiz_ref and _quiz_ref._current_question else "",
                            "difficulty_level": _quiz_ref.level if _quiz_ref else "",
                            "concept": _quiz_ref._current_question.get("concept", "") if _quiz_ref and _quiz_ref._current_question else ""
                        },
                        correct_answer=_quiz_ref._current_question.get("correct_answer", "") if _quiz_ref and _quiz_ref._current_question else "",
                        accepted_answers=_quiz_ref._current_question.get("accepted_answers", []) if _quiz_ref and _quiz_ref._current_question else []
                    )
                    state.obs_snapshot = None

                log.save()

    except KeyboardInterrupt:
        print("\nExiting.")

    state.running = False
    cam_thread.join(timeout=3.0)
    log.save()


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera. Check CAMERA_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    tracker = MovementTracker()
    builder = ObservationBuilder()
    log     = SessionLogger(log_dir=".")

    # Shared state created here so the main-thread display loop can read frames
    # written by any phase or camera thread.
    state = SharedState()

    # Ctrl+C sets state.running = False so the pipeline thread exits cleanly.
    def _sigint_handler(sig, _frame):
        print("\nExiting.")
        state.running = False
    signal.signal(signal.SIGINT, _sigint_handler)

    print("=" * 60)
    print("  Marty Edu Robot — Main Pipeline (threaded)")
    print("=" * 60)

    if SHOW_WINDOW:
        cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)

    def _pipeline():
        try:
            mode = phase1_camera_check(cap, tracker, builder, log, state)
            if state.running and mode != "quit":
                if mode == "quiz":
                    phase234_main_loop(cap, tracker, builder, log, state)
                else:
                    phase5_monitor_only(cap, tracker, builder, log, state)
        except Exception as e:
            print(f"\nUnexpected error in pipeline: {e}")
        finally:
            state.running = False  # signal main thread to exit display loop

    pipeline_thread = threading.Thread(target=_pipeline, daemon=True, name="pipeline")
    pipeline_thread.start()

    # Main thread does nothing but pump Qt events — every phase writes frames
    # to state.latest_frame; display loop picks them up here continuously.
    while state.running or pipeline_thread.is_alive():
        if SHOW_WINDOW and state.latest_frame is not None:
            cv2.imshow(_WINDOW_NAME, state.latest_frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            state.running = False

    pipeline_thread.join(timeout=10.0)

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    cap.release()
    try:
        face_landmarker.close()
        pose_landmarker.close()
    except Exception:
        pass
    log.save()
    print("\nSession ended. Goodbye!")


if __name__ == "__main__":
    main()
