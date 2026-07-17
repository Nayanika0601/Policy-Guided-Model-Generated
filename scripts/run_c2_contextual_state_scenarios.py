import argparse
import copy
import json
import random
import time
import requests
import datetime
from pathlib import Path
import sys
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hand_raise_manager import handle_hand_raise


CONDITION = "C2"
CONDITION_NAME = "Contextual State Prompting"
CONCEPTS = ["variables", "data_types", "loops", "conditionals"]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "C2_contextual_state"

_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"
try:
    _SETTINGS = json.loads(_SETTINGS_PATH.read_text())
except Exception:
    _SETTINGS = {}

OLLAMA_URL = _SETTINGS.get("ollama_url", "http://localhost:11434/api/generate")
OLLAMA_MODEL = _SETTINGS.get("ollama_model", "llama3.1:8b")
OLLAMA_TIMEOUT = _SETTINGS.get("ollama_timeout", _SETTINGS.get("llm_timeout_sec", 30))
LLM_BACKEND = _SETTINGS.get("llm_backend", "ollama")

TEMPERATURES = [0.7, 0.9, 1.0, 1.1, 1.2]
FORBIDDEN_ANSWER_WORDS = {
    "break",
    "brake",
    "breaking",
    "breaks",
    "for",
    "four",
    "fore",
    "assign",
    "assignment",
    "equals",
    "equal",
    "float",
    "string",
    "integer",
    "int",
    "dictionary",
    "dict",
    "list",
    "if",
    "else",
    "while",
    "and",
    "or",
}

CODE_CONTEXT_HALLUCINATION_TERMS = [
    "code",
    "syntax",
    "print",
    "console",
    "function",
    "indentation",
    "output",
    "expression",
    "error message",
    "code snippet",
    "line of code",
    "if statement",
    "conditional statement",
    "vocabulary",
    "vocabulory",
]


def make_obs(event_type: str) -> dict:
    snapshots = {
        "engaged_fast": {
            "face_detected": True,
            "body_detected": True,
            "gaze_on_robot": 0.8,
            "head_yaw_deg": 0.0,
            "head_pitch_deg": 0.0,
            "head_moving": False,
            "hand_raised": False,
            "hand_raise_side": "none",
            "no_movement_sec": 1.0,
            "still_there_prompt": False,
            "speech_energy": "medium",
        },
        "engaged_normal": {
            "face_detected": True,
            "body_detected": True,
            "gaze_on_robot": 0.75,
            "head_yaw_deg": 0.0,
            "head_pitch_deg": 0.0,
            "head_moving": False,
            "hand_raised": False,
            "hand_raise_side": "none",
            "no_movement_sec": 2.0,
            "still_there_prompt": False,
            "speech_energy": "low",
        },
        "hesitant": {
            "face_detected": True,
            "body_detected": True,
            "gaze_on_robot": 0.55,
            "head_yaw_deg": 4.0,
            "head_pitch_deg": 3.0,
            "head_moving": False,
            "hand_raised": False,
            "hand_raise_side": "none",
            "no_movement_sec": 9.0,
            "still_there_prompt": False,
            "speech_energy": "low",
        },
        "still_waiting": {
            "face_detected": True,
            "body_detected": True,
            "gaze_on_robot": 0.65,
            "head_yaw_deg": 2.0,
            "head_pitch_deg": 1.0,
            "head_moving": False,
            "hand_raised": False,
            "hand_raise_side": "none",
            "no_movement_sec": 15.0,
            "still_there_prompt": True,
            "speech_energy": "low",
        },
        "head_turned": {
            "face_detected": True,
            "body_detected": True,
            "gaze_on_robot": 0.15,
            "head_yaw_deg": 42.0,
            "head_pitch_deg": 3.0,
            "head_moving": True,
            "hand_raised": False,
            "hand_raise_side": "none",
            "no_movement_sec": 1.5,
            "still_there_prompt": False,
            "speech_energy": "low",
        },
        "face_absent": {
            "face_detected": False,
            "body_detected": False,
            "gaze_on_robot": 0.0,
            "head_yaw_deg": 0.0,
            "head_pitch_deg": 0.0,
            "head_moving": False,
            "hand_raised": False,
            "hand_raise_side": "none",
            "no_movement_sec": 6.0,
            "still_there_prompt": False,
            "speech_energy": "low",
        },
        "no_movement": {
            "face_detected": True,
            "body_detected": True,
            "gaze_on_robot": 0.7,
            "head_yaw_deg": 2.0,
            "head_pitch_deg": 1.0,
            "head_moving": False,
            "hand_raised": False,
            "hand_raise_side": "none",
            "no_movement_sec": 18.0,
            "still_there_prompt": True,
            "speech_energy": "low",
        },
        "hand_raised": {
            "face_detected": True,
            "body_detected": True,
            "gaze_on_robot": 0.8,
            "head_yaw_deg": 5.0,
            "head_pitch_deg": 0.0,
            "head_moving": True,
            "hand_raised": True,
            "hand_raise_side": "right",
            "no_movement_sec": 0.5,
            "still_there_prompt": False,
            "speech_energy": "medium",
        },
        "hand_raise": {
            "face_detected": True,
            "body_detected": True,
            "gaze_on_robot": 0.8,
            "head_yaw_deg": 5.0,
            "head_pitch_deg": 0.0,
            "head_moving": True,
            "hand_raised": True,
            "hand_raise_side": "right",
            "no_movement_sec": 0.5,
            "still_there_prompt": False,
            "speech_energy": "medium",
        },
    }
    return copy.deepcopy(snapshots.get(event_type, snapshots["engaged_normal"]))


def make_timing(response_kind: str, rng: random.Random) -> dict:
    if response_kind == "slow":
        response_time_sec = rng.uniform(9.0, 14.0)
        whisper_transcribe_sec = rng.uniform(0.7, 1.4)
        mic_listen_sec = response_time_sec - whisper_transcribe_sec
        hesitation_high = True
    elif response_kind == "no_answer":
        response_time_sec = rng.uniform(15.0, 15.5)
        mic_listen_sec = rng.uniform(14.8, 15.3)
        whisper_transcribe_sec = 0.0
        hesitation_high = True
    else:
        response_time_sec = rng.uniform(2.0, 5.0)
        whisper_transcribe_sec = rng.uniform(0.7, 1.2)
        mic_listen_sec = response_time_sec - whisper_transcribe_sec
        hesitation_high = False

    tts_question_latency_sec = rng.uniform(2.0, 5.0)
    tts_feedback_latency_sec = 0.0
    total_cycle_sec = (
        tts_question_latency_sec + response_time_sec + tts_feedback_latency_sec
    )
    return {
        "response_time_sec": round(response_time_sec, 3),
        "mic_listen_sec": round(mic_listen_sec, 3),
        "whisper_transcribe_sec": round(whisper_transcribe_sec, 3),
        "tts_question_latency_sec": round(tts_question_latency_sec, 3),
        "tts_feedback_latency_sec": round(tts_feedback_latency_sec, 3),
        "total_cycle_sec": round(total_cycle_sec, 3),
        "hesitation_high": hesitation_high,
    }


def classify_response_behavior(answer_status: str, response_time_sec: float) -> str:
    if answer_status == "no_answer":
        return "no_answer"
    if answer_status == "correct":
        return "slow_correct" if response_time_sec >= 8.0 else "fast_correct"
    if answer_status == "incorrect":
        return "slow_wrong" if response_time_sec >= 8.0 else "fast_wrong"
    return "unknown"


def initial_learner_state() -> dict:
    return {
        "correct_streak": 0,
        "wrong_streak": 0,
        "rolling_accuracy": 0.0,
        "confidence_level": "medium",
        "weak_concept": None,
        "concept_attempts": {concept: 0 for concept in CONCEPTS},
        "concept_correct": {concept: 0 for concept in CONCEPTS},
        "concept_wrong": {concept: 0 for concept in CONCEPTS},
        "repeated_hesitation_count": 0,
        "no_answer_count": 0,
        "total_questions_answered": 0,
    }


def update_learner_state(state: dict, quiz_event: dict) -> dict:
    updated = copy.deepcopy(state)
    concept = quiz_event["concept"]
    answer_status = quiz_event["answer_status"]
    response_behavior = quiz_event["response_behavior"]

    updated["total_questions_answered"] += 1
    updated["concept_attempts"][concept] += 1
    if answer_status == "correct":
        updated["correct_streak"] += 1
        updated["wrong_streak"] = 0
        updated["concept_correct"][concept] += 1
    elif answer_status == "incorrect":
        updated["wrong_streak"] += 1
        updated["correct_streak"] = 0
        updated["concept_wrong"][concept] += 1
    elif answer_status == "no_answer":
        updated["no_answer_count"] += 1
        updated["correct_streak"] = 0
        updated["concept_wrong"][concept] += 1

    if response_behavior in ("slow_wrong", "slow_correct", "no_answer"):
        updated["repeated_hesitation_count"] += 1

    total_correct = sum(updated["concept_correct"].values())
    total_answered = updated["total_questions_answered"]
    updated["rolling_accuracy"] = (
        round(total_correct / total_answered, 3) if total_answered else 0.0
    )
    if updated["rolling_accuracy"] >= 0.75:
        updated["confidence_level"] = "high"
    elif updated["rolling_accuracy"] >= 0.4:
        updated["confidence_level"] = "medium"
    else:
        updated["confidence_level"] = "low"

    weakest = None
    weakest_accuracy = None
    for concept_name in CONCEPTS:
        attempts = updated["concept_attempts"][concept_name]
        if attempts < 2:
            continue
        accuracy = updated["concept_correct"][concept_name] / attempts
        if weakest_accuracy is None or accuracy < weakest_accuracy:
            weakest_accuracy = accuracy
            weakest = concept_name
    updated["weak_concept"] = weakest
    return updated


def build_interaction_state(quiz_event: dict) -> dict:
    response_time_sec = quiz_event["response_time_sec"]
    hesitation_high = quiz_event["hesitation_high"]
    response_behavior = quiz_event["response_behavior"]
    answer_status = quiz_event["answer_status"]
    rapid_guess = (
        response_behavior == "fast_wrong"
        and response_time_sec is not None
        and response_time_sec <= 3.5
    )
    return {
        "hesitation_time_sec": response_time_sec if hesitation_high else 0.0,
        "hesitation_high": hesitation_high,
        "rapid_guess": rapid_guess,
        "clarification_requested": quiz_event.get("clarification_requested", False),
        "last_answer_time_sec": response_time_sec,
        "last_transcript": quiz_event["student_answer"],
        "last_response_behavior": response_behavior,
        "no_answer": answer_status == "no_answer",
    }


def make_quiz_event(
    scenario_name,
    concept,
    question_text,
    correct_answer,
    accepted_answers,
    student_answer,
    answer_status,
    response_kind,
    rng,
    setup_only=False,
) -> dict:
    timing = make_timing(response_kind, rng)
    response_time_sec = timing["response_time_sec"]
    response_behavior = classify_response_behavior(answer_status, response_time_sec)
    is_correct = answer_status == "correct"
    rapid_guess = response_behavior == "fast_wrong" and response_time_sec <= 3.5
    return {
        "event_type": "quiz_answer",
        "trigger": "no_answer" if answer_status == "no_answer" else "quiz_answer",
        "scenario_name": scenario_name,
        "concept": concept,
        "question_text": question_text,
        "correct_answer": correct_answer,
        "accepted_answers": accepted_answers,
        "student_answer": student_answer,
        "is_correct": is_correct,
        "answer_status": answer_status,
        "response_time_sec": response_time_sec,
        "response_behavior": response_behavior,
        "hesitation_high": timing["hesitation_high"],
        "rapid_guess": rapid_guess,
        "mic_listen_sec": timing["mic_listen_sec"],
        "whisper_transcribe_sec": timing["whisper_transcribe_sec"],
        "tts_question_latency_sec": timing["tts_question_latency_sec"],
        "tts_feedback_latency_sec": timing["tts_feedback_latency_sec"],
        "total_cycle_sec": timing["total_cycle_sec"],
        "setup_only": setup_only,
    }


def build_c2_prompt(
    event: dict,
    learner_state: dict,
    interaction_state: dict,
    obs: dict,
) -> tuple[str, str]:
    system_prompt = (
        "You are a warm robot tutor helping an adult student during a beginner "
        "programming concepts quiz. Write one short spoken tutor response "
        "based only on the provided behavior, interaction, and learner-state "
        "context. Do not reveal the answer."
    )

    lines = [
        "Situation:",
        "You are tutoring a beginner programming concepts learner.",
        "",
        "Event context:",
        f"- event_type: {event.get('event_type')}",
        f"- trigger: {event.get('trigger')}",
    ]
    if event.get("answer_status") is not None:
        lines.append(f"- answer_status: {event.get('answer_status')}")
    if event.get("response_behavior") is not None:
        lines.append(f"- response_behavior: {event.get('response_behavior')}")
    if event.get("student_answer"):
        lines.append(f"- student_answer: {event.get('student_answer')}")
    lines.extend([
        f"- response_time_sec: {event.get('response_time_sec')}",
        f"- hesitation_high: {interaction_state.get('hesitation_high')}",
        f"- rapid_guess: {interaction_state.get('rapid_guess')}",
        f"- no_answer: {interaction_state.get('no_answer')}",
        "",
        "Behavior context:",
        f"- face_detected: {obs.get('face_detected')}",
        f"- gaze_on_robot: {obs.get('gaze_on_robot')}",
        f"- head_yaw_deg: {obs.get('head_yaw_deg')}",
        f"- no_movement_sec: {obs.get('no_movement_sec')}",
        f"- hand_raised: {obs.get('hand_raised')}",
        "",
        "Learner state:",
        f"- correct_streak: {learner_state.get('correct_streak')}",
        f"- wrong_streak: {learner_state.get('wrong_streak')}",
        f"- rolling_accuracy: {learner_state.get('rolling_accuracy')}",
        f"- confidence_level: {learner_state.get('confidence_level')}",
        f"- weak_concept: {learner_state.get('weak_concept')}",
        f"- repeated_hesitation_count: {learner_state.get('repeated_hesitation_count')}",
        f"- no_answer_count: {learner_state.get('no_answer_count')}",
    ])

    if event.get("event_type") == "quiz_answer":
        lines.extend([
            "",
            "Question context:",
            f"- question_text: {event.get('question_text')}",
        ])

    lines.extend([
        "",
        "Rules:",
        "- Output exactly one sentence.",
        "- 8 to 18 words.",
        "- Speak directly using \"you\".",
        "- Do not reveal the answer.",
        "- Do not add a second sentence.",
        "- After the sentence, stop immediately.",
        "- Do not mention that you are an AI.",
        "- Do not say the selected strategy.",
        "- Do not use labels like scaffold, praise, or reassure.",
        "- This is a term question, not a code-writing task.",
        "- Do not mention code, syntax, print, console, function, indentation, output, expression, error messages, or code snippets.",
        "- Do not assume the learner is writing code.",
    ])
    return system_prompt, "\n".join(lines)


def _clean_text(text: str) -> str:
    text = (text or "").strip().strip("\"'").strip()
    text = text.splitlines()[0].strip() if text else ""
    if ":" in text and text.index(":") < 18:
        text = text.split(":", 1)[1].strip()
    return text.strip().strip("\"'").strip()


def _is_answer_leak(text: str) -> bool:
    words = set(re.findall(r"\b[a-zA-Z_]+\b", text.lower()))
    return bool(words & FORBIDDEN_ANSWER_WORDS)


def _has_code_context_hallucination(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in CODE_CONTEXT_HALLUCINATION_TERMS)


def _is_valid_response(text: str, apply_answer_validation: bool) -> tuple[bool, str | None]:
    if not text:
        return False, "empty_response"
    word_count = len(text.split())
    if word_count < 8 or word_count > 18:
        return False, f"word_count_{word_count}"
    if apply_answer_validation and _is_answer_leak(text):
        return False, "answer_leak"
    if any(label in text.lower() for label in ("scaffold", "praise", "reassure")):
        return False, "strategy_label"
    if _has_code_context_hallucination(text):
        return False, "code_context_hallucination"
    return True, None


def generate_c2_response(
    system_prompt: str,
    user_prompt: str,
    apply_answer_validation: bool = False,
) -> dict:
    retry_log = []
    llm_raw_response = ""
    t0 = time.time()
    for attempt, temp in enumerate(TEMPERATURES, start=1):
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": {
                    "temperature": temp,
                    "num_predict": 60,
                },
            }
            response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            latency = round(time.time() - t0, 3)
            if response.status_code != 200:
                retry_log.append({
                    "attempt": attempt,
                    "temperature": temp,
                    "rejection_reason": f"http_{response.status_code}",
                    "raw_response": "",
                })
                continue
            llm_raw_response = response.json().get("response", "").strip()
            text = _clean_text(llm_raw_response)
            valid, reason = _is_valid_response(text, apply_answer_validation)
            retry_log.append({
                "attempt": attempt,
                "temperature": temp,
                "rejection_reason": reason,
                "raw_response": llm_raw_response,
            })
            if valid:
                return {
                    "response_text": text,
                    "llm_used": True,
                    "llm_latency_sec": latency,
                    "llm_prompt_sent": user_prompt,
                    "llm_raw_response": llm_raw_response,
                    "llm_retry_log": retry_log,
                    "response_source": "llm",
                    "llm_backend": LLM_BACKEND,
                }
        except requests.exceptions.RequestException as exc:
            retry_log.append({
                "attempt": attempt,
                "temperature": temp,
                "rejection_reason": exc.__class__.__name__,
                "raw_response": llm_raw_response,
            })

    latency = round(time.time() - t0, 3)
    return {
        "response_text": "Take a moment and think through the question carefully.",
        "llm_used": False,
        "llm_latency_sec": latency,
        "llm_prompt_sent": user_prompt,
        "llm_raw_response": llm_raw_response,
        "llm_retry_log": retry_log,
        "response_source": "deterministic_fallback",
        "llm_backend": LLM_BACKEND,
    }


def run_c2_quiz_event(event, learner_state, rng, run_index, event_index):
    updated_learner_state = update_learner_state(learner_state, event)
    interaction_state = build_interaction_state(event)
    response_behavior = event.get("response_behavior")
    if response_behavior in ("fast_correct", "fast_wrong"):
        obs = make_obs("engaged_fast")
    elif response_behavior in ("slow_correct", "slow_wrong"):
        obs = make_obs("hesitant")
    elif response_behavior == "no_answer":
        obs = make_obs("still_waiting")
    else:
        obs = make_obs("engaged_normal")
    system_prompt, user_prompt = build_c2_prompt(
        event,
        updated_learner_state,
        interaction_state,
        obs,
    )
    llm_result = generate_c2_response(
        system_prompt,
        user_prompt,
        apply_answer_validation=True,
    )
    result = _base_event_result(event, updated_learner_state, interaction_state, obs)
    result.update({
        "run_index": run_index,
        "event_index": event_index,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        **llm_result,
    })
    return result, updated_learner_state


def _base_event_result(event, learner_state, interaction_state, obs):
    return {
        "condition": CONDITION,
        "condition_name": CONDITION_NAME,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "scenario_name": event["scenario_name"],
        "event_type": event["event_type"],
        "setup_only": event["setup_only"],
        "concept": event["concept"],
        "question_text": event["question_text"],
        "student_answer": event["student_answer"],
        "answer_status": event["answer_status"],
        "response_behavior": event["response_behavior"],
        "response_time_sec": event["response_time_sec"],
        "hesitation_high": event["hesitation_high"],
        "rapid_guess": event["rapid_guess"],
        "mic_listen_sec": event["mic_listen_sec"],
        "whisper_transcribe_sec": event["whisper_transcribe_sec"],
        "tts_question_latency_sec": event["tts_question_latency_sec"],
        "tts_feedback_latency_sec": event["tts_feedback_latency_sec"],
        "total_cycle_sec": event["total_cycle_sec"],
        "learner_state": copy.deepcopy(learner_state),
        "interaction_state": interaction_state,
        "obs": obs,
    }


def run_c2_behavior_event(
    trigger,
    learner_state,
    concept,
    question_text,
    run_index,
    event_index,
):
    obs = make_obs(trigger)
    interaction_state = {
        "hesitation_high": False,
        "rapid_guess": False,
        "no_answer": False,
        "last_response_behavior": None,
    }
    event = {
        "event_type": "behavior_event",
        "trigger": trigger,
        "scenario_name": trigger,
        "concept": concept,
        "question_text": question_text,
        "response_time_sec": None,
        "answer_status": None,
        "response_behavior": None,
    }
    system_prompt, user_prompt = build_c2_prompt(event, learner_state, interaction_state, obs)
    llm_result = generate_c2_response(system_prompt, user_prompt, False)
    return {
        "condition": CONDITION,
        "condition_name": CONDITION_NAME,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_index": run_index,
        "event_index": event_index,
        "event_type": "behavior_event",
        "trigger": trigger,
        "learner_state": copy.deepcopy(learner_state),
        "interaction_state": interaction_state,
        "obs": obs,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        **llm_result,
    }


def run_hand_raise_event(
    spoken_request,
    expected_help_request_type,
    concept,
    question_text,
    run_index,
    event_index,
):
    obs = make_obs("hand_raise")
    hand_raise_result = handle_hand_raise(
        spoken_request,
        question_text=question_text,
        concept=concept,
        use_llm=True,
    )
    return {
        "condition": CONDITION,
        "condition_name": CONDITION_NAME,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_index": run_index,
        "event_index": event_index,
        "event_type": "hand_raise_help_action",
        "trigger": "hand_raised",
        "obs": obs,
        "spoken_request": spoken_request,
        "expected_help_request_type": expected_help_request_type,
        "help_request_type": hand_raise_result["help_request_type"],
        "classification_correct": (
            hand_raise_result["help_request_type"] == expected_help_request_type
        ),
        "response_policy": hand_raise_result["response_policy"],
        "response_text": hand_raise_result["response_text"],
        "routes_to_strategy": hand_raise_result["routes_to_strategy"],
        "concept": concept,
        "question_text": question_text,
        "llm_used": hand_raise_result.get("llm_used", False),
        "llm_latency_sec": hand_raise_result.get("llm_latency_sec", 0.0),
        "llm_prompt_sent": hand_raise_result.get("llm_prompt_sent", ""),
        "llm_raw_response": hand_raise_result.get("llm_raw_response", ""),
        "llm_retry_log": hand_raise_result.get("llm_retry_log", []),
        "llm_backend": hand_raise_result.get("llm_backend", LLM_BACKEND),
        "response_source": (
            "llm" if hand_raise_result.get("llm_used") else "deterministic_fallback"
        ),
    }


def build_event_sequence(rng: random.Random) -> list[dict]:
    return [
        make_quiz_event("praise", "variables", "What does a variable store?", "value", ["value"], "value", "correct", "fast", rng),
        make_quiz_event("confidence_boost", "data_types", "What type stores decimal numbers?", "float", ["float"], "float", "correct", "slow", rng),
        make_quiz_event("challenge", "conditionals", "What keyword starts a condition?", "if", ["if"], "if", "correct", "fast", rng),
        make_quiz_event("slow_down", "conditionals", "What operator needs both conditions true?", "and", ["and"], "equals", "incorrect", "fast", rng),
        make_quiz_event("metacognitive_prompt", "variables", "What symbol assigns a value?", "equals", ["equals"], "value", "incorrect", "fast", rng),
        make_quiz_event("setup_reassure_low_confidence", "conditionals", "What word gives another choice?", "else", ["else"], "if", "incorrect", "fast", rng, setup_only=True),
        make_quiz_event("setup_reassure_low_confidence_2", "loops", "What word repeats while a rule is true?", "while", ["while"], "for", "incorrect", "fast", rng, setup_only=True),
        make_quiz_event("setup_reassure_low_confidence_3", "data_types", "What type stores true or false?", "boolean", ["boolean", "bool"], "string", "incorrect", "fast", rng, setup_only=True),
        make_quiz_event("reassure", "variables", "What word means give a variable a value?", "assign", ["assign"], "value", "incorrect", "fast", rng),
        make_quiz_event("scaffold_hint_no_answer", "loops", "What keyword exits a loop immediately?", "break", ["break", "brake"], "", "no_answer", "no_answer", rng),
        make_quiz_event("setup_reset_variables_accuracy", "variables", "What can store text or numbers?", "variable", ["variable"], "variable", "correct", "fast", rng, setup_only=True),
        make_quiz_event("setup_reset_conditionals_accuracy", "conditionals", "What word starts a condition?", "if", ["if"], "if", "correct", "fast", rng, setup_only=True),
        make_quiz_event("give_example", "loops", "What keyword starts a counting loop?", "for", ["for", "four", "fore"], "function", "incorrect", "slow", rng),
        make_quiz_event("setup_reset_before_elaboration", "conditionals", "What word gives another branch?", "else", ["else"], "else", "correct", "fast", rng, setup_only=True),
        make_quiz_event("setup_loop_recovery_before_elaboration", "loops", "What keyword repeats over a range?", "for", ["for", "four", "fore"], "for", "correct", "fast", rng, setup_only=True),
        make_quiz_event("setup_data_types_no_answer_1", "data_types", "What type stores text?", "string", ["string", "str"], "", "no_answer", "no_answer", rng, setup_only=True),
        make_quiz_event("setup_data_types_no_answer_2", "data_types", "What type stores whole numbers?", "integer", ["integer", "int"], "", "no_answer", "no_answer", rng, setup_only=True),
        make_quiz_event("elaborate_concept", "data_types", "What type stores key value pairs?", "dictionary", ["dictionary", "dict"], "list", "incorrect", "slow", rng),
    ]


def build_summary(target_events, setup_events, behavior_events, hand_raise_events, final_learner_state):
    all_llm_events = target_events + setup_events + behavior_events + hand_raise_events
    latencies = [
        event.get("llm_latency_sec", 0.0)
        for event in all_llm_events
        if event.get("llm_latency_sec") is not None
    ]
    return {
        "target_events_count": len(target_events),
        "setup_events_count": len(setup_events),
        "behavior_events_count": len(behavior_events),
        "hand_raise_total": len(hand_raise_events),
        "total_events_logged": len(all_llm_events),
        "final_learner_state": final_learner_state,
        "llm_used_count": sum(1 for event in all_llm_events if event.get("llm_used")),
        "deterministic_fallback_count": sum(
            1 for event in all_llm_events
            if event.get("response_source") == "deterministic_fallback"
        ),
        "avg_llm_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "scenario_names_covered": sorted({event["scenario_name"] for event in target_events + setup_events}),
    }


def run_session(run_index: int, seed: int):
    rng = random.Random(seed + run_index - 1)
    learner_state = initial_learner_state()
    target_events = []
    setup_events = []
    behavior_events = []
    hand_raise_events = []
    event_index = 1

    for event in build_event_sequence(rng):
        result, learner_state = run_c2_quiz_event(
            event, learner_state, rng, run_index, event_index
        )
        if event["setup_only"]:
            setup_events.append(result)
        else:
            target_events.append(result)
        event_index += 1

    current_concept = "variables"
    current_question = "What does a variable store?"
    for trigger in ("head_turned", "face_absent", "no_movement"):
        behavior_events.append(
            run_c2_behavior_event(
                trigger,
                learner_state,
                current_concept,
                current_question,
                run_index,
                event_index,
            )
        )
        event_index += 1

    hand_specs = [
        ("I have a question", "ask_question"),
        ("can you repeat the question", "repeat_question"),
        ("can I get a hint", "hint_request"),
        ("what is a variable", "concept_explanation"),
        ("", "clarify_help_request"),
    ]
    for spoken_request, expected in hand_specs:
        hand_raise_events.append(
            run_hand_raise_event(
                spoken_request,
                expected,
                current_concept,
                current_question,
                run_index,
                event_index,
            )
        )
        event_index += 1

    return target_events, setup_events, behavior_events, hand_raise_events, learner_state


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run realistic scripted C2 contextual-state scenarios."
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runs = max(1, args.runs)
    all_target_events = []
    all_setup_events = []
    all_behavior_events = []
    all_hand_raise_events = []
    final_learner_state = initial_learner_state()

    for run_index in range(1, runs + 1):
        print(f"Running C2 scripted session {run_index}/{runs}")
        target_events, setup_events, behavior_events, hand_raise_events, final_learner_state = run_session(
            run_index, args.seed
        )
        all_target_events.extend(target_events)
        all_setup_events.extend(setup_events)
        all_behavior_events.extend(behavior_events)
        all_hand_raise_events.extend(hand_raise_events)

    summary = build_summary(
        all_target_events,
        all_setup_events,
        all_behavior_events,
        all_hand_raise_events,
        final_learner_state,
    )
    payload = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "condition": CONDITION,
        "condition_name": CONDITION_NAME,
        "runs": runs,
        "target_events": all_target_events,
        "setup_events": all_setup_events,
        "behavior_events": all_behavior_events,
        "hand_raise_events": all_hand_raise_events,
        "summary": summary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"C2_contextual_state_{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2))

    print("\nC2 CONTEXTUAL STATE SUMMARY")
    print(f"Runs: {runs}")
    print(f"Target events: {summary['target_events_count']}")
    print(f"Setup events: {summary['setup_events_count']}")
    print(f"Behavior events: {summary['behavior_events_count']}")
    print(f"Hand raise events: {summary['hand_raise_total']}")
    print(f"LLM used: {summary['llm_used_count']}")
    print(f"Deterministic fallback: {summary['deterministic_fallback_count']}")
    print(f"Avg LLM latency: {summary['avg_llm_latency_sec']}")
    print(f"Total events logged: {summary['total_events_logged']}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
