
import json
import time
import os
import uuid
import threading
import copy


class SessionLogger:
    def __init__(self, log_dir: str = "."):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{ts}_{uuid.uuid4().hex[:6]}"
        self.start_time = time.strftime("%Y-%m-%dT%H:%M:%S")

        try:
            _cfg_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "config")
            with open(os.path.join(_cfg_dir, "settings.json"), "r") as f:
                _cfg = json.load(f)
            _mode = _cfg.get("llm_mode", "a")
            _condition = _cfg.get("condition", "C4").upper()
            mode_folder = "mode_b1" if _mode == "b" else f"mode_{_mode}"
        except Exception:
            mode_folder = "mode_a"
            _condition = "C4"

        self._log_dir = os.path.join(log_dir, mode_folder)
        os.makedirs(self._log_dir, exist_ok=True)
        self._filename = os.path.join(self._log_dir, f"{_condition}_session_{ts}.json")
        self._lock = threading.Lock()

        self._data = {
            "session_id":         self.session_id,
            "start_time":         self.start_time,
            "observations":       [],
            "quiz_results":       [],
            "engagement_events":  [],
            "summary":            {},
        }


    def log_observation(self, obs_dict: dict) -> None:
        with self._lock:
            self._data["observations"].append(copy.deepcopy(obs_dict))

    def log_quiz_result(self, result_dict: dict) -> None:
        with self._lock:
            self._data["quiz_results"].append(copy.deepcopy(result_dict))

    def log_engagement_event(self, event_dict: dict) -> None:
        entry = copy.deepcopy(event_dict)
        entry.setdefault("strategy", "")
        entry.setdefault("selected_strategy", None)
        entry.setdefault("strategy_reason", "")
        entry.setdefault("strategy_description", "")
        entry.setdefault("strategy_concept", None)
        entry.setdefault("strategy_weak_concept", None)
        entry.setdefault("interaction_state", {})
        entry.setdefault("learner_state", {})
        entry.setdefault("condition", self._default_condition())
        with self._lock:
            self._data["engagement_events"].append(entry)


    def save(self) -> str:
        with self._lock:
            self._data["summary"] = self._build_summary()
            with open(self._filename, "w") as f:
                json.dump(self._data, f, indent=2)
        print(f"  [logger] Saved -> {self._filename}")
        return self._filename


    def _build_summary(self) -> dict:
        observations = self._data["observations"]
        results      = self._data["quiz_results"]
        engagement   = self._data["engagement_events"]

        def _avg(lst):
            return round(sum(lst) / len(lst), 3) if lst else 0.0

        def _pct(count, total):
            return round(count / total * 100, 1) if total else 0.0


        total_obs = len(observations)
        face_detected_count = sum(
            1 for o in observations if o.get("face_detected"))
        body_detected_count = sum(
            1 for o in observations if o.get("body_detected"))
        head_moving_count = sum(
            1 for o in observations if o.get("head_moving"))
        hand_raised_count = sum(
            1 for o in observations if o.get("hand_raised"))
        gaze_scores = [o.get("gaze_on_robot", 0.0) for o in observations
                       if o.get("face_detected")]

        detection_accuracy = {
            "total_observations":     total_obs,
            "face_detection_rate_pct": _pct(face_detected_count, total_obs),
            "body_detection_rate_pct": _pct(body_detected_count, total_obs),
            "head_moving_rate_pct":    _pct(head_moving_count, total_obs),
            "hand_raised_count":       hand_raised_count,
            "avg_gaze_score":          _avg(gaze_scores),
        }


        if results:
            total_q = len(results)
            completed = [r for r in results if not r.get("interrupted")]
            total_completed = len(completed)
            speech_heard = sum(
                1 for r in completed if r.get("student_answer"))
            correct = sum(
                1 for r in completed if r.get("is_correct"))
            face_during = sum(
                1 for r in completed if r.get("face_detected_during"))
            interrupted = sum(
                1 for r in results if r.get("interrupted"))

            speech_accuracy = {
                "total_questions":          total_q,
                "completed_questions":      total_completed,
                "interrupted_questions":    interrupted,
                "speech_captured_count":    speech_heard,
                "speech_capture_rate_pct":  _pct(speech_heard, total_completed),
                "correct_answers":          correct,
                "answer_accuracy_pct":      _pct(correct, total_completed),
                "face_present_during_pct":  _pct(face_during, total_completed),
                "completion_rate_pct":      _pct(total_completed, total_q),
            }
        else:
            speech_accuracy = {
                "total_questions":          0,
                "speech_capture_rate_pct":  0.0,
                "answer_accuracy_pct":      0.0,
            }


        per_level = {}
        if results:
            levels: dict[int, dict] = {}
            for r in results:
                if r.get("interrupted"):
                    continue
                lvl = r.get("difficulty_level", 0)
                if lvl not in levels:
                    levels[lvl] = {"correct": 0, "total": 0}
                levels[lvl]["total"] += 1
                if r.get("is_correct"):
                    levels[lvl]["correct"] += 1

            for lvl, d in sorted(levels.items()):
                per_level[f"level_{lvl}"] = {
                    "correct":      d["correct"],
                    "total":        d["total"],
                    "accuracy_pct": _pct(d["correct"], d["total"]),
                }


        trigger_counts: dict[str, int] = {}
        for e in engagement:
            t = e.get("trigger", "unknown")
            trigger_counts[t] = trigger_counts.get(t, 0) + 1

        engagement_accuracy = {
            "total_events":       len(engagement),
            "triggers_by_type":   trigger_counts,
        }

        condition_summary = self._build_condition_summary(
            observations,
            results,
            engagement,
        )
        strategy_summary = self._build_strategy_summary(engagement)


        completed_results = [r for r in results if not r.get("interrupted")]
        concept_summary = self._build_concept_summary(completed_results, _pct)
        interaction_summary = self._build_interaction_summary(
            engagement,
            results,
            _avg,
        )
        learner_summary = self._build_learner_summary(engagement, results)

        tts_q_lats   = [r["tts_question_latency_sec"] for r in completed_results
                        if r.get("tts_question_latency_sec")]
        tts_fb_lats  = [r["tts_feedback_latency_sec"] for r in completed_results
                        if r.get("tts_feedback_latency_sec")]
        mic_lats     = [r["mic_listen_sec"] for r in completed_results
                        if r.get("mic_listen_sec")]
        whisper_lats = [r["whisper_transcribe_sec"] for r in completed_results
                        if r.get("whisper_transcribe_sec")]
        cycle_lats   = [r["total_cycle_sec"] for r in completed_results
                        if r.get("total_cycle_sec")]
        resp_times   = [r["response_time_sec"] for r in completed_results
                        if r.get("response_time_sec")]

        latency = {
            "avg_tts_question_sec":   _avg(tts_q_lats),
            "avg_tts_feedback_sec":   _avg(tts_fb_lats),
            "avg_mic_listen_sec":     _avg(mic_lats),
            "avg_whisper_transcribe_sec": _avg(whisper_lats),
            "avg_response_time_sec":  _avg(resp_times),
            "avg_total_cycle_sec":    _avg(cycle_lats),
        }


        det_score   = detection_accuracy["face_detection_rate_pct"]
        speech_score = speech_accuracy.get("speech_capture_rate_pct", 0.0)
        quiz_score   = speech_accuracy.get("answer_accuracy_pct", 0.0)
        comp_score   = speech_accuracy.get("completion_rate_pct", 0.0)

        pipeline_score = round(
            (det_score * 0.25) +
            (speech_score * 0.25) +
            (quiz_score * 0.25) +
            (comp_score * 0.25), 1)

        return {
            "detection_accuracy":   detection_accuracy,
            "speech_accuracy":      speech_accuracy,
            "per_level":            per_level,
            "engagement":           engagement_accuracy,
            "condition_summary":     condition_summary,
            "strategy_summary":      strategy_summary,
            "concept_summary":       concept_summary,
            "interaction_summary":   interaction_summary,
            "learner_summary":       learner_summary,
            "latency":              latency,
            "pipeline_score":       pipeline_score,
        }

    def _default_condition(self) -> str:
        try:
            _cfg_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "config")
            with open(os.path.join(_cfg_dir, "settings.json")) as f:
                return json.load(f).get("condition", "unknown")
        except Exception:
            return "unknown"

    def _build_condition_summary(self,
                                 observations: list[dict],
                                 results: list[dict],
                                 engagement_events: list[dict]) -> dict:
        condition = ""
        for event in reversed(engagement_events):
            if event.get("condition"):
                condition = event.get("condition")
                break
        if not condition:
            condition = self._default_condition()

        return {
            "condition": condition or "unknown",
            "total_engagement_events": len(engagement_events),
            "total_quiz_results": len(results),
            "total_observations": len(observations),
        }

    def _build_strategy_summary(self, engagement_events: list[dict]) -> dict:
        strategies_by_type: dict[str, int] = {}
        triggers_by_strategy: dict[str, dict[str, int]] = {}

        for event in engagement_events:
            strategy = (
                event.get("selected_strategy")
                or event.get("strategy")
                or "unknown"
            )
            trigger = event.get("trigger") or "unknown"

            strategies_by_type[strategy] = strategies_by_type.get(strategy, 0) + 1
            if trigger not in triggers_by_strategy:
                triggers_by_strategy[trigger] = {}
            trigger_counts = triggers_by_strategy[trigger]
            trigger_counts[strategy] = trigger_counts.get(strategy, 0) + 1

        return {
            "total_strategy_events": len(engagement_events),
            "strategies_by_type": strategies_by_type,
            "triggers_by_strategy": triggers_by_strategy,
        }

    def _build_concept_summary(self, completed_results: list[dict], pct_fn) -> dict:
        attempts_by_concept: dict[str, int] = {}
        correct_by_concept: dict[str, int] = {}
        answer_status_counts: dict[str, int] = {}
        response_behavior_counts: dict[str, int] = {}

        for result in completed_results:
            concept = result.get("concept") or "unknown"
            attempts_by_concept[concept] = attempts_by_concept.get(concept, 0) + 1
            if result.get("is_correct"):
                correct_by_concept[concept] = correct_by_concept.get(concept, 0) + 1

            answer_status = result.get("answer_status") or self._infer_answer_status(result)
            answer_status_counts[answer_status] = (
                answer_status_counts.get(answer_status, 0) + 1
            )

            response_behavior = result.get("response_behavior") or "unknown"
            response_behavior_counts[response_behavior] = (
                response_behavior_counts.get(response_behavior, 0) + 1
            )

        accuracy_by_concept_pct = {}
        for concept, attempts in attempts_by_concept.items():
            correct = correct_by_concept.get(concept, 0)
            accuracy_by_concept_pct[concept] = pct_fn(correct, attempts)

        return {
            "attempts_by_concept": attempts_by_concept,
            "correct_by_concept": correct_by_concept,
            "accuracy_by_concept_pct": accuracy_by_concept_pct,
            "answer_status_counts": answer_status_counts,
            "response_behavior_counts": response_behavior_counts,
        }

    def _infer_answer_status(self, result: dict) -> str:
        if result.get("is_correct") is True:
            return "correct"
        if result.get("student_answer"):
            return "incorrect"
        return "no_answer"

    def _build_interaction_summary(self,
                                   engagement_events: list[dict],
                                   quiz_results: list[dict],
                                   avg_fn) -> dict:
        hesitation_event_count = 0
        rapid_guess_count = 0
        clarification_request_count = 0
        engagement_no_answer_count = 0
        hesitation_times = []
        response_behavior_counts: dict[str, int] = {}

        for event in engagement_events:
            if event.get("trigger") == "no_answer":
                engagement_no_answer_count += 1

            state = event.get("interaction_state", {})
            if not isinstance(state, dict):
                continue

            if state.get("hesitation_high"):
                hesitation_event_count += 1
            if state.get("rapid_guess"):
                rapid_guess_count += 1
            if state.get("clarification_requested"):
                clarification_request_count += 1

            hesitation_time = state.get("hesitation_time_sec")
            if isinstance(hesitation_time, (int, float)) and hesitation_time > 0:
                hesitation_times.append(hesitation_time)

            # This may include repeated/stale interaction states across engagement events.
            behavior = state.get("last_response_behavior")
            if behavior:
                response_behavior_counts[behavior] = (
                    response_behavior_counts.get(behavior, 0) + 1
                )

        quiz_no_answer_count = 0
        for result in quiz_results:
            if (result.get("answer_status") or self._infer_answer_status(result)) == "no_answer":
                quiz_no_answer_count += 1
            behavior = result.get("response_behavior")
            if behavior:
                response_behavior_counts[behavior] = (
                    response_behavior_counts.get(behavior, 0) + 1
                )

        return {
            "hesitation_event_count": hesitation_event_count,
            "rapid_guess_count": rapid_guess_count,
            "clarification_request_count": clarification_request_count,
            "quiz_no_answer_count": quiz_no_answer_count,
            "engagement_no_answer_count": engagement_no_answer_count,
            "total_no_answer_count": quiz_no_answer_count,
            "avg_hesitation_time_sec": avg_fn(hesitation_times),
            "response_behavior_counts": response_behavior_counts,
        }

    def _build_learner_summary(self,
                               engagement_events: list[dict],
                               quiz_results=None) -> dict:
        if quiz_results:
            completed = [r for r in quiz_results if not r.get("interrupted")]
            if completed:
                correct_streak = 0
                wrong_streak = 0
                correct_count = 0
                concept_attempts: dict[str, int] = {}
                concept_correct: dict[str, int] = {}
                concept_wrong: dict[str, int] = {}
                learner_no_answer_count = 0
                repeated_hesitation_count = 0

                for result in completed:
                    concept = result.get("concept") or "unknown"
                    concept_attempts[concept] = concept_attempts.get(concept, 0) + 1

                    if result.get("is_correct") is True:
                        correct_count += 1
                        correct_streak += 1
                        wrong_streak = 0
                        concept_correct[concept] = concept_correct.get(concept, 0) + 1
                    else:
                        wrong_streak += 1
                        correct_streak = 0
                        concept_wrong[concept] = concept_wrong.get(concept, 0) + 1

                    answer_status = result.get("answer_status") or self._infer_answer_status(result)
                    if answer_status == "no_answer":
                        learner_no_answer_count += 1

                    response_behavior = result.get("response_behavior")
                    if response_behavior in {"slow_wrong", "slow_correct", "no_answer"}:
                        repeated_hesitation_count += 1

                total_completed = len(completed)
                rolling_accuracy = round(correct_count / total_completed, 3)
                if rolling_accuracy >= 0.75 and wrong_streak == 0:
                    confidence_level = "high"
                elif rolling_accuracy >= 0.4:
                    confidence_level = "medium"
                else:
                    confidence_level = "low"

                weak_concept = None
                weakest_accuracy = None
                for concept, attempts in concept_attempts.items():
                    if attempts < 2:
                        continue
                    accuracy = concept_correct.get(concept, 0) / attempts
                    if weakest_accuracy is None or accuracy < weakest_accuracy:
                        weakest_accuracy = accuracy
                        weak_concept = concept

                return {
                    "final_correct_streak": correct_streak,
                    "final_wrong_streak": wrong_streak,
                    "final_rolling_accuracy": rolling_accuracy,
                    "final_confidence_level": confidence_level,
                    "final_weak_concept": weak_concept,
                    "final_concept_attempts": concept_attempts,
                    "final_concept_correct": concept_correct,
                    "final_concept_wrong": concept_wrong,
                    "repeated_hesitation_count": repeated_hesitation_count,
                    "learner_no_answer_count": learner_no_answer_count,
                    "total_questions_answered": total_completed,
                }

        latest = {}
        for event in reversed(engagement_events):
            learner_state = event.get("learner_state", {})
            if isinstance(learner_state, dict) and learner_state:
                latest = learner_state
                break

        return {
            "final_correct_streak": latest.get("correct_streak", 0),
            "final_wrong_streak": latest.get("wrong_streak", 0),
            "final_rolling_accuracy": latest.get("rolling_accuracy", 0.0),
            "final_confidence_level": latest.get(
                "confidence_level",
                latest.get("confidence", ""),
            ),
            "final_weak_concept": latest.get("weak_concept"),
            "final_concept_attempts": latest.get("concept_attempts", {}),
            "final_concept_correct": latest.get("concept_correct", {}),
            "final_concept_wrong": latest.get("concept_wrong", {}),
            "repeated_hesitation_count": latest.get("repeated_hesitation_count", 0),
            "learner_no_answer_count": latest.get("no_answer_count", 0),
            "total_questions_answered": latest.get("total_questions_answered", 0),
        }

    def summarise(self) -> None:
        with self._lock:
            s = self._build_summary()

        det  = s.get("detection_accuracy", {})
        spc  = s.get("speech_accuracy", {})
        eng  = s.get("engagement", {})
        lat  = s.get("latency", {})
        plvl = s.get("per_level", {})
        cond = s.get("condition_summary", {})
        strat = s.get("strategy_summary", {})
        conc = s.get("concept_summary", {})
        inter = s.get("interaction_summary", {})
        learner = s.get("learner_summary", {})

        print("\n" + "=" * 60)
        print("  FULL PIPELINE SUMMARY")
        print("=" * 60)

        print("\n  DETECTION ACCURACY (camera + MediaPipe)")
        print(f"    Total observations   : {det.get('total_observations', 0)}")
        print(f"    Face detection rate   : {det.get('face_detection_rate_pct', 0)}%")
        print(f"    Body detection rate   : {det.get('body_detection_rate_pct', 0)}%")
        print(f"    Head moving rate      : {det.get('head_moving_rate_pct', 0)}%")
        print(f"    Hand raised count     : {det.get('hand_raised_count', 0)}")
        print(f"    Avg gaze score        : {det.get('avg_gaze_score', 0)}")

        print("\n  CONDITION SUMMARY")
        print(f"    Condition             : {cond.get('condition', 'unknown')}")
        print(f"    Engagement events     : {cond.get('total_engagement_events', 0)}")
        print(f"    Quiz results          : {cond.get('total_quiz_results', 0)}")
        print(f"    Observations          : {cond.get('total_observations', 0)}")

        print("\n  SPEECH RECOGNITION / ANSWER CAPTURE")
        print(f"    Total questions       : {spc.get('total_questions', 0)}")
        print(f"    Completed             : {spc.get('completed_questions', 0)}")
        print(f"    Interrupted           : {spc.get('interrupted_questions', 0)}")
        print(f"    Speech captured       : {spc.get('speech_captured_count', 0)}")
        print(f"    Speech capture rate   : {spc.get('speech_capture_rate_pct', 0)}%")
        print(f"    Face present during   : {spc.get('face_present_during_pct', 0)}%")
        print(f"    Completion rate       : {spc.get('completion_rate_pct', 0)}%")

        print("\n  QUIZ ACCURACY")
        print(f"    Correct answers       : {spc.get('correct_answers', 0)}")
        print(f"    Answer accuracy       : {spc.get('answer_accuracy_pct', 0)}%")
        for lvl_key in sorted(plvl):
            d = plvl[lvl_key]
            print(f"    {lvl_key}: {d['correct']}/{d['total']} "
                  f"({d['accuracy_pct']}%)")

        print("\n  ENGAGEMENT")
        print(f"    Total events          : {eng.get('total_events', 0)}")
        triggers = eng.get("triggers_by_type", {})
        for t, c in sorted(triggers.items()):
            print(f"      {t}: {c}")

        print("\n  STRATEGY SUMMARY")
        print(f"    Total strategy events: {strat.get('total_strategy_events', 0)}")
        print("    Strategy counts:")
        for name, count in sorted(strat.get("strategies_by_type", {}).items()):
            print(f"      {name}: {count}")
        print("    Trigger -> strategy counts:")
        for trigger, strategies in sorted(strat.get("triggers_by_strategy", {}).items()):
            print(f"      {trigger}:")
            for name, count in sorted(strategies.items()):
                print(f"        {name}: {count}")

        print("\n  CONCEPT SUMMARY")
        print("    Concept accuracy:")
        attempts = conc.get("attempts_by_concept", {})
        correct_by_concept = conc.get("correct_by_concept", {})
        accuracy = conc.get("accuracy_by_concept_pct", {})
        for concept in sorted(attempts):
            print(f"      {concept}: {correct_by_concept.get(concept, 0)}/"
                  f"{attempts.get(concept, 0)} ({accuracy.get(concept, 0)}%)")
        print("    Answer status counts:")
        for status, count in sorted(conc.get("answer_status_counts", {}).items()):
            print(f"      {status}: {count}")
        print("    Response behavior counts:")
        for behavior, count in sorted(conc.get("response_behavior_counts", {}).items()):
            print(f"      {behavior}: {count}")

        print("\n  INTERACTION SUMMARY")
        print(f"    Hesitation events     : {inter.get('hesitation_event_count', 0)}")
        print(f"    Rapid guesses         : {inter.get('rapid_guess_count', 0)}")
        print(f"    Clarification requests: {inter.get('clarification_request_count', 0)}")
        print(f"    Quiz no answers       : {inter.get('quiz_no_answer_count', 0)}")
        print(f"    Engagement no answers : {inter.get('engagement_no_answer_count', 0)}")
        print(f"    Total no answers      : {inter.get('total_no_answer_count', 0)}")
        print(f"    Avg hesitation time   : {inter.get('avg_hesitation_time_sec', 0)}s")
        print("    Response behavior counts:")
        for behavior, count in sorted(inter.get("response_behavior_counts", {}).items()):
            print(f"      {behavior}: {count}")

        print("\n  LEARNER SUMMARY")
        print(f"    Final correct streak  : {learner.get('final_correct_streak', 0)}")
        print(f"    Final wrong streak    : {learner.get('final_wrong_streak', 0)}")
        print(f"    Rolling accuracy      : {learner.get('final_rolling_accuracy', 0.0)}")
        print(f"    Confidence level      : {learner.get('final_confidence_level', '')}")
        print(f"    Weak concept          : {learner.get('final_weak_concept')}")
        print(f"    Repeated hesitation   : {learner.get('repeated_hesitation_count', 0)}")
        print(f"    Learner no answers    : {learner.get('learner_no_answer_count', 0)}")
        print(f"    Questions answered    : {learner.get('total_questions_answered', 0)}")
        print(f"    Concept attempts      : {learner.get('final_concept_attempts', {})}")
        print(f"    Concept correct       : {learner.get('final_concept_correct', {})}")
        print(f"    Concept wrong         : {learner.get('final_concept_wrong', {})}")

        print("\n  LATENCY")
        print(f"    Avg TTS (question)    : {lat.get('avg_tts_question_sec', 0)}s")
        print(f"    Avg TTS (feedback)    : {lat.get('avg_tts_feedback_sec', 0)}s")
        print(f"    Avg mic listen        : {lat.get('avg_mic_listen_sec', 0)}s")
        print(f"    Avg Whisper transcr.  : {lat.get('avg_whisper_transcribe_sec', 0)}s")
        print(f"    Avg response time     : {lat.get('avg_response_time_sec', 0)}s")
        print(f"    Avg total cycle       : {lat.get('avg_total_cycle_sec', 0)}s")

        print("\n  OVERALL PIPELINE SCORE")
        print(f"    Score: {s.get('pipeline_score', 0)} / 100")
        print(f"      Detection  (25%): {det.get('face_detection_rate_pct', 0)}%")
        print(f"      Speech     (25%): {spc.get('speech_capture_rate_pct', 0)}%")
        print(f"      Quiz       (25%): {spc.get('answer_accuracy_pct', 0)}%")
        print(f"      Completion (25%): {spc.get('completion_rate_pct', 0)}%")
        print("=" * 60)


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  logger.py — self-test")
    print("=" * 60)

    tests_passed = 0
    total_tests = 4

    log = SessionLogger(log_dir=tempfile.gettempdir())

    print("\n[Test 1] log_observation works?")
    log.log_observation({"timestamp": "2025-01-01T00:00:00",
                         "face_detected": True, "body_detected": True,
                         "gaze_on_robot": 0.85, "head_moving": True,
                         "hand_raised": False, "no_movement_sec": 0.0})
    log.log_observation({"timestamp": "2025-01-01T00:00:05",
                         "face_detected": True, "body_detected": True,
                         "gaze_on_robot": 0.72, "head_moving": False,
                         "hand_raised": False, "no_movement_sec": 3.0})
    log.log_observation({"timestamp": "2025-01-01T00:00:10",
                         "face_detected": False, "body_detected": False,
                         "gaze_on_robot": 0.0, "head_moving": False,
                         "hand_raised": False, "no_movement_sec": 8.0})
    if len(log._data["observations"]) == 3:
        print("  PASS")
        tests_passed += 1
    else:
        print("  FAIL")

    print("\n[Test 2] log_quiz_result + log_engagement_event?")
    log.log_quiz_result({
        "question_id": "L1_Q01", "question_text": "What is 6 times 4?",
        "correct_answer": "24", "student_answer": "twenty four",
        "is_correct": True, "response_time_sec": 4.5,
        "concept": "loops", "answer_status": "correct",
        "response_behavior": "fast_correct",
        "weak_concept": None, "confidence_level": "high",
        "difficulty_level": 1, "face_detected_during": True,
        "tts_question_latency_sec": 1.2, "tts_feedback_latency_sec": 0.8,
        "mic_listen_sec": 3.5, "whisper_transcribe_sec": 1.0,
        "total_cycle_sec": 6.5, "interrupted": False,
    })
    log.log_quiz_result({
        "question_id": "L1_Q02", "question_text": "What is 15 minus 8?",
        "correct_answer": "7", "student_answer": "nine",
        "is_correct": False, "response_time_sec": 6.1,
        "concept": "loops", "answer_status": "incorrect",
        "response_behavior": "incorrect",
        "weak_concept": "loops", "confidence_level": "medium",
        "difficulty_level": 1, "face_detected_during": True,
        "tts_question_latency_sec": 1.0, "tts_feedback_latency_sec": 1.1,
        "mic_listen_sec": 4.2, "whisper_transcribe_sec": 0.9,
        "total_cycle_sec": 7.2, "interrupted": False,
    })
    log.log_quiz_result({
        "question_id": "L1_Q03", "question_text": "What is 9 times 3?",
        "correct_answer": "27", "student_answer": "",
        "is_correct": False, "response_time_sec": 0.0,
        "concept": "conditionals", "answer_status": "no_answer",
        "response_behavior": "no_answer",
        "difficulty_level": 1, "face_detected_during": False,
        "tts_question_latency_sec": 1.1, "tts_feedback_latency_sec": 0.0,
        "mic_listen_sec": 2.0, "whisper_transcribe_sec": 0.0,
        "total_cycle_sec": 3.1, "interrupted": True,
    })
    log.log_quiz_result({
        "question_id": "L1_Q04", "question_text": "What stores text?",
        "correct_answer": "string", "student_answer": "",
        "is_correct": False, "response_time_sec": 15.0,
        "concept": "data_types", "answer_status": "no_answer",
        "response_behavior": "no_answer",
        "difficulty_level": 1, "face_detected_during": True,
        "tts_question_latency_sec": 1.0, "tts_feedback_latency_sec": 0.0,
        "mic_listen_sec": 15.0, "whisper_transcribe_sec": 0.0,
        "total_cycle_sec": 16.0, "interrupted": False,
    })
    log.log_engagement_event({
        "timestamp": "2025-01-01T00:00:10",
        "trigger": "no_movement",
        "response_text": "Are you still there?",
        "selected_strategy": "stillness_check",
        "strategy_reason": "learner appears inactive",
        "strategy_description": "Briefly check in and invite continuation.",
        "strategy_concept": None,
        "strategy_weak_concept": "loops",
        "interaction_state": {
            "hesitation_time_sec": 9.0,
            "hesitation_high": True,
            "rapid_guess": False,
            "clarification_requested": False,
            "last_answer_time_sec": 9.0,
            "last_transcript": "nine",
            "last_response_behavior": "slow_wrong",
            "no_answer": False,
        },
        "learner_state": {
            "correct_streak": 0,
            "wrong_streak": 1,
            "rolling_accuracy": 0.5,
            "confidence_level": "medium",
            "weak_concept": "loops",
            "concept_attempts": {"loops": 2},
            "concept_correct": {"loops": 1},
            "concept_wrong": {"loops": 1},
            "repeated_hesitation_count": 1,
            "no_answer_count": 0,
            "total_questions_answered": 2,
        },
        "condition": "C4",
    })
    log.log_engagement_event({
        "timestamp": "2025-01-01T00:00:25",
        "trigger": "hand_raised",
        "response_text": "Do you have a question?",
        "student_question": "What is algebra?",
        "strategy": "answer_question",
        "selected_strategy": "answer_question",
        "strategy_reason": "learner raised a hand",
        "strategy_description": (
            "Answer the learner's spoken question without revealing the current quiz answer."
        ),
        "interaction_state": {
            "hesitation_time_sec": 0.0,
            "hesitation_high": False,
            "rapid_guess": True,
            "clarification_requested": True,
            "last_answer_time_sec": 2.0,
            "last_transcript": "huh",
            "last_response_behavior": "fast_wrong",
            "no_answer": False,
        },
        "learner_state": {
            "correct_streak": 0,
            "wrong_streak": 2,
            "rolling_accuracy": 0.33,
            "confidence_level": "low",
            "weak_concept": "loops",
            "concept_attempts": {"loops": 2, "conditionals": 0},
            "concept_correct": {"loops": 1, "conditionals": 0},
            "concept_wrong": {"loops": 1, "conditionals": 0},
            "repeated_hesitation_count": 1,
            "no_answer_count": 0,
            "total_questions_answered": 2,
        },
        "condition": "C4",
    })
    mutable_learner_state = {
        "correct_streak": 9,
        "wrong_streak": 0,
        "rolling_accuracy": 1.0,
        "confidence_level": "high",
        "weak_concept": None,
        "concept_attempts": {"loops": 1},
        "concept_correct": {"loops": 1},
        "concept_wrong": {"loops": 0},
        "repeated_hesitation_count": 0,
        "no_answer_count": 0,
        "total_questions_answered": 1,
    }
    log.log_engagement_event({
        "timestamp": "2025-01-01T00:00:30",
        "trigger": "deep_copy_test",
        "strategy": "scaffold_hint",
        "selected_strategy": "scaffold_hint",
        "strategy_reason": "test snapshot",
        "learner_state": mutable_learner_state,
        "condition": "C4",
    })
    mutable_learner_state["concept_attempts"]["loops"] = 999
    log.log_engagement_event({
        "timestamp": "2025-01-01T00:00:35",
        "trigger": "no_answer",
        "response_text": "Try one small step.",
        "selected_strategy": "scaffold_hint",
        "strategy_reason": "learner did not answer",
        "interaction_state": {
            "hesitation_time_sec": 0.0,
            "hesitation_high": False,
            "rapid_guess": False,
            "clarification_requested": False,
            "last_answer_time_sec": 15.0,
            "last_transcript": "",
            "last_response_behavior": "no_answer",
            "no_answer": True,
        },
        "condition": "C4",
    })
    log.log_engagement_event({
        "timestamp": "2025-01-01T00:00:45",
        "trigger": "head_turned",
        "response_text": "Let's come back to this.",
        "selected_strategy": "re_engage",
        "strategy_reason": "attention appears to have drifted",
        "interaction_state": {
            "hesitation_time_sec": 0.0,
            "hesitation_high": False,
            "rapid_guess": False,
            "clarification_requested": False,
            "last_answer_time_sec": 15.0,
            "last_transcript": "",
            "last_response_behavior": "no_answer",
            "no_answer": True,
        },
        "condition": "C4",
    })
    ok = (len(log._data["quiz_results"]) == 4 and
          len(log._data["engagement_events"]) == 5 and
          log._data["engagement_events"][2]["learner_state"]["concept_attempts"]["loops"] == 1)
    if ok:
        print("  PASS")
        tests_passed += 1
    else:
        print("  FAIL")

    print("\n[Test 3] save() writes file with pipeline score?")
    path = log.save()
    if os.path.isfile(path):
        with open(path) as f:
            saved = json.load(f)
        has_score = "pipeline_score" in saved.get("summary", {})
        has_det = "detection_accuracy" in saved.get("summary", {})
        has_condition = "condition_summary" in saved.get("summary", {})
        has_strategy = "strategy_summary" in saved.get("summary", {})
        has_concept = "concept_summary" in saved.get("summary", {})
        has_interaction = "interaction_summary" in saved.get("summary", {})
        has_learner = "learner_summary" in saved.get("summary", {})
        interaction_summary = saved.get("summary", {}).get("interaction_summary", {})
        no_answer_counts_ok = (
            interaction_summary.get("quiz_no_answer_count") == 2
            and interaction_summary.get("engagement_no_answer_count") == 1
            and interaction_summary.get("total_no_answer_count") == 2
        )
        strategy_summary = saved.get("summary", {}).get("strategy_summary", {})
        answer_question_ok = (
            strategy_summary.get("strategies_by_type", {}).get("answer_question") == 1
        )
        learner_summary = saved.get("summary", {}).get("learner_summary", {})
        learner_summary_ok = (
            learner_summary.get("total_questions_answered") == 3
            and learner_summary.get("final_concept_attempts") == {
                "loops": 2,
                "data_types": 1,
            }
            and learner_summary.get("final_concept_correct") == {"loops": 1}
            and learner_summary.get("final_concept_wrong") == {
                "loops": 1,
                "data_types": 1,
            }
        )
        size = os.path.getsize(path)
        if (has_score and has_det and has_condition and has_strategy
                and has_concept and has_interaction and has_learner
                and no_answer_counts_ok and answer_question_ok
                and learner_summary_ok):
            print(f"  PASS  ({size} bytes, pipeline_score="
                  f"{saved['summary']['pipeline_score']})")
        else:
            print("  FAIL  (missing expected summary sections)")
        tests_passed += 1 if (
            has_score and has_det and has_condition and has_strategy
            and has_concept and has_interaction and has_learner
            and no_answer_counts_ok and answer_question_ok
            and learner_summary_ok
        ) else 0
        os.remove(path)
    else:
        print(f"  FAIL  (file not found: {path})")

    print("\n[Test 4] summarise() prints full pipeline report?")
    try:
        log.summarise()
        print("  PASS")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")

    print("-" * 60)
    print(f"  Result: {tests_passed}/{total_tests} PASS")
    if tests_passed == total_tests:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
