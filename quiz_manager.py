
import re
import time
import random
import threading

from question_bank_programming import get_all_questions
from speech_output import speak, speak_async, wait_speech, is_speaking, stop as speech_stop
from speech_input import listen_for_answer
from interaction_state import InteractionState
from learner_state import LearnerState


_WORD_MAP = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}

_TEENS_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _words_to_number(text: str) -> str | None:
    text = text.strip().lower()

    if re.fullmatch(r"-?\d+\.?\d*", text):
        return text

    if text in _WORD_MAP:
        return _WORD_MAP[text]

    parts = re.split(r"[\s\-]+", text)
    if len(parts) == 2:
        tens_word, ones_word = parts
        if tens_word in _TEENS_TENS and ones_word in _WORD_MAP:
            tens_val = _TEENS_TENS[tens_word]
            ones_val = int(_WORD_MAP[ones_word])
            if ones_val < 10:
                return str(tens_val + ones_val)

    if len(parts) == 2 and parts[1] == "hundred" and parts[0] in _WORD_MAP:
        return str(int(_WORD_MAP[parts[0]]) * 100)

    if len(parts) >= 3 and parts[1] == "hundred":
        hundreds = int(_WORD_MAP.get(parts[0], "0")) * 100
        rest = " ".join(parts[2:])
        rest_num = _words_to_number(rest)
        if rest_num is not None:
            return str(hundreds + int(rest_num))

    return None


def normalise_answer(raw: str) -> str:
    text = raw.strip().lower()

    for filler in ["the answer is", "i think it's", "i think it is",
                   "it's", "it is", "that's", "that is", "um", "uh",
                   "so", "well"]:
        text = re.sub(rf"\b{re.escape(filler)}\b", "", text)
    text = re.sub(r"\s+", " ", text).strip().strip(".")

    words = text.split()
    dedup: list[str] = []
    for w in words:
        if not dedup or w != dedup[-1]:
            dedup.append(w)
    text = " ".join(dedup)

    num = _words_to_number(text)
    if num is not None:
        return num

    return text


def classify_response_behavior(answer_status: str,
                               response_time_sec: float | None,
                               hesitation_threshold_sec: float = 8.0) -> str:
    if answer_status == "no_answer":
        return "no_answer"
    if (answer_status == "correct"
            and response_time_sec is not None
            and response_time_sec >= hesitation_threshold_sec):
        return "slow_correct"
    if answer_status == "correct":
        return "fast_correct"
    if (answer_status == "incorrect"
            and response_time_sec is not None
            and response_time_sec >= hesitation_threshold_sec):
        return "slow_wrong"
    if answer_status == "incorrect":
        return "fast_wrong"
    return "no_answer"


def build_turn_state(question: dict,
                     answer_status: str,
                     response_time_sec: float | None,
                     hesitation_threshold_sec: float = 8.0) -> dict:
    return {
        "concept": question.get("concept"),
        "answer_status": answer_status,
        "response_time_sec": response_time_sec,
        "response_behavior": classify_response_behavior(
            answer_status,
            response_time_sec,
            hesitation_threshold_sec,
        ),
    }


def determine_answer_status(raw_answer: str, is_correct: bool) -> str:
    cleaned = (raw_answer or "").strip()
    if cleaned == "":
        return "no_answer"
    if is_correct:
        return "correct"
    return "incorrect"


def classify_answer_status(raw_answer: str, is_correct: bool) -> str:
    return determine_answer_status(raw_answer, is_correct)


class QuizManager:
    def __init__(self, start_level: int = 1, max_questions: int = 10,
                 enable_builtin_feedback: bool = False):
        # The cleaned programming bank is concept-based, not difficulty-based.
        # self.level is kept as 1 only so older logger/main code still works.
        self.level = 1
        self.max_questions = max_questions
        self.enable_builtin_feedback = enable_builtin_feedback
        self.questions_asked = 0
        self._streak_correct = 0
        self._streak_wrong   = 0
        self._asked_ids: set[str] = set()
        self._questions_cache: list[dict] | None = None
        self._current_question: dict | None = None
        self.last_turn_state: dict | None = None
        self.interaction_state = InteractionState()
        self.learner_state = LearnerState()

    @property
    def is_done(self) -> bool:
        return self.questions_asked >= self.max_questions

    def skip_current(self) -> None:
        """Count current question as skipped and move to the next one."""
        self.questions_asked += 1
        self._current_question = None

    def _runtime_question(self, raw: dict, index: int) -> dict:
        """Add runtime-only fields without polluting question_bank_programming.py."""
        concept_slug = raw["concept"].upper().replace("_", "")[:6]
        return {
            **raw,
            "question_id": f"{concept_slug}_{index + 1:02d}",
            "question_text": raw["question"],
            "correct_answer": raw["answer"],
            "difficulty_level": self.level,
        }

    def _get_pool(self) -> list[dict]:
        if self._questions_cache is None:
            self._questions_cache = [
                self._runtime_question(q, i)
                for i, q in enumerate(get_all_questions())
            ]
        pool = [q for q in self._questions_cache
                if q["question_id"] not in self._asked_ids]
        if not pool:
            self._asked_ids.clear()
            pool = list(self._questions_cache)
        random.shuffle(pool)
        return pool

    def _adjust_difficulty(self, is_correct: bool):
        """Track streaks only. Difficulty adaptation is disabled for the study bank."""
        if is_correct:
            self._streak_correct += 1
            self._streak_wrong = 0
        else:
            self._streak_wrong += 1
            self._streak_correct = 0

    def ask_next_question(self, listen_timeout: float = 15.0,
                          face_detected: bool = True,
                          resume: bool = False,
                          stop_event: threading.Event | None = None) -> dict:
        cycle_start = time.time()

        interrupted_during_tts = False

        if resume and self._current_question is not None:
            q = self._current_question
            print(f"\n  [quiz] Concept {q.get('concept', 'unknown')} | {q['question_id']}")
            print(f"  [quiz] Q: {q['question_text']} (resuming)")

            tts_q_start = time.time()
            speak_async("Let's get back to the question. " + q["question_text"])
            while is_speaking():
                if stop_event is not None and stop_event.is_set():
                    speech_stop()
                    interrupted_during_tts = True
                    break
                time.sleep(0.05)
            if not interrupted_during_tts:
                wait_speech()
            tts_q_latency = round(time.time() - tts_q_start, 3)
        else:
            pool = self._get_pool()
            q = pool[0]
            self._asked_ids.add(q["question_id"])
            self._current_question = q

            print(f"\n  [quiz] Concept {q.get('concept', 'unknown')} | {q['question_id']}")
            print(f"  [quiz] Q: {q['question_text']}")

            tts_q_start = time.time()
            speak_async(q["question_text"])
            while is_speaking():
                if stop_event is not None and stop_event.is_set():
                    speech_stop()
                    interrupted_during_tts = True
                    break
                time.sleep(0.05)
            if not interrupted_during_tts:
                wait_speech()
            tts_q_latency = round(time.time() - tts_q_start, 3)

        if interrupted_during_tts:
            total_cycle = round(time.time() - cycle_start, 3)
            print("  [quiz] Interrupted during question speech.")
            return {
                "question_id":              q["question_id"],
                "question_text":            q["question_text"],
                "correct_answer":           q["correct_answer"],
                "student_answer":           "",
                "is_correct":               False,
                "response_time_sec":        0.0,
                "difficulty_level":         self.level,
                "face_detected_during":     face_detected,
                "tts_question_latency_sec": tts_q_latency,
                "tts_feedback_latency_sec": 0.0,
                "mic_listen_sec":           0.0,
                "whisper_transcribe_sec":   0.0,
                "total_cycle_sec":          total_cycle,
                "interrupted":              True,
            }

        if stop_event is not None and stop_event.is_set():
            total_cycle = round(time.time() - cycle_start, 3)
            print("  [quiz] Interrupted before mic listening.")
            return {
                "question_id":              q["question_id"],
                "question_text":            q["question_text"],
                "correct_answer":           q["correct_answer"],
                "student_answer":           "",
                "is_correct":               False,
                "response_time_sec":        0.0,
                "difficulty_level":         self.level,
                "face_detected_during":     face_detected,
                "tts_question_latency_sec": tts_q_latency,
                "tts_feedback_latency_sec": 0.0,
                "mic_listen_sec":           0.0,
                "whisper_transcribe_sec":   0.0,
                "total_cycle_sec":          total_cycle,
                "interrupted":              True,
            }

        listen_result = listen_for_answer(timeout_sec=listen_timeout,
                                          stop_event=stop_event)
        raw_answer      = listen_result["transcript"] or ""
        raw_answer_clean = raw_answer.strip()
        mic_listen_sec  = listen_result["mic_listen_sec"]
        whisper_sec     = listen_result["whisper_transcribe_sec"]
        was_interrupted = listen_result.get("interrupted", False)

        if was_interrupted:
            total_cycle = round(time.time() - cycle_start, 3)
            print("  [quiz] Interrupted — will resume this question later.")
            return {
                "question_id":              q["question_id"],
                "question_text":            q["question_text"],
                "correct_answer":           q["correct_answer"],
                "student_answer":           "",
                "is_correct":               False,
                "response_time_sec":        0.0,
                "difficulty_level":         self.level,
                "face_detected_during":     face_detected,
                "tts_question_latency_sec": tts_q_latency,
                "tts_feedback_latency_sec": 0.0,
                "mic_listen_sec":           mic_listen_sec,
                "whisper_transcribe_sec":   whisper_sec,
                "total_cycle_sec":          total_cycle,
                "interrupted":              True,
            }

        response_time = round(mic_listen_sec + whisper_sec, 3)

        student_normalised = normalise_answer(raw_answer_clean)
        accepted = [a.lower().strip() for a in q["accepted_answers"]]
        is_correct = bool(raw_answer_clean) and student_normalised in accepted
        answer_status = determine_answer_status(raw_answer_clean, is_correct)
        turn_state = build_turn_state(q, answer_status, response_time)
        self.last_turn_state = turn_state

        print(f"  [quiz] Raw answer : '{raw_answer_clean}'")
        print(f"  [quiz] Normalised : '{student_normalised}'")
        print(f"  [quiz] Correct?   : {is_correct}  "
              f"(accepted: {accepted})")

        if not raw_answer_clean:
            fb_text = ""
        elif is_correct:
            fb_text = "Correct!"
        else:
            fb_text = f"No, it's {q['correct_answer']}."

        if self.enable_builtin_feedback and fb_text:
            speak_async(fb_text)

        self._adjust_difficulty(is_correct)

        self.interaction_state.update(
            response_time_sec=response_time,
            is_correct=is_correct,
            transcript=raw_answer_clean
        )

        concept = q.get("concept", "unknown")
        if hasattr(self.learner_state, "update_from_turn"):
            self.learner_state.update_from_turn(turn_state)
        else:
            self.learner_state.update(
                is_correct=is_correct,
                response_time_sec=response_time,
                concept=concept,
                difficulty_level=self.level
            )

        self.questions_asked += 1
        self._current_question = None

        if self.enable_builtin_feedback and fb_text:
            tts_fb_latency = wait_speech()
        else:
            tts_fb_latency = 0.0

        total_cycle = round(time.time() - cycle_start, 3)

        print(f"  [quiz] Latency — tts_q: {tts_q_latency:.3f}s, "
              f"mic: {mic_listen_sec:.3f}s, whisper: {whisper_sec:.3f}s, "
              f"tts_fb: {tts_fb_latency:.3f}s, total: {total_cycle:.3f}s")

        return {
            "question_id":              q["question_id"],
            "question_text":            q["question_text"],
            "correct_answer":           q["correct_answer"],
            "accepted_answers":         q.get("accepted_answers", []),
            "student_answer":           raw_answer_clean,
            "is_correct":               is_correct,
            "answer_status":            answer_status,
            "response_time_sec":        response_time,
            "response_behavior":        turn_state["response_behavior"],
            "difficulty_level":         self.level,
            "concept":                  q.get("concept", "unknown"),
            "weak_concept":             getattr(self.learner_state, "weak_concept", None),
            "confidence_level":         getattr(
                self.learner_state,
                "confidence_level",
                getattr(self.learner_state, "confidence", None),
            ),
            "face_detected_during":     face_detected,
            "tts_question_latency_sec": tts_q_latency,
            "tts_feedback_latency_sec": tts_fb_latency,
            "mic_listen_sec":           mic_listen_sec,
            "whisper_transcribe_sec":   whisper_sec,
            "total_cycle_sec":          total_cycle,
            "interrupted":              False,
        }


if __name__ == "__main__":
    print("=" * 55)
    print("  quiz_manager.py — self-test")
    print("  (pure logic tests — no mic or speaker needed)")
    print("=" * 55)

    tests_passed = 0
    total_tests = 5

    print("\n[Test 1] normalise_answer('  24 ') == '24'?")
    r = normalise_answer("  24 ")
    if r == "24":
        print(f"  PASS  (got '{r}')")
        tests_passed += 1
    else:
        print(f"  FAIL  (got '{r}')")

    print("\n[Test 2] normalise_answer('twenty four') == '24'?")
    r = normalise_answer("twenty four")
    if r == "24":
        print(f"  PASS  (got '{r}')")
        tests_passed += 1
    else:
        print(f"  FAIL  (got '{r}')")

    print("\n[Test 3] normalise_answer('the answer is seven') == '7'?")
    r = normalise_answer("the answer is seven")
    if r == "7":
        print(f"  PASS  (got '{r}')")
        tests_passed += 1
    else:
        print(f"  FAIL  (got '{r}')")

    print("\n[Test 4] QuizManager loads cleaned concept pool?")
    qm = QuizManager(start_level=1)
    pool = qm._get_pool()
    if len(pool) == 60 and {"concept", "question", "answer", "accepted_answers"}.issubset(pool[0].keys()):
        print(f"  PASS  ({len(pool)} cleaned programming questions)")
        tests_passed += 1
    else:
        print(f"  FAIL  (pool size={len(pool)})")

    print("\n[Test 5] 3 correct -> streak tracked without level change?")
    qm2 = QuizManager(start_level=1)
    for _ in range(3):
        qm2._adjust_difficulty(True)
    if qm2.level == 1 and qm2._streak_correct == 3:
        print(f"  PASS  (level = {qm2.level}, streak = {qm2._streak_correct})")
        tests_passed += 1
    else:
        print(f"  FAIL  (level = {qm2.level}, streak = {qm2._streak_correct})")

    total_tests += 1
    print("\n[Test 6] max_questions=10, is_done after 10?")
    qm3 = QuizManager(start_level=1, max_questions=10)
    qm3.questions_asked = 9
    if not qm3.is_done:
        qm3.questions_asked = 10
        if qm3.is_done:
            print("  PASS")
            tests_passed += 1
        else:
            print("  FAIL  (not done at 10)")
    else:
        print("  FAIL  (done at 9)")

    total_tests += 1
    print("\n[Test 7] _current_question preserved for resume?")
    qm4 = QuizManager(start_level=1)
    pool = qm4._get_pool()
    qm4._current_question = pool[0]
    q_id = pool[0]["question_id"]
    if qm4._current_question["question_id"] == q_id and qm4._current_question.get("question_text"):
        print(f"  PASS  (can resume {q_id})")
        tests_passed += 1
    else:
        print("  FAIL")

    total_tests += 1
    print("\n[Test 8] classify_response_behavior maps answer timing?")
    behavior_cases = [
        ("correct", 2.0, "fast_correct"),
        ("correct", 9.0, "slow_correct"),
        ("incorrect", 2.0, "fast_wrong"),
        ("incorrect", 9.0, "slow_wrong"),
        ("no_answer", None, "no_answer"),
    ]
    behavior_results = [
        classify_response_behavior(status, response_time) == expected
        for status, response_time, expected in behavior_cases
    ]
    if all(behavior_results):
        print("  PASS")
        tests_passed += 1
    else:
        for status, response_time, expected in behavior_cases:
            got = classify_response_behavior(status, response_time)
            print(f"  {status}, {response_time}: got {got}, expected {expected}")
        print("  FAIL")

    total_tests += 1
    print("\n[Test 9] determine_answer_status uses stripped transcript?")
    status_cases = [
        ("", False, "no_answer"),
        ("cat", False, "incorrect"),
        ("Boolean", True, "correct"),
        ("   ", False, "no_answer"),
    ]
    status_results = [
        determine_answer_status(raw, is_correct) == expected
        for raw, is_correct, expected in status_cases
    ]
    if all(status_results):
        print("  PASS")
        tests_passed += 1
    else:
        for raw, is_correct, expected in status_cases:
            got = determine_answer_status(raw, is_correct)
            print(f"  {raw!r}, {is_correct}: got {got}, expected {expected}")
        print("  FAIL")

    print("-" * 55)
    print(f"  Result: {tests_passed}/{total_tests} PASS")
    if tests_passed == total_tests:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 55)
