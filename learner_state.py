from dataclasses import dataclass, field
from typing import Optional


CONCEPTS = ["variables", "data_types", "loops", "conditionals"]


def _empty_concept_counts() -> dict:
    return {concept: 0 for concept in CONCEPTS}


@dataclass
class LearnerState:
    correct_streak: int = 0
    wrong_streak: int = 0
    rolling_accuracy: float = 0.0
    confidence_level: str = "low"
    weak_concept: Optional[str] = None
    concept_attempts: dict = field(default_factory=_empty_concept_counts)
    concept_correct: dict = field(default_factory=_empty_concept_counts)
    concept_wrong: dict = field(default_factory=_empty_concept_counts)
    recent_results: list = field(default_factory=list)
    repeated_hesitation_count: int = 0
    no_answer_count: int = 0
    total_questions_answered: int = 0

    @property
    def confidence(self) -> str:
        return self.confidence_level

    @confidence.setter
    def confidence(self, value: str) -> None:
        self.confidence_level = value

    def update_from_turn(self, turn_state: dict) -> None:
        concept = turn_state.get("concept")
        answer_status = turn_state.get("answer_status")
        response_behavior = turn_state.get("response_behavior")

        if concept in CONCEPTS:
            self.concept_attempts[concept] += 1
            if answer_status == "correct":
                self.concept_correct[concept] += 1
            elif answer_status in ("incorrect", "no_answer"):
                self.concept_wrong[concept] += 1

        if answer_status == "correct":
            self.correct_streak += 1
            self.wrong_streak = 0
            self.total_questions_answered += 1
            self.recent_results.append(True)
        elif answer_status == "incorrect":
            self.wrong_streak += 1
            self.correct_streak = 0
            self.total_questions_answered += 1
            self.recent_results.append(False)
        elif answer_status == "no_answer":
            self.correct_streak = 0
            self.no_answer_count += 1
            self.total_questions_answered += 1
            self.recent_results.append(False)

        if response_behavior in ("slow_wrong", "slow_correct", "no_answer"):
            self.repeated_hesitation_count += 1

        self.recent_results = self.recent_results[-5:]
        self._refresh_derived_state()

    def update(
        self,
        is_correct: bool,
        response_time_sec: float,
        concept: str,
        difficulty_level: int,
    ) -> None:
        response_behavior = "slow_correct" if is_correct else "slow_wrong"
        if response_time_sec is not None and response_time_sec < 8.0:
            response_behavior = "fast_correct" if is_correct else "fast_wrong"

        self.update_from_turn({
            "concept": concept,
            "answer_status": "correct" if is_correct else "incorrect",
            "response_time_sec": response_time_sec,
            "response_behavior": response_behavior,
        })

    def get_summary(self) -> dict:
        return {
            "correct_streak": self.correct_streak,
            "wrong_streak": self.wrong_streak,
            "rolling_accuracy": self.rolling_accuracy,
            "confidence_level": self.confidence_level,
            "weak_concept": self.weak_concept,
            "concept_attempts": self.concept_attempts,
            "concept_correct": self.concept_correct,
            "concept_wrong": self.concept_wrong,
            "repeated_hesitation_count": self.repeated_hesitation_count,
            "no_answer_count": self.no_answer_count,
            "total_questions_answered": self.total_questions_answered,
        }

    def to_dict(self) -> dict:
        summary = self.get_summary()
        summary["confidence"] = self.confidence
        return summary

    def reset(self) -> None:
        self.correct_streak = 0
        self.wrong_streak = 0
        self.rolling_accuracy = 0.0
        self.confidence_level = "low"
        self.weak_concept = None
        self.concept_attempts = _empty_concept_counts()
        self.concept_correct = _empty_concept_counts()
        self.concept_wrong = _empty_concept_counts()
        self.recent_results = []
        self.repeated_hesitation_count = 0
        self.no_answer_count = 0
        self.total_questions_answered = 0

    def _refresh_derived_state(self) -> None:
        if self.recent_results:
            self.rolling_accuracy = sum(self.recent_results) / len(self.recent_results)
        else:
            self.rolling_accuracy = 0.0

        if self.rolling_accuracy >= 0.75:
            self.confidence_level = "high"
        elif self.rolling_accuracy >= 0.4:
            self.confidence_level = "medium"
        else:
            self.confidence_level = "low"

        weakest = None
        weakest_accuracy = None
        for concept in CONCEPTS:
            attempts = self.concept_attempts[concept]
            if attempts < 2:
                continue
            accuracy = self.concept_correct[concept] / attempts
            if weakest_accuracy is None or accuracy < weakest_accuracy:
                weakest_accuracy = accuracy
                weakest = concept
        self.weak_concept = weakest


if __name__ == "__main__":
    print("=" * 55)
    print("  learner_state.py self-test")
    print("=" * 55)

    passed = 0
    tests = []

    s = LearnerState()
    s.update_from_turn({
        "concept": "loops",
        "answer_status": "incorrect",
        "response_behavior": "fast_wrong",
    })
    tests.append(("incorrect increments wrong_streak", s.wrong_streak == 1))

    s.update_from_turn({
        "concept": "loops",
        "answer_status": "no_answer",
        "response_behavior": "no_answer",
    })
    tests.append((
        "no_answer increments no_answer_count only",
        s.no_answer_count == 1 and s.wrong_streak == 1,
    ))
    tests.append(("concept_wrong counts no_answer", s.concept_wrong["loops"] == 2))

    one_attempt = LearnerState()
    one_attempt.update_from_turn({
        "concept": "variables",
        "answer_status": "no_answer",
        "response_behavior": "no_answer",
    })
    tests.append((
        "weak_concept remains None below 2 attempts",
        one_attempt.weak_concept is None,
    ))
    tests.append(("weak_concept appears after 2 failed attempts", s.weak_concept == "loops"))

    s.update_from_turn({
        "concept": "loops",
        "answer_status": "correct",
        "response_behavior": "fast_correct",
    })
    tests.append(("correct resets wrong_streak", s.wrong_streak == 0))

    for label, ok in tests:
        if ok:
            print(f"PASS: {label}")
            passed += 1
        else:
            print(f"FAIL: {label}")

    print("-" * 55)
    print(f"Result: {passed}/{len(tests)} PASS")
    if passed == len(tests):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
