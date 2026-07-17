HESITATION_THRESHOLD_SEC = 8.0
RAPID_GUESS_THRESHOLD_SEC = 3.0


_CLARIFICATION_KEYWORDS = (
    "repeat",
    "what",
    "pardon",
    "again",
    "i don't understand",
    "didn't get",
    "can you repeat",
    "say again",
    "can you say",
    "say that again",
    "i don't get it",
    "huh",
)


class InteractionState:
    def __init__(self):
        self.last_answer_time_sec: float = 0.0
        self.hesitation_time_sec: float = 0.0
        self.hesitation_high: bool = False
        self.rapid_guess: bool = False
        self.clarification_requested: bool = False
        self.last_transcript: str = ""
        self.last_response_behavior: str = ""
        self.no_answer: bool = False

    def update(self, response_time_sec: float | None, is_correct: bool,
               transcript: str) -> None:
        text = transcript or ""
        self.last_transcript = text

        self.last_answer_time_sec = (
            response_time_sec if response_time_sec is not None else 0.0
        )

        self.no_answer = text.strip() == ""

        self.hesitation_high = (
            response_time_sec is not None
            and response_time_sec >= HESITATION_THRESHOLD_SEC
        )
        self.hesitation_time_sec = (
            response_time_sec if self.hesitation_high else 0.0
        )

        self.rapid_guess = (
            response_time_sec is not None
            and response_time_sec < RAPID_GUESS_THRESHOLD_SEC
            and is_correct is False
            and self.no_answer is False
        )

        lower_text = text.lower()
        self.clarification_requested = any(
            kw in lower_text for kw in _CLARIFICATION_KEYWORDS
        )

        if self.no_answer:
            self.last_response_behavior = "no_answer"
        elif is_correct and self.hesitation_high:
            self.last_response_behavior = "slow_correct"
        elif is_correct:
            self.last_response_behavior = "fast_correct"
        elif not is_correct and self.rapid_guess:
            self.last_response_behavior = "fast_wrong"
        elif not is_correct and self.hesitation_high:
            self.last_response_behavior = "slow_wrong"
        else:
            self.last_response_behavior = "incorrect"

    def to_dict(self) -> dict:
        return {
            "hesitation_time_sec": self.hesitation_time_sec,
            "hesitation_high": self.hesitation_high,
            "rapid_guess": self.rapid_guess,
            "clarification_requested": self.clarification_requested,
            "last_answer_time_sec": self.last_answer_time_sec,
            "last_transcript": self.last_transcript,
            "last_response_behavior": self.last_response_behavior,
            "no_answer": self.no_answer,
        }

    def reset(self) -> None:
        self.last_answer_time_sec = 0.0
        self.hesitation_time_sec = 0.0
        self.hesitation_high = False
        self.rapid_guess = False
        self.clarification_requested = False
        self.last_transcript = ""
        self.last_response_behavior = ""
        self.no_answer = False

    def __repr__(self) -> str:
        return (
            "InteractionState("
            f"hesitation_high={self.hesitation_high}, "
            f"rapid_guess={self.rapid_guess}, "
            f"clarification_requested={self.clarification_requested}, "
            f"no_answer={self.no_answer}, "
            f"last_response_behavior='{self.last_response_behavior}', "
            f"last_answer_time_sec={self.last_answer_time_sec:.2f}s)"
        )


if __name__ == "__main__":
    print("=" * 55)
    print("  interaction_state.py - self-test")
    print("=" * 55)

    passed = 0
    total = 5

    print("\n[Test 1] fast wrong under 3 sec -> rapid_guess + fast_wrong")
    s = InteractionState()
    s.update(response_time_sec=1.5, is_correct=False, transcript="ten")
    if s.rapid_guess is True and s.last_response_behavior == "fast_wrong":
        print(f"  PASS  ({s!r})")
        passed += 1
    else:
        print(f"  FAIL  ({s!r})")

    print("\n[Test 2] slow correct >= 8 sec -> hesitation_high + slow_correct")
    s = InteractionState()
    s.update(response_time_sec=9.0, is_correct=True, transcript="twenty four")
    if s.hesitation_high is True and s.last_response_behavior == "slow_correct":
        print(f"  PASS  ({s!r})")
        passed += 1
    else:
        print(f"  FAIL  ({s!r})")

    print("\n[Test 3] clarification keyword -> clarification_requested True")
    s = InteractionState()
    s.update(response_time_sec=2.0, is_correct=False,
             transcript="Can you say that again?")
    if s.clarification_requested is True:
        print(f"  PASS  ({s!r})")
        passed += 1
    else:
        print(f"  FAIL  ({s!r})")

    print("\n[Test 4] empty transcript -> no_answer + no_answer behavior")
    s = InteractionState()
    s.update(response_time_sec=None, is_correct=False, transcript="")
    if s.no_answer is True and s.last_response_behavior == "no_answer":
        print(f"  PASS  ({s!r})")
        passed += 1
    else:
        print(f"  FAIL  ({s!r})")

    print("\n[Test 5] to_dict contains all expected keys")
    s = InteractionState()
    s.update(response_time_sec=4.0, is_correct=True, transcript="forty two")
    d = s.to_dict()
    expected = {
        "hesitation_time_sec",
        "hesitation_high",
        "rapid_guess",
        "clarification_requested",
        "last_answer_time_sec",
        "last_transcript",
        "last_response_behavior",
        "no_answer",
    }
    if set(d.keys()) == expected:
        print(f"  PASS  (keys: {sorted(d.keys())})")
        passed += 1
    else:
        print(f"  FAIL  (got keys: {sorted(d.keys())})")

    print("-" * 55)
    print(f"  Result: {passed}/{total} PASS")
    if passed == total:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 55)
