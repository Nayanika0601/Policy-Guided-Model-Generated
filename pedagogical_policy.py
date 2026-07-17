VALID_STRATEGIES = {
    "praise",
    "confidence_boost",
    "slow_down",
    "scaffold_hint",
    "reassure",
    "challenge",
    "re_engage",
    "stillness_check",
    "metacognitive_prompt",
    "give_example",
    "elaborate_concept",
}


STRATEGY_DESCRIPTIONS = {
    "praise": "Briefly praise a correct answer.",
    "confidence_boost": (
        "Encourage the learner after a correct answer that required hesitation."
    ),
    "challenge": "Encourage the learner to keep thinking carefully after strong performance.",
    "slow_down": "Encourage careful reasoning after a quick incorrect answer.",
    "scaffold_hint": "Give a small hint without revealing the answer.",
    "metacognitive_prompt": (
        "Ask the learner to explain how they are thinking about the question."
    ),
    "reassure": "Reduce frustration after repeated incorrect spoken answers.",
    "give_example": (
        "Give one simple example related to the current concept without revealing the answer."
    ),
    "elaborate_concept": (
        "Briefly explain the current concept in beginner-friendly language without revealing the answer."
    ),
    "re_engage": "Gently bring the learner back when attention appears to drift.",
    "stillness_check": "Check in when the learner appears inactive or stuck.",
}


def get_strategy_description(strategy: str) -> str:
    return STRATEGY_DESCRIPTIONS.get(
        strategy,
        STRATEGY_DESCRIPTIONS["scaffold_hint"],
    )


def select_strategy(
    turn_state: dict | None = None,
    learner_state=None,
    engagement_state: dict | None = None,
) -> dict:
    turn_state, learner_state, engagement_state = _normalize_inputs(
        turn_state,
        learner_state,
        engagement_state,
    )

    concept = turn_state.get("concept")
    answer_status = turn_state.get("answer_status")
    response_behavior = turn_state.get("response_behavior")
    wrong_streak = _learner_get(learner_state, "wrong_streak", 0)
    correct_streak = _learner_get(learner_state, "correct_streak", 0)
    rolling_accuracy = _learner_get(learner_state, "rolling_accuracy", 0.0)
    confidence_level = _learner_get(
        learner_state,
        "confidence_level",
        _learner_get(learner_state, "confidence", "medium"),
    )
    weak_concept = _learner_get(learner_state, "weak_concept", None)
    concept_attempts = _learner_get(learner_state, "concept_attempts", {})
    concept_correct = _learner_get(learner_state, "concept_correct", {})
    interaction_state = turn_state.get("interaction_state", {})
    if not isinstance(interaction_state, dict):
        interaction_state = {}
    rapid_guess = bool(turn_state.get("rapid_guess") or interaction_state.get("rapid_guess"))
    hesitation_high = bool(
        turn_state.get("hesitation_high") or interaction_state.get("hesitation_high")
    )
    trigger = turn_state.get("trigger")

    if (
        trigger in ("face_absent", "head_turned")
        or
        engagement_state.get("face_absent") is True
        or engagement_state.get("gaze_away") is True
    ):
        return _decision("re_engage", "attention appears to have drifted", concept, weak_concept)

    if trigger == "no_movement" or engagement_state.get("no_movement") is True:
        return _decision("stillness_check", "learner appears inactive", concept, weak_concept)

    if answer_status == "no_answer" or trigger == "no_answer":
        return _decision("scaffold_hint", "learner did not answer", concept, weak_concept)

    if answer_status == "correct" and response_behavior == "slow_correct":
        return _decision("confidence_boost", "correct answer after hesitation", concept, weak_concept)

    if answer_status == "correct" and correct_streak >= 3 and rolling_accuracy >= 0.8:
        return _decision("challenge", "learner is performing strongly", concept, weak_concept)

    if answer_status == "correct":
        return _decision("praise", "correct answer", concept, weak_concept)

    if answer_status == "incorrect" and wrong_streak == 2:
        return _decision(
            "metacognitive_prompt",
            "learner has two consecutive incorrect answers",
            concept,
            weak_concept,
        )

    if (
        answer_status == "incorrect"
        and wrong_streak >= 3
        and confidence_level == "low"
    ):
        return _decision("reassure", "learner has repeated incorrect answers", concept, weak_concept)

    if answer_status == "incorrect" and weak_concept == concept:
        attempts = int(concept_attempts.get(concept, 0) or 0)
        accuracy = _concept_accuracy(concept, concept_attempts, concept_correct)
        if attempts >= 4 and accuracy is not None and accuracy < 0.4:
            return _decision(
                "elaborate_concept",
                "learner shows persistent difficulty with the current concept",
                concept,
                weak_concept,
            )
        if attempts >= 3 and accuracy is not None and accuracy < 0.5:
            return _decision(
                "give_example",
                "learner shows difficulty with the current concept",
                concept,
                weak_concept,
            )

    if response_behavior == "fast_wrong" or rapid_guess is True:
        return _decision("slow_down", "quick incorrect answer", concept, weak_concept)

    if response_behavior == "slow_wrong" or hesitation_high is True:
        return _decision("scaffold_hint", "incorrect answer after hesitation", concept, weak_concept)

    if answer_status == "incorrect":
        return _decision("scaffold_hint", "incorrect answer", concept, weak_concept)

    return _decision("scaffold_hint", "default support", concept, weak_concept)


def choose_strategy(*args, **kwargs) -> dict:
    return select_strategy(*args, **kwargs)


def get_strategy(*args, **kwargs) -> dict:
    return select_strategy(*args, **kwargs)


def select_pedagogical_strategy(*args, **kwargs) -> dict:
    return select_strategy(*args, **kwargs)


def policy_decision(*args, **kwargs) -> dict:
    return select_strategy(*args, **kwargs)


def _decision(strategy: str, reason: str, concept, weak_concept) -> dict:
    if strategy not in VALID_STRATEGIES:
        strategy = "scaffold_hint"

    return {
        "strategy": strategy,
        "strategy_description": get_strategy_description(strategy),
        "reason": reason,
        "concept": concept,
        "weak_concept": weak_concept,
    }


def _normalize_inputs(turn_state, learner_state, engagement_state):
    if turn_state is None:
        turn_state = {}
    if not isinstance(turn_state, dict):
        turn_state = {}

    if isinstance(engagement_state, str):
        turn_state, engagement_state = _inputs_from_legacy_trigger(
            turn_state,
            engagement_state,
        )

    if engagement_state is None:
        engagement_state = {}
    if not isinstance(engagement_state, dict):
        engagement_state = {}

    return turn_state, learner_state, engagement_state


def _inputs_from_legacy_trigger(turn_state: dict, trigger: str) -> tuple[dict, dict]:
    turn_state.setdefault("trigger", trigger)
    if trigger == "no_answer":
        turn_state.setdefault("answer_status", "no_answer")
        turn_state.setdefault("response_behavior", "no_answer")
        return turn_state, {}
    if trigger == "hand_raised":
        return turn_state, {"hand_raised": True}
    if trigger == "face_absent":
        return turn_state, {"face_absent": True}
    if trigger == "head_turned":
        return turn_state, {"gaze_away": True}
    if trigger == "no_movement":
        return turn_state, {"no_movement": True}
    return turn_state, {}


def _learner_get(learner_state, attr: str, default):
    if learner_state is None:
        return default
    if isinstance(learner_state, dict):
        return learner_state.get(attr, default)
    return getattr(learner_state, attr, default)


def _concept_accuracy(concept, concept_attempts, concept_correct):
    attempts = int(concept_attempts.get(concept, 0) or 0)
    correct = int(concept_correct.get(concept, 0) or 0)
    if attempts <= 0:
        return None
    return correct / attempts


if __name__ == "__main__":
    print("=" * 55)
    print("  pedagogical_policy.py self-test")
    print("=" * 55)

    class DummyLearnerState:
        def __init__(
            self,
            correct_streak=0,
            wrong_streak=0,
            rolling_accuracy=0.0,
            confidence_level="medium",
            weak_concept=None,
            concept_attempts=None,
            concept_correct=None,
        ):
            self.correct_streak = correct_streak
            self.wrong_streak = wrong_streak
            self.rolling_accuracy = rolling_accuracy
            self.weak_concept = weak_concept
            self.confidence_level = confidence_level
            self.concept_attempts = concept_attempts or {}
            self.concept_correct = concept_correct or {}

    tests = [
        (
            "correct fast -> praise",
            {"trigger": "quiz_answer", "answer_status": "correct", "response_behavior": "fast_correct"},
            DummyLearnerState(),
            {},
            "praise",
        ),
        (
            "correct slow -> confidence_boost",
            {"trigger": "quiz_answer", "answer_status": "correct", "response_behavior": "slow_correct"},
            DummyLearnerState(),
            {},
            "confidence_boost",
        ),
        (
            "fast wrong with low streak -> slow_down",
            {"trigger": "quiz_answer", "answer_status": "incorrect", "response_behavior": "fast_wrong"},
            DummyLearnerState(wrong_streak=1),
            {},
            "slow_down",
        ),
        (
            "slow wrong with low streak -> scaffold_hint",
            {"trigger": "quiz_answer", "answer_status": "incorrect", "response_behavior": "slow_wrong"},
            DummyLearnerState(wrong_streak=1),
            {},
            "scaffold_hint",
        ),
        (
            "no answer -> scaffold_hint",
            {"trigger": "no_answer", "answer_status": "no_answer", "response_behavior": "no_answer"},
            DummyLearnerState(),
            {},
            "scaffold_hint",
        ),
        (
            "no_answer wrong streak >= 5 -> scaffold_hint",
            {"trigger": "no_answer", "answer_status": "no_answer", "response_behavior": "no_answer"},
            DummyLearnerState(wrong_streak=5),
            {},
            "scaffold_hint",
        ),
        (
            "incorrect wrong streak 2 -> metacognitive_prompt",
            {"trigger": "quiz_answer", "answer_status": "incorrect", "response_behavior": "fast_wrong"},
            DummyLearnerState(wrong_streak=2),
            {},
            "metacognitive_prompt",
        ),
        (
            "incorrect wrong streak 3 and confidence low -> reassure",
            {"trigger": "quiz_answer", "answer_status": "incorrect", "response_behavior": "slow_wrong"},
            DummyLearnerState(wrong_streak=3, confidence_level="low"),
            {},
            "reassure",
        ),
        (
            "weak concept moderate -> give_example",
            {"trigger": "quiz_answer", "concept": "loops", "answer_status": "incorrect", "response_behavior": "fast_wrong"},
            DummyLearnerState(
                wrong_streak=1,
                weak_concept="loops",
                concept_attempts={"loops": 3},
                concept_correct={"loops": 1},
            ),
            {},
            "give_example",
        ),
        (
            "weak concept severe -> elaborate_concept",
            {"trigger": "quiz_answer", "concept": "loops", "answer_status": "incorrect", "response_behavior": "fast_wrong"},
            DummyLearnerState(
                wrong_streak=1,
                weak_concept="loops",
                concept_attempts={"loops": 4},
                concept_correct={"loops": 1},
            ),
            {},
            "elaborate_concept",
        ),
        (
            "correct strong streak -> challenge",
            {"trigger": "quiz_answer", "answer_status": "correct", "response_behavior": "fast_correct"},
            DummyLearnerState(correct_streak=3, rolling_accuracy=0.8),
            {},
            "challenge",
        ),
        (
            "head_turned still -> re_engage",
            {"trigger": "head_turned"},
            DummyLearnerState(),
            {},
            "re_engage",
        ),
        (
            "no_movement still -> stillness_check",
            {"trigger": "no_movement"},
            DummyLearnerState(),
            {},
            "stillness_check",
        ),
        (
            "legacy no_answer trigger -> scaffold_hint",
            {},
            None,
            "no_answer",
            "scaffold_hint",
        ),
    ]

    passed = 0
    for label, turn_state, learner_state, engagement_state, expected in tests:
        result = select_strategy(turn_state, learner_state, engagement_state)
        selected = result["strategy"]
        if label == "legacy no_answer trigger -> scaffold_hint":
            assert result["strategy"] == "scaffold_hint"
            assert result["reason"] == "learner did not answer"
        status = "PASS" if selected == expected else "FAIL"
        if selected == expected:
            passed += 1
        print(f"{status}: {label} => {selected} (expected {expected})")

    print("-" * 55)
    print(f"Result: {passed}/{len(tests)} PASS")
    if passed == len(tests):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
