import json
import os


_FALLBACK_CONFIG = {
    "system_prompt": (
        "You are a warm robot tutor helping an adult student learn beginner "
        "Python. Write one short spoken response. Do not decide the strategy."
    ),
    "situations": {
        "no_answer": "No answer was heard for the current question.",
        "head_turned": "Attention appears to have briefly drifted.",
        "face_absent": "The learner appears to have moved out of view.",
        "no_movement": "There has been little movement for a while.",
        "hand_raised": "You raised your hand.",
        "default": "You are working through a beginner Python quiz.",
    },
    "strategy_instructions": {
        "praise": "Briefly praise the correct answer. Do not ask another question.",
        "confidence_boost": (
            "Say the answer was correct and encourage the learner for thinking "
            "carefully. Do not ask another question."
        ),
        "slow_down": "Encourage careful reasoning before answering.",
        "scaffold_hint": "Give one small hint without revealing the answer.",
        "reassure": "Offer encouragement and reduce frustration.",
        "challenge": (
            "Encourage the learner to keep thinking carefully after strong performance."
        ),
        "re_engage": "Gently bring attention back to the task.",
        "stillness_check": "Briefly check in and invite continuation.",
        "metacognitive_prompt": (
            "Ask the learner to briefly explain how they are thinking about the question."
        ),
        "give_example": (
            "Give one simple example related to the current concept without revealing the answer."
        ),
        "elaborate_concept": (
            "Briefly explain the current concept in beginner-friendly language without revealing the answer."
        ),
    },
    "engagement_only_strategies": [
        "re_engage",
        "stillness_check",
    ],
    "rules": [
        "Output exactly one sentence.",
        "8 to 18 words.",
        'Speak directly using "you".',
        "Do not reveal the answer.",
    ],
    "history_limit": 3,
}


_REQUIRED_STRATEGY_INSTRUCTIONS = dict(_FALLBACK_CONFIG["strategy_instructions"])
_REQUIRED_ENGAGEMENT_ONLY_STRATEGIES = [
    "re_engage",
    "stillness_check",
]
_HIDE_CONCEPT_AND_QUESTION = {
    "praise",
    "confidence_boost",
    "challenge",
    "slow_down",
    "metacognitive_prompt",
    "reassure",
    "re_engage",
    "stillness_check",
}
_SHOW_CONCEPT_ONLY = set()
_SHOW_CONCEPT_AND_QUESTION = {
    "scaffold_hint",
    "give_example",
    "elaborate_concept",
}
_STRATEGY_INSTRUCTION_OVERRIDES = {
    "reassure": (
        "Offer brief emotional encouragement only. Do not give a hint, "
        "explanation, example, keyword, or answer."
    ),
    "give_example": (
        "Give one simple everyday example related to the concept without using "
        "the quiz answer or code syntax."
    ),
    "elaborate_concept": (
        "Briefly explain the current concept in beginner-friendly language "
        "without answering the quiz question."
    ),
    "metacognitive_prompt": "Ask the learner to briefly explain how they are thinking.",
    "re_engage": "Gently invite the learner back to the current quiz.",
    "stillness_check": "Briefly check whether the learner is ready to continue.",
}
_EXTRA_RULES_BY_STRATEGY = {
    "scaffold_hint": [
        "Do not use the correct answer or any spelling variant of the correct answer.",
        "Do not repeat the question.",
    ],
    "praise": [
        "Do not say close, almost, try again, or ask the question again.",
        "Do not mention the concept.",
        "Do not repeat the question.",
        "Do not explain the answer.",
    ],
    "confidence_boost": [
        "Do not say close, almost, try again, or ask the question again.",
        "Do not mention the concept.",
        "Do not repeat the question.",
        "Do not explain the answer.",
    ],
    "challenge": [
        "Do not mention the concept.",
        "Do not repeat the question.",
        "Do not explain the answer.",
    ],
    "slow_down": [
        "Do not repeat the question.",
        "Do not mention the concept.",
        "Do not give a hint.",
    ],
    "metacognitive_prompt": [
        "Ask about thinking process, not the answer.",
        "Do not mention the concept.",
        "Do not repeat the question.",
    ],
    "reassure": [
        "Do not mention the concept.",
        "Do not mention the question.",
        "Do not give a hint.",
        "Do not reveal or imply the answer.",
    ],
    "give_example": [
        "Use the question only to understand the topic.",
        "Do not repeat the question.",
        "Do not include the answer.",
        "Do not use code.",
        "Give a simple everyday example.",
    ],
    "elaborate_concept": [
        "Use the question only to understand the learner difficulty.",
        "Do not repeat the question.",
        "Do not include the answer.",
        "Explain the concept generally, not the specific answer.",
    ],
    "re_engage": [
        "Do not mention code snippets.",
        "Do not ask a new quiz question.",
        "Do not mention the concept.",
        "Do not mention the answer.",
    ],
    "stillness_check": [
        "Do not ask a new quiz question.",
        "Do not mention the concept.",
        "Do not mention the answer.",
    ],
}


def _load_config() -> dict:
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config",
        "prompt_assembler.json",
    )
    try:
        with open(config_path, "r") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            return dict(_FALLBACK_CONFIG)
    except Exception:
        return dict(_FALLBACK_CONFIG)

    config = dict(_FALLBACK_CONFIG)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            merged = dict(config[key])
            merged.update(value)
            config[key] = merged
        else:
            config[key] = value
    config["strategy_instructions"] = {
        **config.get("strategy_instructions", {}),
        **_REQUIRED_STRATEGY_INSTRUCTIONS,
    }
    config["engagement_only_strategies"] = _REQUIRED_ENGAGEMENT_ONLY_STRATEGIES
    return config


_PROMPT_CONFIG = _load_config()


def assemble_prompt(obs,
                    trigger,
                    interaction,
                    learner,
                    strategy,
                    strategy_description,
                    history=None,
                    quiz_context=None) -> tuple[str, str]:
    history = history or []
    quiz_context = quiz_context or {}
    learner = learner or {}

    situation = _build_situation(trigger)
    strategy_instruction = _build_strategy_instruction(
        strategy,
        strategy_description,
    )
    concept = _get_concept(quiz_context, learner)

    sections = [
        f"Situation: {situation}",
        f"Teaching move: {strategy_instruction}",
    ]

    if _show_concept(strategy):
        sections.append(f"Concept: {concept}")

        question_text = quiz_context.get("question_text")
        if question_text and _show_question(strategy):
            sections.append(f"Question: {question_text}")

    history_limit = int(_PROMPT_CONFIG.get("history_limit", 3) or 3)
    if history and history_limit > 0:
        history_lines = ["Recent responses to avoid repeating:"]
        for item in history[-history_limit:]:
            history_lines.append(f"- {item}")
        sections.append("\n".join(history_lines))

    sections.append(_response_rules(strategy))

    return _PROMPT_CONFIG.get("system_prompt", _FALLBACK_CONFIG["system_prompt"]), "\n\n".join(sections)


def _build_situation(trigger: str | None) -> str:
    situations = _PROMPT_CONFIG.get("situations", {})
    if trigger == "no_answer":
        return situations.get("no_answer", _FALLBACK_CONFIG["situations"]["no_answer"])
    if trigger == "head_turned":
        return situations.get("head_turned", _FALLBACK_CONFIG["situations"]["head_turned"])
    if trigger == "face_absent":
        return situations.get("face_absent", _FALLBACK_CONFIG["situations"]["face_absent"])
    if trigger == "no_movement":
        return situations.get("no_movement", _FALLBACK_CONFIG["situations"]["no_movement"])
    if trigger == "hand_raised" or str(trigger).startswith("hand_raised"):
        return situations.get("hand_raised", _FALLBACK_CONFIG["situations"]["hand_raised"])
    return situations.get("default", _FALLBACK_CONFIG["situations"]["default"])


def _is_engagement_only_trigger(trigger: str | None) -> bool:
    if trigger == "hand_raised" or str(trigger).startswith("hand_raised"):
        return True
    return trigger in {"face_absent", "head_turned", "no_movement"}


def _build_strategy_instruction(strategy: str | None,
                                strategy_description: str | None) -> str:
    if strategy in _STRATEGY_INSTRUCTION_OVERRIDES:
        return _STRATEGY_INSTRUCTION_OVERRIDES[strategy]

    instructions = _PROMPT_CONFIG.get("strategy_instructions", {})
    if strategy in instructions:
        return instructions[strategy]
    if strategy_description:
        return strategy_description
    return "Give one small supportive hint without revealing the answer."


def _get_concept(quiz_context: dict, learner: dict) -> str:
    concept = quiz_context.get("concept")
    if not concept:
        concept = learner.get("weak_concept")
    return concept or "beginner Python"


def _show_concept(strategy: str | None) -> bool:
    if strategy in _HIDE_CONCEPT_AND_QUESTION:
        return False
    if strategy in _SHOW_CONCEPT_ONLY or strategy in _SHOW_CONCEPT_AND_QUESTION:
        return True
    return True


def _show_question(strategy: str | None) -> bool:
    if strategy in _HIDE_CONCEPT_AND_QUESTION or strategy in _SHOW_CONCEPT_ONLY:
        return False
    if strategy in _SHOW_CONCEPT_AND_QUESTION:
        return True
    return True


def _response_rules(strategy: str | None = None) -> str:
    configured_rules = _PROMPT_CONFIG.get("rules", _FALLBACK_CONFIG["rules"])
    rules = []
    for rule in configured_rules:
        if rule == "One sentence.":
            rules.append("Output exactly one sentence.")
        else:
            rules.append(rule)
    common_rules = [
        "Output exactly one sentence.",
        "Do not add a second sentence.",
        "After the sentence, stop immediately.",
    ]
    rules = common_rules + [rule for rule in rules if rule not in common_rules]
    if strategy in _EXTRA_RULES_BY_STRATEGY:
        rules = list(rules) + _EXTRA_RULES_BY_STRATEGY[strategy]
    return "\n".join(["Rules:"] + [f"- {rule}" for rule in rules])


if __name__ == "__main__":
    cases = [
        (
            "praise",
            {
                "trigger": "quiz_answer",
                "strategy": "praise",
                "strategy_description": "",
                "quiz_context": {
                    "concept": "variables",
                    "question_text": "What does a variable store?",
                },
                "history": [],
            },
        ),
        (
            "confidence_boost",
            {
                "trigger": "quiz_answer",
                "strategy": "confidence_boost",
                "strategy_description": "",
                "quiz_context": {
                    "concept": "data_types",
                    "question_text": "What type stores decimal numbers?",
                },
                "history": [],
            },
        ),
        (
            "challenge",
            {
                "trigger": "quiz_answer",
                "strategy": "challenge",
                "strategy_description": "",
                "quiz_context": {
                    "concept": "conditionals",
                    "question_text": "What keyword starts a condition?",
                },
                "history": [],
            },
        ),
        (
            "slow_down",
            {
                "trigger": "quiz_answer",
                "strategy": "slow_down",
                "strategy_description": "",
                "quiz_context": {
                    "concept": "conditionals",
                    "question_text": "What operator needs both conditions true?",
                },
                "history": [],
            },
        ),
        (
            "reassure",
            {
                "trigger": "quiz_answer",
                "strategy": "reassure",
                "strategy_description": "",
                "quiz_context": {
                    "concept": "variables",
                    "question_text": "What word means give a variable a value?",
                },
                "history": [],
            },
        ),
        (
            "give_example",
            {
                "trigger": "quiz_answer",
                "strategy": "give_example",
                "strategy_description": "",
                "quiz_context": {
                    "concept": "loops",
                    "question_text": "What keyword starts a counting loop?",
                },
                "history": [],
            },
        ),
        (
            "elaborate_concept",
            {
                "trigger": "quiz_answer",
                "strategy": "elaborate_concept",
                "strategy_description": "",
                "quiz_context": {
                    "concept": "data_types",
                    "question_text": "What type stores key value pairs?",
                },
                "history": [],
            },
        ),
        (
            "re_engage",
            {
                "trigger": "head_turned",
                "strategy": "re_engage",
                "strategy_description": "",
                "quiz_context": {
                    "concept": "conditionals",
                    "question_text": "What keyword starts a condition?",
                },
                "history": [],
            },
        ),
        (
            "stillness_check",
            {
                "trigger": "no_movement",
                "strategy": "stillness_check",
                "strategy_description": "",
                "quiz_context": {
                    "concept": "variables",
                    "question_text": "What does a variable store?",
                },
                "history": [],
            },
        ),
        (
            "scaffold_hint",
            {
                "trigger": "no_answer",
                "strategy": "scaffold_hint",
                "strategy_description": "",
                "quiz_context": {
                    "concept": "loops",
                    "question_text": "What keyword starts a loop in Python?",
                },
                "history": ["Take your time, I am here with you."],
            },
        ),
        (
            "metacognitive_prompt",
            {
                "trigger": "quiz_answer",
                "strategy": "metacognitive_prompt",
                "strategy_description": "",
                "quiz_context": {
                    "concept": "data_types",
                    "question_text": "What type stores text?",
                },
                "history": [],
            },
        ),
    ]

    for label, kwargs in cases:
        system_prompt, user_prompt = assemble_prompt(
            obs={
                "gaze_on_robot": 0.2,
                "head_yaw_deg": 12,
                "face_detected": True,
                "no_movement_sec": 5,
            },
            interaction={"wrong_streak": 2},
            learner={"rolling_accuracy": 0.4, "weak_concept": "data_types"},
            **kwargs,
        )

        print("=" * 55)
        print(label)
        print("\n--- SYSTEM PROMPT ---")
        print(system_prompt)
        print("\n--- USER PROMPT ---")
        print(user_prompt)
