import argparse
import copy
import json
import random
from datetime import datetime
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pedagogical_policy import select_strategy
from prompt_assembler import assemble_prompt
from llm_responder import generate as llm_generate
from hand_raise_manager import handle_hand_raise


CONDITION = "C3"
CONDITION_NAME = "Strategy-Selected State-Conditioned Pipeline"

PEDAGOGICAL_STRATEGIES = [
    "praise",
    "confidence_boost",
    "challenge",
    "slow_down",
    "scaffold_hint",
    "metacognitive_prompt",
    "reassure",
    "give_example",
    "elaborate_concept",
    "re_engage",
    "stillness_check",
]

HAND_RAISE_HELP_ACTIONS = [
    "ask_question",
    "repeat_question",
    "concept_explanation",
    "clarify_help_request",
]

CONCEPTS = ["variables", "data_types", "loops", "conditionals"]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "C3_strategy_pipeline"

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

CONCEPT_QUIZ_RULES = [
    "- This is a term question, not a code-writing task.",
    "- Do not mention code, syntax, print, console, function, indentation, output, expression, error messages, or code snippets.",
    "- Do not assume the learner is writing code.",
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
    expected_strategy,
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
        "expected_strategy": expected_strategy,
        "setup_only": setup_only,
    }


def _generate_response(
    obs,
    trigger,
    system_prompt,
    user_prompt,
    force_llm_for_content,
) -> dict:
    if force_llm_for_content:
        retry_log = []
        last_result = {}
        for attempt in range(1, 6):
            result = llm_generate(
                obs=obs,
                trigger=trigger,
                mode="a",
                assembled_prompt=(system_prompt, user_prompt),
            )
            last_result = result
            if not _has_code_context_hallucination(result.get("response_text", "")):
                return result
            retry_log.extend(result.get("llm_retry_log", []))
            retry_log.append({
                "attempt": attempt,
                "rejection_reason": "code_context_hallucination",
                "raw_response": result.get("llm_raw_response", ""),
            })
        return {
            "response_text": "Simulated C3 tutor response.",
            "llm_used": False,
            "llm_latency_sec": 0.0,
            "llm_prompt_sent": user_prompt,
            "llm_raw_response": last_result.get("llm_raw_response", ""),
            "llm_retry_log": retry_log,
        }
    return {
        "response_text": "Simulated C3 tutor response.",
        "llm_used": False,
        "llm_latency_sec": 0.0,
        "llm_prompt_sent": user_prompt,
        "llm_raw_response": "",
        "llm_retry_log": [],
    }


def _has_code_context_hallucination(text: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in CODE_CONTEXT_HALLUCINATION_TERMS)


def _adapt_c3_prompt_for_concepts(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    system_prompt = (
        "You are a warm robot tutor for a beginner programming concepts quiz. "
        "Write one short spoken response using the given situation and teaching "
        "move. Do not choose the strategy or invent new content."
    )
    replacements = {
        "beginner " + "Py" + "thon " + "quiz": "beginner programming concepts quiz",
        "beginner " + "Py" + "thon " + "learner": "beginner programming concepts learner",
        "learn beginner " + "Py" + "thon": "learn beginner programming concepts",
        "Py" + "thon " + "quiz": "programming concepts quiz",
        "Py" + "thon " + "question": "programming term question",
        "Situation: You are working through a beginner " + "Py" + "thon " + "quiz.": (
            "Situation: You are working through a beginner programming term question."
        ),
    }
    for old, new in replacements.items():
        user_prompt = user_prompt.replace(old, new)
    for rule in CONCEPT_QUIZ_RULES:
        if rule not in user_prompt:
            user_prompt = f"{user_prompt}\n{rule}"
    return system_prompt, user_prompt


def run_strategy_event(
    event,
    learner_state,
    rng,
    run_index,
    event_index,
    force_llm_for_content=False,
) -> tuple[dict, dict]:
    updated_learner_state = update_learner_state(learner_state, event)
    interaction_state = build_interaction_state(event)
    trigger = event["trigger"]
    response_behavior = event.get("response_behavior")
    if response_behavior in ("fast_correct", "fast_wrong"):
        obs = make_obs("engaged_fast")
    elif response_behavior in ("slow_correct", "slow_wrong"):
        obs = make_obs("hesitant")
    elif response_behavior == "no_answer":
        obs = make_obs("still_waiting")
    else:
        obs = make_obs("engaged_normal")
    turn_state = {
        "concept": event["concept"],
        "question_text": event["question_text"],
        "answer_status": event["answer_status"],
        "response_behavior": event["response_behavior"],
        "response_time_sec": event["response_time_sec"],
        "student_answer": event["student_answer"],
        "is_correct": event["is_correct"],
        "hesitation_high": event["hesitation_high"],
        "rapid_guess": event["rapid_guess"],
        "clarification_requested": False,
        "trigger": trigger,
    }
    strategy_decision = select_strategy(turn_state, updated_learner_state, trigger)
    strategy = strategy_decision["strategy"]
    quiz_context = {
        "concept": event["concept"],
        "question_text": event["question_text"],
    }
    system_prompt, user_prompt = assemble_prompt(
        obs=obs,
        trigger=trigger,
        interaction=interaction_state,
        learner=updated_learner_state,
        strategy=strategy,
        strategy_description=strategy_decision.get("strategy_description", ""),
        history=[],
        quiz_context=quiz_context,
    )
    system_prompt, user_prompt = _adapt_c3_prompt_for_concepts(
        system_prompt,
        user_prompt,
    )
    llm_result = _generate_response(
        obs,
        trigger,
        system_prompt,
        user_prompt,
        force_llm_for_content,
    )

    result = {
        "condition": CONDITION,
        "condition_name": CONDITION_NAME,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_index": run_index,
        "event_index": event_index,
        "scenario_name": event["scenario_name"],
        "event_type": event["event_type"],
        "setup_only": event["setup_only"],
        "concept": event["concept"],
        "question_text": event["question_text"],
        "correct_answer": event["correct_answer"],
        "accepted_answers": event["accepted_answers"],
        "student_answer": event["student_answer"],
        "is_correct": event["is_correct"],
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
        "expected_strategy": event["expected_strategy"],
        "selected_strategy": strategy,
        "strategy_correct": strategy == event["expected_strategy"],
        "strategy_reason": strategy_decision.get("reason", ""),
        "strategy_description": strategy_decision.get("strategy_description", ""),
        "learner_state": copy.deepcopy(updated_learner_state),
        "interaction_state": interaction_state,
        "obs": obs,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response_text": llm_result.get("response_text", ""),
        "llm_used": llm_result.get("llm_used", False),
        "response_source": (
            "llm" if llm_result.get("llm_used") else "deterministic_fallback"
        ),
        "llm_latency_sec": llm_result.get("llm_latency_sec", 0.0),
        "llm_prompt_sent": llm_result.get("llm_prompt_sent", ""),
        "llm_raw_response": llm_result.get("llm_raw_response", ""),
        "llm_retry_log": llm_result.get("llm_retry_log", []),
    }
    return result, updated_learner_state


def run_setup_event(event, learner_state, run_index, event_index) -> tuple[dict, dict]:
    updated_learner_state = update_learner_state(learner_state, event)
    interaction_state = build_interaction_state(event)
    result = {
        "condition": CONDITION,
        "condition_name": CONDITION_NAME,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_index": run_index,
        "event_index": event_index,
        "scenario_name": event["scenario_name"],
        "event_type": event["event_type"],
        "setup_only": True,
        "concept": event["concept"],
        "question_text": event["question_text"],
        "student_answer": event["student_answer"],
        "answer_status": event["answer_status"],
        "response_behavior": event["response_behavior"],
        "learner_state": copy.deepcopy(updated_learner_state),
        "interaction_state": interaction_state,
    }
    return result, updated_learner_state


def run_behavior_event(
    trigger,
    expected_strategy,
    learner_state,
    concept,
    question_text,
    run_index,
    event_index,
    force_llm_for_content=False,
) -> dict:
    obs = make_obs(trigger)
    interaction_state = {
        "hesitation_high": False,
        "rapid_guess": False,
        "no_answer": False,
        "last_response_behavior": None,
    }
    turn_state = {
        "concept": concept,
        "question_text": question_text,
        "answer_status": None,
        "response_behavior": None,
        "trigger": trigger,
    }
    strategy_decision = select_strategy(turn_state, learner_state, trigger)
    strategy = strategy_decision["strategy"]
    system_prompt, user_prompt = assemble_prompt(
        obs=obs,
        trigger=trigger,
        interaction=interaction_state,
        learner=learner_state,
        strategy=strategy,
        strategy_description=strategy_decision.get("strategy_description", ""),
        history=[],
        quiz_context={"concept": concept, "question_text": question_text},
    )
    system_prompt, user_prompt = _adapt_c3_prompt_for_concepts(
        system_prompt,
        user_prompt,
    )
    llm_result = _generate_response(
        obs,
        trigger,
        system_prompt,
        user_prompt,
        force_llm_for_content,
    )
    return {
        "condition": CONDITION,
        "condition_name": CONDITION_NAME,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_index": run_index,
        "event_index": event_index,
        "scenario_name": f"{trigger}_{expected_strategy}",
        "event_type": "behavior_event",
        "trigger": trigger,
        "concept": concept,
        "question_text": question_text,
        "expected_strategy": expected_strategy,
        "selected_strategy": strategy,
        "strategy_correct": strategy == expected_strategy,
        "strategy_reason": strategy_decision.get("reason", ""),
        "strategy_description": strategy_decision.get("strategy_description", ""),
        "learner_state": copy.deepcopy(learner_state),
        "interaction_state": interaction_state,
        "obs": obs,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response_text": llm_result.get("response_text", ""),
        "llm_used": llm_result.get("llm_used", False),
        "response_source": (
            "llm" if llm_result.get("llm_used") else "deterministic_fallback"
        ),
        "llm_latency_sec": llm_result.get("llm_latency_sec", 0.0),
        "llm_prompt_sent": llm_result.get("llm_prompt_sent", ""),
        "llm_raw_response": llm_result.get("llm_raw_response", ""),
        "llm_retry_log": llm_result.get("llm_retry_log", []),
    }


def run_hand_raise_event(
    spoken_request,
    expected_help_request_type,
    concept,
    question_text,
    run_index,
    event_index,
    use_llm=True,
) -> dict:
    obs = make_obs("hand_raise")
    hand_raise_result = handle_hand_raise(
        spoken_request,
        question_text=question_text,
        concept=concept,
        use_llm=use_llm,
    )
    help_request_type = hand_raise_result["help_request_type"]
    llm_used = hand_raise_result.get("llm_used", False)
    return {
        "condition": CONDITION,
        "condition_name": CONDITION_NAME,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_index": run_index,
        "event_index": event_index,
        "event_type": "hand_raise_help_action",
        "trigger": "hand_raised",
        "obs": obs,
        "spoken_request": spoken_request,
        "expected_help_request_type": expected_help_request_type,
        "help_request_type": help_request_type,
        "classification_correct": help_request_type == expected_help_request_type,
        "response_policy": hand_raise_result["response_policy"],
        "response_text": hand_raise_result["response_text"],
        "routes_to_strategy": hand_raise_result["routes_to_strategy"],
        "concept": concept,
        "question_text": question_text,
        "llm_used": llm_used,
        "llm_latency_sec": hand_raise_result.get("llm_latency_sec", 0.0),
        "llm_prompt_sent": hand_raise_result.get("llm_prompt_sent", ""),
        "llm_raw_response": hand_raise_result.get("llm_raw_response", ""),
        "llm_retry_log": hand_raise_result.get("llm_retry_log", []),
        "llm_backend": hand_raise_result.get("llm_backend", ""),
        "response_source": "llm" if llm_used else "deterministic_fallback",
    }


def build_event_sequence(rng: random.Random) -> list[dict]:
    events = [
        make_quiz_event(
            "praise",
            "variables",
            "What does a variable store?",
            "value",
            ["value"],
            "value",
            "correct",
            "fast",
            "praise",
            rng,
        ),
        make_quiz_event(
            "confidence_boost",
            "data_types",
            "What type stores decimal numbers?",
            "float",
            ["float"],
            "float",
            "correct",
            "slow",
            "confidence_boost",
            rng,
        ),
        make_quiz_event(
            "challenge",
            "conditionals",
            "What keyword starts a condition?",
            "if",
            ["if"],
            "if",
            "correct",
            "fast",
            "challenge",
            rng,
        ),
        make_quiz_event(
            "slow_down",
            "conditionals",
            "What operator needs both conditions true?",
            "and",
            ["and"],
            "equals",
            "incorrect",
            "fast",
            "slow_down",
            rng,
        ),
        make_quiz_event(
            "metacognitive_prompt",
            "variables",
            "What symbol assigns a value?",
            "equals",
            ["equals"],
            "value",
            "incorrect",
            "fast",
            "metacognitive_prompt",
            rng,
        ),
        make_quiz_event(
            "setup_reassure_low_confidence",
            "conditionals",
            "What word gives another choice?",
            "else",
            ["else"],
            "if",
            "incorrect",
            "fast",
            "setup_only",
            rng,
            setup_only=True,
        ),
        make_quiz_event(
            "setup_reassure_low_confidence_2",
            "loops",
            "What word repeats while a rule is true?",
            "while",
            ["while"],
            "for",
            "incorrect",
            "fast",
            "setup_only",
            rng,
            setup_only=True,
        ),
        make_quiz_event(
            "setup_reassure_low_confidence_3",
            "data_types",
            "What type stores true or false?",
            "boolean",
            ["boolean", "bool"],
            "string",
            "incorrect",
            "fast",
            "setup_only",
            rng,
            setup_only=True,
        ),
        make_quiz_event(
            "reassure",
            "variables",
            "What word means give a variable a value?",
            "assign",
            ["assign"],
            "value",
            "incorrect",
            "fast",
            "reassure",
            rng,
        ),
        make_quiz_event(
            "scaffold_hint_no_answer",
            "loops",
            "What keyword exits a loop immediately?",
            "break",
            ["break", "brake"],
            "",
            "no_answer",
            "no_answer",
            "scaffold_hint",
            rng,
        ),
        make_quiz_event(
            "setup_reset_variables_accuracy",
            "variables",
            "What can store text or numbers?",
            "variable",
            ["variable"],
            "variable",
            "correct",
            "fast",
            "setup_only",
            rng,
            setup_only=True,
        ),
        make_quiz_event(
            "setup_reset_conditionals_accuracy",
            "conditionals",
            "What word starts a condition?",
            "if",
            ["if"],
            "if",
            "correct",
            "fast",
            "setup_only",
            rng,
            setup_only=True,
        ),
        make_quiz_event(
            "give_example",
            "loops",
            "What keyword starts a counting loop?",
            "for",
            ["for", "four", "fore"],
            "function",
            "incorrect",
            "slow",
            "give_example",
            rng,
        ),
        make_quiz_event(
            "setup_reset_before_elaboration",
            "conditionals",
            "What word gives another branch?",
            "else",
            ["else"],
            "else",
            "correct",
            "fast",
            "setup_only",
            rng,
            setup_only=True,
        ),
        make_quiz_event(
            "setup_loop_recovery_before_elaboration",
            "loops",
            "What keyword repeats over a range?",
            "for",
            ["for", "four", "fore"],
            "for",
            "correct",
            "fast",
            "setup_only",
            rng,
            setup_only=True,
        ),
        make_quiz_event(
            "setup_data_types_no_answer_1",
            "data_types",
            "What type stores text?",
            "string",
            ["string", "str"],
            "",
            "no_answer",
            "no_answer",
            "setup_only",
            rng,
            setup_only=True,
        ),
        make_quiz_event(
            "setup_data_types_no_answer_2",
            "data_types",
            "What type stores whole numbers?",
            "integer",
            ["integer", "int"],
            "",
            "no_answer",
            "no_answer",
            "setup_only",
            rng,
            setup_only=True,
        ),
        make_quiz_event(
            "elaborate_concept",
            "data_types",
            "What type stores key value pairs?",
            "dictionary",
            ["dictionary", "dict"],
            "list",
            "incorrect",
            "slow",
            "elaborate_concept",
            rng,
        ),
    ]
    return events


def build_summary(
    target_strategy_events,
    setup_events,
    behavior_events,
    hand_raise_events,
    final_learner_state,
) -> dict:
    target_correct = sum(1 for event in target_strategy_events if event["strategy_correct"])
    behavior_correct = sum(1 for event in behavior_events if event["strategy_correct"])
    hand_correct = sum(
        1 for event in hand_raise_events if event["classification_correct"]
    )
    all_response_events = target_strategy_events + behavior_events
    llm_used_count = sum(1 for event in all_response_events if event.get("llm_used"))
    fallback_count = sum(
        1
        for event in all_response_events
        if event.get("response_source") == "deterministic_fallback"
    )
    latencies = [
        event.get("llm_latency_sec", 0.0)
        for event in all_response_events
        if event.get("llm_latency_sec") is not None
    ]
    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    strategies_covered = sorted(
        {event["selected_strategy"] for event in all_response_events}
    )

    return {
        "strategy_summary": {
            "target_strategy_events": len(target_strategy_events),
            "strategy_correct_count": target_correct,
            "strategy_accuracy_pct": round(
                100 * target_correct / len(target_strategy_events), 1
            ) if target_strategy_events else 0.0,
            "strategies_covered": strategies_covered,
            "failed_strategy_scenarios": [
                event["scenario_name"]
                for event in target_strategy_events
                if not event["strategy_correct"]
            ],
        },
        "behavior_summary": {
            "behavior_events_count": len(behavior_events),
            "behavior_triggers_covered": sorted(
                {event["trigger"] for event in behavior_events}
            ),
            "behavior_strategy_correct_count": behavior_correct,
            "behavior_strategy_accuracy_pct": round(
                100 * behavior_correct / len(behavior_events), 1
            ) if behavior_events else 0.0,
        },
        "hand_raise_summary": {
            "hand_raise_total": len(hand_raise_events),
            "hand_raise_correct_count": hand_correct,
            "hand_raise_accuracy_pct": round(
                100 * hand_correct / len(hand_raise_events), 1
            ) if hand_raise_events else 0.0,
            "help_request_types_covered": sorted(
                {event["help_request_type"] for event in hand_raise_events}
            ),
            "separate_help_actions": HAND_RAISE_HELP_ACTIONS,
            "hand_raise_llm_used_count": sum(
                1 for event in hand_raise_events if event.get("llm_used")
            ),
            "hand_raise_deterministic_count": sum(
                1
                for event in hand_raise_events
                if event.get("response_source") == "deterministic_fallback"
            ),
        },
        "state_summary": {
            "final_learner_state": final_learner_state,
        },
        "llm_summary": {
            "llm_used_count": llm_used_count,
            "deterministic_fallback_count": fallback_count,
            "avg_llm_latency_sec": avg_latency,
        },
        "total_events_logged": (
            len(target_strategy_events)
            + len(setup_events)
            + len(behavior_events)
            + len(hand_raise_events)
        ),
        "setup_events_count": len(setup_events),
    }


def run_session(run_index, seed, force_llm_for_content, hand_raise_use_llm=True):
    rng = random.Random(seed + run_index - 1)
    learner_state = initial_learner_state()
    target_strategy_events = []
    setup_events = []
    behavior_events = []
    hand_raise_events = []
    event_index = 1

    for event in build_event_sequence(rng):
        if event["setup_only"]:
            result, learner_state = run_setup_event(
                event,
                learner_state,
                run_index,
                event_index,
            )
            setup_events.append(result)
        else:
            result, learner_state = run_strategy_event(
                event,
                learner_state,
                rng,
                run_index,
                event_index,
                force_llm_for_content=force_llm_for_content,
            )
            target_strategy_events.append(result)
        event_index += 1

    current_concept = "variables"
    current_question = "What does a variable store?"
    behavior_specs = [
        ("head_turned", "re_engage"),
        ("face_absent", "re_engage"),
        ("no_movement", "stillness_check"),
    ]
    for trigger, expected_strategy in behavior_specs:
        behavior_events.append(
            run_behavior_event(
                trigger,
                expected_strategy,
                learner_state,
                current_concept,
                current_question,
                run_index,
                event_index,
                force_llm_for_content=force_llm_for_content,
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
    for spoken_request, expected_help_request_type in hand_specs:
        hand_raise_events.append(
            run_hand_raise_event(
                spoken_request,
                expected_help_request_type,
                current_concept,
                current_question,
                run_index,
                event_index,
                use_llm=hand_raise_use_llm,
            )
        )
        event_index += 1

    return (
        target_strategy_events,
        setup_events,
        behavior_events,
        hand_raise_events,
        learner_state,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run realistic scripted C3 strategy pipeline scenarios."
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-llm-for-content", action="store_true")
    parser.add_argument("--no-hand-raise-llm", action="store_true")
    args = parser.parse_args()

    runs = max(1, args.runs)
    all_target_events = []
    all_setup_events = []
    all_behavior_events = []
    all_hand_raise_events = []
    final_learner_state = initial_learner_state()

    for run_index in range(1, runs + 1):
        print(f"Running C3 scripted session {run_index}/{runs}")
        (
            target_events,
            setup_events,
            behavior_events,
            hand_raise_events,
            final_learner_state,
        ) = run_session(
            run_index,
            args.seed,
            args.force_llm_for_content,
            hand_raise_use_llm=not args.no_hand_raise_llm,
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
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "condition": CONDITION,
        "condition_name": CONDITION_NAME,
        "runs": runs,
        "target_strategy_events": all_target_events,
        "setup_events": all_setup_events,
        "behavior_events": all_behavior_events,
        "hand_raise_events": all_hand_raise_events,
        "summary": summary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"C3_strategy_pipeline_{timestamp}.json"
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    strategy_summary = summary["strategy_summary"]
    behavior_summary = summary["behavior_summary"]
    hand_summary = summary["hand_raise_summary"]
    llm_summary = summary["llm_summary"]

    print("\nC3 STRATEGY PIPELINE SUMMARY")
    print(f"Runs: {runs}")
    print(
        "Target strategy accuracy: "
        f"{strategy_summary['strategy_correct_count']}/"
        f"{strategy_summary['target_strategy_events']} "
        f"({strategy_summary['strategy_accuracy_pct']}%)"
    )
    print(
        "Behavior strategy accuracy: "
        f"{behavior_summary['behavior_strategy_correct_count']}/"
        f"{behavior_summary['behavior_events_count']} "
        f"({behavior_summary['behavior_strategy_accuracy_pct']}%)"
    )
    print(
        "Hand-raise classification accuracy: "
        f"{hand_summary['hand_raise_correct_count']}/"
        f"{hand_summary['hand_raise_total']} "
        f"({hand_summary['hand_raise_accuracy_pct']}%)"
    )
    print(f"Hand-raise LLM used: {hand_summary['hand_raise_llm_used_count']}")
    print(f"Hand-raise deterministic: {hand_summary['hand_raise_deterministic_count']}")
    print(f"Strategies covered: {strategy_summary['strategies_covered']}")
    print(f"LLM used: {llm_summary['llm_used_count']}")
    print(f"Deterministic fallback: {llm_summary['deterministic_fallback_count']}")
    print(f"Total events logged: {summary['total_events_logged']}")
    print(f"Saved: {output_path}")

    failed = (
        strategy_summary["failed_strategy_scenarios"]
        or [
            event["scenario_name"]
            for event in all_behavior_events
            if not event["strategy_correct"]
        ]
        or [
            event["spoken_request"]
            for event in all_hand_raise_events
            if not event["classification_correct"]
        ]
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
