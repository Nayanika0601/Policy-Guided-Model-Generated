import argparse
import json
import os
import time

import requests


HELP_REQUEST_TYPES = [
    "ask_question",
    "repeat_question",
    "hint_request",
    "concept_explanation",
    "clarify_help_request",
]


_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")


def _load_settings() -> dict:
    try:
        with open(os.path.join(_CFG_DIR, "settings.json"), "r") as f:
            return json.load(f)
    except Exception:
        return {}


_SETTINGS = _load_settings()
OLLAMA_URL = _SETTINGS.get("ollama_url", "http://localhost:11434/api/generate")
OLLAMA_MODEL = _SETTINGS.get("ollama_model", "llama3.1:8b")
OLLAMA_TIMEOUT = _SETTINGS.get(
    "ollama_timeout",
    _SETTINGS.get("llm_timeout_sec", 20),
)
LLM_BACKEND = _SETTINGS.get("llm_backend", "ollama")

SAFE_CLARIFY_RESPONSE = (
    "What would you like help with: repeating, a hint, or an explanation?"
)

CODE_CONTEXT_TERMS = [
    "python code",
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
    "vocabulary",
    "vocabulory",
]


def classify_help_request(text: str) -> str:
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return "clarify_help_request"

    if any(
        phrase in cleaned
        for phrase in ("repeat", "again", "say that", "what was the question")
    ):
        return "repeat_question"

    if any(
        phrase in cleaned
        for phrase in ("hint", "clue", "help me", "small help")
    ):
        return "hint_request"

    if any(
        phrase in cleaned
        for phrase in ("explain", "what is", "what does", "mean", "concept")
    ):
        return "concept_explanation"

    if any(
        phrase in cleaned
        for phrase in ("question", "ask", "why", "how", "can i")
    ):
        return "ask_question"

    return "clarify_help_request"


def build_hand_raise_prompt(
    help_request_type: str,
    question_text: str = "",
    concept: str = "",
    spoken_request: str = "",
) -> tuple[str, str, str]:
    system_prompt = (
        "You are a warm robot tutor helping an adult student learn beginner "
        "programming concepts. Write one short spoken response for a "
        "hand-raise help request. Do not decide the help request type."
    )
    common_rules = [
        "Rules:",
        "- Output exactly one sentence.",
        "- Do not add extra explanation.",
        "- After the sentence, stop immediately.",
        "- 8 to 18 words.",
        "- Use one short spoken sentence.",
        "- Speak directly using \"you\".",
        "- This is a beginner programming concepts quiz, not a code-writing task.",
        "- Do not mention code, syntax, print, console, function, indentation, output, expression, error messages, or code snippets.",
        "- Do not assume the learner is writing code.",
        "- Do not reveal the answer unless the learner explicitly asks for a concept explanation.",
    ]

    if help_request_type == "ask_question":
        user_prompt = "\n".join([
            "Situation: The learner raised a hand and wants to ask a question.",
            f"Spoken request: {spoken_request}",
            *common_rules,
            "- Invite the learner to ask their question.",
            "- Do not answer a question yet.",
        ])
        fallback = "Sure, go ahead and ask your question."
    elif help_request_type == "repeat_question":
        user_prompt = ""
        if question_text:
            fallback = f"Sure, the question is: {question_text}"
        else:
            fallback = "Sure, I can repeat the question."
    elif help_request_type == "hint_request":
        user_prompt = "\n".join([
            "Situation: The learner raised a hand and asked for a hint.",
            f"Current concept: {concept}",
            f"Current question: {question_text}",
            *common_rules,
            "- Give a small term hint.",
            "- Do not reveal the answer.",
        ])
        fallback = "Think about what the question is asking the term to store or represent."
    elif help_request_type == "concept_explanation":
        user_prompt = "\n".join([
            "Situation: The learner raised a hand and asked for a concept explanation.",
            f"Concept: {concept}",
            f"Spoken request: {spoken_request}",
            *common_rules,
            "- Offer to briefly explain the concept.",
            "- Do not answer the current quiz question.",
        ])
        if concept:
            fallback = f"Sure, I can briefly explain {concept}."
        else:
            fallback = "Sure, I can briefly explain this concept."
    else:
        user_prompt = "\n".join([
            "Situation: The learner raised a hand, but the help request was unclear.",
            *common_rules,
            "- Ask what kind of help they want.",
            "- Offer repeat, hint, or explanation.",
        ])
        fallback = SAFE_CLARIFY_RESPONSE

    return system_prompt, user_prompt, fallback


def generate_hand_raise_response(
    help_request_type: str,
    question_text: str = "",
    concept: str = "",
    spoken_request: str = "",
    use_llm: bool = True,
) -> dict:
    system_prompt, user_prompt, fallback = build_hand_raise_prompt(
        help_request_type,
        question_text=question_text,
        concept=concept,
        spoken_request=spoken_request,
    )

    if help_request_type == "repeat_question" or not use_llm:
        return _fallback_result(fallback, user_prompt)

    retry_log = []
    llm_raw_response = ""
    t0 = time.time()
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 60,
            },
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        latency = round(time.time() - t0, 3)
        if response.status_code != 200:
            retry_log.append({
                "attempt": 1,
                "rejection_reason": f"http_{response.status_code}",
                "raw_response": "",
            })
            return _fallback_result(fallback, user_prompt, latency, retry_log)

        llm_raw_response = response.json().get("response", "").strip()
        text = _clean_response(llm_raw_response)
        rejection_reason = _validation_rejection_reason(text, help_request_type)
        if rejection_reason:
            retry_log.append({
                "attempt": 1,
                "rejection_reason": rejection_reason,
                "raw_response": llm_raw_response,
            })
            return _fallback_result(fallback, user_prompt, latency, retry_log, llm_raw_response)

        retry_log.append({
            "attempt": 1,
            "rejection_reason": None,
            "raw_response": llm_raw_response,
        })
        return {
            "response_text": text,
            "llm_used": True,
            "llm_latency_sec": latency,
            "llm_prompt_sent": user_prompt,
            "llm_raw_response": llm_raw_response,
            "llm_retry_log": retry_log,
            "llm_backend": LLM_BACKEND,
        }
    except requests.exceptions.RequestException as exc:
        latency = round(time.time() - t0, 3)
        retry_log.append({
            "attempt": 1,
            "rejection_reason": exc.__class__.__name__,
            "raw_response": llm_raw_response,
        })
        return _fallback_result(fallback, user_prompt, latency, retry_log, llm_raw_response)
    except Exception as exc:
        latency = round(time.time() - t0, 3)
        retry_log.append({
            "attempt": 1,
            "rejection_reason": f"error: {exc}",
            "raw_response": llm_raw_response,
        })
        return _fallback_result(fallback, user_prompt, latency, retry_log, llm_raw_response)


def _fallback_result(
    fallback: str,
    prompt: str,
    latency: float = 0.0,
    retry_log: list | None = None,
    raw_response: str = "",
) -> dict:
    if _has_code_context(fallback):
        fallback = SAFE_CLARIFY_RESPONSE
    return {
        "response_text": fallback,
        "llm_used": False,
        "llm_latency_sec": latency,
        "llm_prompt_sent": prompt,
        "llm_raw_response": raw_response,
        "llm_retry_log": retry_log or [],
        "llm_backend": LLM_BACKEND,
    }


def _clean_response(text: str) -> str:
    text = (text or "").strip().strip('"\'').strip()
    text = text.splitlines()[0].strip() if text else ""
    if ":" in text and text.index(":") < 18:
        text = text.split(":", 1)[1].strip()
    for marker in ("\n\n", "Student:", "The answer", "Answer:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    words = text.split()
    if len(words) > 25:
        text = " ".join(words[:25]).rstrip(",;:")
        if not text.endswith((".", "?", "!")):
            text += "."
    return text.strip().strip('"\'').strip()


def _validation_rejection_reason(text: str, help_request_type: str) -> str | None:
    if not text:
        return "empty_response"

    word_count = len(text.split())
    if word_count < 3 or word_count > 25:
        return f"word_count_{word_count}"

    lower = text.lower()
    if _has_code_context(text):
        return "code_context_hallucination"

    if help_request_type == "ask_question":
        banned = ("the answer", "you should", "use ")
        if any(phrase in lower for phrase in banned):
            return "ask_question_answering_content"
    elif help_request_type == "hint_request":
        banned = ("the answer", "is", "use")
        if any(phrase in lower for phrase in banned):
            return "hint_request_answer_leak"
    elif help_request_type == "concept_explanation":
        banned = ("the answer", "use ", "keyword")
        if any(phrase in lower for phrase in banned):
            return "concept_explanation_answer_leak"
    elif help_request_type == "clarify_help_request":
        if "?" not in text or "help" not in lower:
            return "clarify_help_request_not_question"

    return None


def _has_code_context(text: str) -> bool:
    lower = (text or "").lower()
    return any(term in lower for term in CODE_CONTEXT_TERMS)


def build_hand_raise_response(
    help_request_type: str,
    question_text: str = "",
    concept: str = "",
    spoken_request: str = "",
) -> dict:
    if help_request_type not in HELP_REQUEST_TYPES:
        help_request_type = "clarify_help_request"

    if help_request_type == "ask_question":
        response_policy = "ask_question"
        response_text = "Sure, go ahead and ask your question."
        routes_to_strategy = None
    elif help_request_type == "repeat_question":
        response_policy = "repeat_question"
        if question_text:
            response_text = f"Sure, the question is: {question_text}"
        else:
            response_text = "Sure, I can repeat the question."
        routes_to_strategy = None
    elif help_request_type == "hint_request":
        response_policy = "scaffold_hint"
        response_text = "Sure, I can give you a small hint."
        routes_to_strategy = "scaffold_hint"
    elif help_request_type == "concept_explanation":
        response_policy = "concept_explanation"
        if concept:
            response_text = f"Sure, I can briefly explain {concept}."
        else:
            response_text = "Sure, I can briefly explain the concept."
        routes_to_strategy = None
    else:
        response_policy = "clarify_help_request"
        response_text = (
            "What would you like help with: repeating, a hint, or an explanation?"
        )
        routes_to_strategy = None

    return {
        "help_request_type": help_request_type,
        "response_policy": response_policy,
        "response_text": response_text,
        "routes_to_strategy": routes_to_strategy,
    }


def handle_hand_raise(
    spoken_request: str,
    question_text: str = "",
    concept: str = "",
    use_llm: bool = True,
) -> dict:
    help_request_type = classify_help_request(spoken_request)
    deterministic_response = build_hand_raise_response(
        help_request_type,
        question_text=question_text,
        concept=concept,
        spoken_request=spoken_request,
    )
    generated_response = generate_hand_raise_response(
        help_request_type,
        question_text=question_text,
        concept=concept,
        spoken_request=spoken_request,
        use_llm=use_llm,
    )
    return {
        "trigger": "hand_raised",
        "spoken_request": spoken_request,
        "help_request_type": deterministic_response["help_request_type"],
        "response_policy": deterministic_response["response_policy"],
        "response_text": generated_response["response_text"],
        "routes_to_strategy": deterministic_response["routes_to_strategy"],
        "concept": concept,
        "question_text": question_text,
        "llm_used": generated_response["llm_used"],
        "llm_latency_sec": generated_response["llm_latency_sec"],
        "llm_prompt_sent": generated_response["llm_prompt_sent"],
        "llm_raw_response": generated_response["llm_raw_response"],
        "llm_retry_log": generated_response["llm_retry_log"],
        "llm_backend": generated_response["llm_backend"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()

    tests = [
        ("I have a question", "ask_question"),
        ("can you repeat the question", "repeat_question"),
        ("can I get a hint", "hint_request"),
        ("what is a variable", "concept_explanation"),
        ("", "clarify_help_request"),
    ]

    passed = 0
    for text, expected in tests:
        event = handle_hand_raise(
            text,
            question_text="What does a variable store?",
            concept="variables",
            use_llm=False,
        )
        result = event["help_request_type"]
        ok = result == expected
        print(
            f"{'PASS' if ok else 'FAIL'}: {text!r} -> {result}; "
            f"response={event['response_text']!r}"
        )
        if ok:
            passed += 1

    print(f"Result: {passed}/{len(tests)} PASS")
    assert passed == len(tests)

    if args.llm:
        print("\nLLM examples:")
        for text, _expected in tests:
            event = handle_hand_raise(
                text,
                question_text="What does a variable store?",
                concept="variables",
                use_llm=True,
            )
            source = "llm" if event["llm_used"] else "fallback"
            print(f"- {text!r} [{source}]: {event['response_text']}")
