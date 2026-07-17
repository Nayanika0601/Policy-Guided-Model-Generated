
import time
import re
import json
import os
import random
import requests


FALLBACK_PHRASES = [
    "Take your time, I'm right here with you.",
    "No rush at all, we can continue whenever you're ready.",
    "All good, just let me know when you want to keep going.",
    "I'm here whenever you need me, no pressure.",
    "Whenever you're ready, we can carry on together.",
]

_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")

with open(os.path.join(_CFG_DIR, "settings.json"), "r") as f:
    _settings = json.load(f)

with open(os.path.join(_CFG_DIR, "prompts_mode_a.json"), "r") as f:
    _prompts_a = json.load(f)

with open(os.path.join(_CFG_DIR, "prompts_mode_b.json"), "r") as f:
    _prompts_b = json.load(f)

OLLAMA_URL     = _settings["ollama_url"]
OLLAMA_MODEL   = _settings["ollama_model"]
OLLAMA_TIMEOUT = _settings["llm_timeout_sec"]
LLM_MODE       = _settings["llm_mode"]
HISTORY_LENGTH = _settings["llm_history_length"]
MIN_WORDS      = _settings["llm_min_words"]
MAX_WORDS      = _settings["llm_max_words"]
LLM_BACKEND    = _settings["llm_backend"]

_history: list[str] = []


def _add_to_history(text: str) -> None:
    _history.append(text)
    while len(_history) > HISTORY_LENGTH:
        _history.pop(0)


def _history_block() -> str:
    if not _history:
        return ""
    lines = "\n".join(f"- {h}" for h in _history)
    return (
        "Recent responses you have already given (do not repeat or paraphrase these — say something completely different):\n"
        f"{lines}\n\n"
    )


def _build_obs_block(obs: dict) -> str:
    face_detected = obs.get("face_detected", False)
    face = "visible" if face_detected else "not visible"
    body = "detected" if obs.get("body_detected", False) else "not detected"

    energy = obs.get("speech_energy", "low")
    prompted = "yes" if obs.get("already_prompted", False) else "no"

    if face_detected:
        gaze = f"{obs.get('gaze_on_robot', 0.0):.2f} out of 1.0"
        yaw  = f"{obs.get('head_yaw_deg', 0.0):.1f} degrees"
        pitch = f"{obs.get('head_pitch_deg', 0.0):.1f} degrees"
        head = "moving" if obs.get("head_moving", False) else "still"
        no_mov = f"{obs.get('no_movement_sec', 0.0):.0f}"
        hand = "yes" if obs.get("hand_raised", False) else "no"
        hand_side = obs.get("hand_raise_side", "none")
    else:
        gaze = "unknown (face not visible)"
        yaw  = "unknown (face not visible)"
        pitch = "unknown (face not visible)"
        head = "unknown (face not visible)"
        no_mov = "unknown (face not visible)"
        hand = "unknown (face not visible)"
        hand_side = "unknown (face not visible)"

    return (
        "Student state right now:\n"
        f"- Face: {face}\n"
        f"- Body: {body}\n"
        f"- Gaze on robot: {gaze}\n"
        f"- Head yaw: {yaw}\n"
        f"- Head pitch: {pitch}\n"
        f"- Head: {head}\n"
        f"- No movement for: {no_mov} seconds\n"
        f"- Speech energy: {energy}\n"
        f"- Hand raised: {hand}\n"
        f"- Already prompted this window: {prompted}\n"
        f"- Hand raise side: {hand_side}\n"
    )


def _build_prompt_a(obs: dict, trigger: str, extra_context: str | None,
                    mode: str) -> tuple[str, str, str]:
    sys_key = "system_prompt_quiz" if mode == "quiz" else "system_prompt_monitor"
    system = _prompts_a[sys_key]

    trig_data = _prompts_a["triggers"].get(trigger, {})
    prompt_key = "quiz" if mode == "quiz" else "monitor"
    situation = trig_data.get(prompt_key, "Say something encouraging and kind. Respond in between 5 and 25 words.")
    if "{student_question}" in situation and extra_context:
        situation = situation.replace("{student_question}", extra_context)

    fallback_key = "fallback_quiz" if mode == "quiz" else "fallback_monitor"
    fallback = trig_data.get(fallback_key, "I'm here with you!")

    obs_block = _build_obs_block(obs)
    history = _history_block()

    prompt = f"{obs_block}\n{history}{situation}"
    return system, prompt, fallback


def _build_prompt_b(obs: dict, trigger: str, extra_context: str | None,
                    mode: str) -> tuple[str, str, str]:
    sys_key = "system_prompt_quiz" if mode == "quiz" else "system_prompt_monitor"
    system = _prompts_b[sys_key]

    prompt_key = "prompt_quiz" if mode == "quiz" else "prompt_monitor"
    situation = _prompts_b[prompt_key]
    fallback_key = "fallback_quiz" if mode == "quiz" else "fallback_monitor"
    fallback = _prompts_b[fallback_key]

    obs_block = _build_obs_block(obs)

    if extra_context:
        obs_block += f"- Student just said: {extra_context}\n"

    history = _history_block()

    prompt = f"{obs_block}\n{history}{situation}"
    return system, prompt, fallback


_STOP_MARKERS = [
    "Student:", "student:",
    "Now provide", "now provide",
    "As an AI", "as an AI",
    "Note:", "(Note",
    "I am an AI", "i am an ai",
    "developed by Microsoft", "developed by Anthropic",
    "provide an answer",
    "(Word count", "(word count",
    "(Total word count", "(total word count",
    "-----",
    "Prompting the",
    "The child is",
    "The student is",
    "robot tutor",
    "tutoring robot",
    "\n\n",
]


def _clean_text(text: str) -> str:
    text = text.split("\n")[0].strip()
    text = text.strip('"\'').strip()
    if ":" in text and text.index(":") < 15:
        text = text.split(":", 1)[1].strip()
    text = text.strip('"\'').strip()
    if text.startswith("- "):
        text = text[2:]
    text = re.sub(r"['\u2018\u2019\u0092]", "'", text)
    text = re.sub(r"Let'se", "Let's e", text)
    text = re.sub(r"Let'self", "Let's", text)
    text = re.sub(r"(\w)'s([a-z])", r"\1's \2", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", "", text).strip()
    for marker in _STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()
    if len(text.split()) > 35:
        match = re.search(r'[.?!]', text)
        if match:
            text = text[:match.end()].strip()
    return text


def _is_invalid_response(text: str) -> bool:
    lower = text.lower()

    SENSOR_FIELDS = [
        "gaze", "pitch", "no_movement", "yaw",
        "face_detected", "body_detected", "head_yaw",
        "head_pitch", "speech_energy", "hand_raise_side",
        "sensor", "detected by", "not detected",
        "heart rate", "body is not", "head is not",
        "data_types", "weak_concept", "wrong_streak", "correct_streak",
        "rolling_accuracy", "hesitation_time_sec", "rapid_guess",
        "clarification_requested", "learner_state", "interaction_state",
    ]

    THIRD_PERSON = [
        "their gaze", "the student", "the child",
        "they have been", "they are", "they seem",
        "student might", "child might", "student needs",
        "child needs", "the learner", "this child",
        "student appears", "child appears",
        "student is", "child is", "student has",
        "child has", "student seems", "child seems",
        "student was", "child was", "student looks",
        "child looks", "student could", "child could",
    ]

    for field in SENSOR_FIELDS:
        if field in lower:
            print(f"  [llm] Rejected — sensor field '{field}' leaked in response")
            return True

    for phrase in THIRD_PERSON:
        if phrase in lower:
            print(f"  [llm] Rejected — third person reference '{phrase}' in response")
            return True

    return False


def _has_answer_clue_leak(text: str) -> bool:
    lower = text.lower()
    banned_phrases = [
        "single word",
        "one word",
        "single character",
        "one character",
        "first letter",
        "last letter",
        "starts with",
        "ends with",
        "sounds like",
        "rhymes with",
        "the answer is",
    ]
    return any(p in lower for p in banned_phrases)


_STRATEGY_FALLBACKS = {
    "praise": "Nice work, that answer is correct.",
    "confidence_boost": "Correct, nice job thinking that through carefully.",
    "challenge": "Nice work, keep using that careful thinking on the next one.",
    "slow_down": "Take a moment and reason through the question before answering.",
    "scaffold_hint": "Try focusing on the main clue in the question.",
    "metacognitive_prompt": "What clue in the question helped you choose that answer?",
    "reassure": "That's okay, take a breath and try the next step carefully.",
    "re_engage": "Let's come back to the quiz when you're ready.",
    "stillness_check": "Are you ready to continue, or do you need a moment?",
}


_CONCEPT_EXAMPLE_FALLBACKS = {
    "variables": "For example, a variable can store a score or a name.",
    "data_types": "For example, text and numbers are different kinds of Python data.",
    "loops": "For example, a loop can repeat a print action several times.",
    "conditionals": "For example, a condition can choose what code runs next.",
}


_CONCEPT_EXPLANATION_FALLBACKS = {
    "variables": "A variable is a name that stores a value for later use.",
    "data_types": "Data types describe what kind of value Python is working with.",
    "loops": "Loops let Python repeat code while a rule or sequence continues.",
    "conditionals": "Conditionals let Python choose what to do based on a test.",
}


_SCAFFOLD_HINT_FALLBACKS = {
    "variables": "Think about how Python keeps information so it can use it later.",
    "data_types": "Think about what kind of information the value represents.",
    "loops": "Think about how Python controls when repetition should stop or continue.",
    "conditionals": "Think about how Python chooses between different paths.",
}


_SCAFFOLD_FORBIDDEN_WORDS = {
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


_SCAFFOLD_TOO_DIRECT_PHRASES = [
    "the answer",
    "keyword is",
    "use the keyword",
    "use ",
    "look for a keyword",
    "word that means",
]


_STRATEGY_INSTRUCTION_MARKERS = {
    "Briefly praise the correct answer. Do not ask another question.": "praise",
    (
        "Say the answer was correct and encourage the learner for thinking "
        "carefully. Do not ask another question."
    ): "confidence_boost",
    (
        "Encourage the learner to keep thinking carefully after strong performance."
    ): "challenge",
    "Encourage careful reasoning before answering.": "slow_down",
    "Give one small hint without revealing the answer.": "scaffold_hint",
    (
        "Ask the learner to briefly explain how they are thinking about the question."
    ): "metacognitive_prompt",
    "Offer encouragement and reduce frustration.": "reassure",
    (
        "Give one simple example related to the current concept without revealing the answer."
    ): "give_example",
    (
        "Briefly explain the current concept in beginner-friendly language without revealing the answer."
    ): "elaborate_concept",
    "Gently bring attention back to the task.": "re_engage",
    "Briefly check in and invite continuation.": "stillness_check",
}


def _infer_strategy_from_prompt(prompt: str) -> str:
    for marker, strategy in _STRATEGY_INSTRUCTION_MARKERS.items():
        if marker in prompt:
            return strategy
    return ""


def _infer_concept_from_prompt(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Concept:"):
            return line.split(":", 1)[1].strip()
    return ""


def _strategy_fallback(strategy: str, prompt: str, default_fallback: str) -> str:
    concept = _infer_concept_from_prompt(prompt)
    if strategy == "give_example":
        return _CONCEPT_EXAMPLE_FALLBACKS.get(
            concept,
            "For example, think of a simple Python case using this concept.",
        )
    if strategy == "elaborate_concept":
        return _CONCEPT_EXPLANATION_FALLBACKS.get(
            concept,
            "This concept helps Python decide how to work with information.",
        )
    if strategy == "scaffold_hint":
        return _SCAFFOLD_HINT_FALLBACKS.get(
            concept,
            "Try focusing on the main clue in the question.",
        )
    return _STRATEGY_FALLBACKS.get(strategy, default_fallback)


def _strategy_rejection_reason(text: str, strategy: str) -> str | None:
    lower = text.lower()

    if strategy in ("praise", "confidence_boost"):
        banned = [
            "close",
            "almost",
            "try again",
            "what type",
            "what keyword",
            "can you tell",
            "let's think this through",
        ]
        if any(phrase in lower for phrase in banned):
            return f"{strategy}_unsafe_feedback"

    if strategy == "metacognitive_prompt":
        banned = ["the answer", "it is", "you should use"]
        if any(phrase in lower for phrase in banned):
            return "metacognitive_answer_or_teaching"

    if strategy == "give_example":
        if "example" not in lower and "for example" not in lower:
            return "give_example_missing_example"

    if strategy == "elaborate_concept":
        banned = [
            "probably",
            "right?",
            "can you",
            "what type",
            "what keyword",
            "you're thinking of",
        ]
        if any(phrase in lower for phrase in banned):
            return "elaborate_concept_unsafe"

    return None


def _scaffold_rejection_reason(text: str) -> str | None:
    lower = text.lower()
    if any(phrase in lower for phrase in _SCAFFOLD_TOO_DIRECT_PHRASES):
        return "scaffold_too_direct"

    words = set(re.findall(r"\b[a-zA-Z_]+\b", lower))
    if words & _SCAFFOLD_FORBIDDEN_WORDS:
        return "scaffold_answer_leak"

    return None


def _is_too_similar(text: str) -> bool:
    if not _history:
        return False

    new_words = set(text.lower().split())
    if not new_words:
        return False

    for prev in _history:
        prev_words = set(prev.lower().split())
        if not prev_words:
            continue
        overlap = new_words & prev_words
        smaller = min(len(new_words), len(prev_words))
        ratio = len(overlap) / smaller
        if ratio >= 0.6:
            print(f"  [llm] Rejected — too similar to previous response "
                  f"({ratio:.0%} word overlap)")
            return True

    return False


def generate(obs: dict, trigger: str,
             extra_context: str | None = None,
             mode: str = "quiz",
             assembled_prompt: tuple | None = None) -> dict:

    active_mode = LLM_MODE
    is_strategy_prompt = assembled_prompt is not None
    effective_max_words = min(MAX_WORDS, 18) if is_strategy_prompt else MAX_WORDS
    prompt_strategy = ""

    if assembled_prompt is not None:
        system, prompt = assembled_prompt
        prompt_strategy = _infer_strategy_from_prompt(prompt)
        fallback = _strategy_fallback(
            prompt_strategy,
            prompt,
            random.choice(FALLBACK_PHRASES),
        )
    elif active_mode == "b":
        system, prompt, fallback = _build_prompt_b(obs, trigger, extra_context, mode)
    else:
        system, prompt, fallback = _build_prompt_a(obs, trigger, extra_context, mode)

    llm_prompt_sent = prompt
    llm_raw_response = ""

    MAX_RETRIES = 5
    temperatures = [0.7, 0.9, 1.0, 1.1, 1.2]
    retry_log = []

    try:
        t0 = time.time()

        for retry in range(MAX_RETRIES):
            temp = temperatures[retry]
            if retry > 0:
                print(f"  [llm] Retry {retry}/{MAX_RETRIES - 1} "
                      f"(temperature={temp})")

            payload = {
                "model": OLLAMA_MODEL,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temp,
                    "num_predict": 60,
                },
            }

            resp = None
            for attempt in range(2):
                resp = requests.post(OLLAMA_URL, json=payload,
                                     timeout=OLLAMA_TIMEOUT)
                if resp.status_code == 200:
                    break
                if resp.status_code == 500 and attempt == 0:
                    print(f"  [llm] HTTP 500, retrying in 2s ...")
                    time.sleep(2)
                    continue
                break

            latency = round(time.time() - t0, 3)

            if resp.status_code != 200:
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": f"http_{resp.status_code}",
                    "raw_response": "",
                })
                print(f"  [llm] HTTP {resp.status_code} after retry, using fallback")
                return {"response_text": fallback, "llm_used": False,
                        "llm_latency_sec": latency,
                        "llm_prompt_sent": llm_prompt_sent,
                        "llm_raw_response": llm_raw_response,
                        "llm_retry_log": retry_log,
                        "llm_backend": LLM_BACKEND}

            data = resp.json()
            llm_raw_response = data.get("response", "").strip()
            text = _clean_text(llm_raw_response)

            if not text or len(text) < 3:
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": "empty_or_too_short",
                    "raw_response": llm_raw_response,
                })
                print(f"  [llm] Empty/too-short response, retrying...")
                continue

            if _is_invalid_response(text):
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": "invalid_response (sensor_leak or third_person)",
                    "raw_response": llm_raw_response,
                })
                continue

            scaffold_rejection = (
                _scaffold_rejection_reason(text)
                if is_strategy_prompt and prompt_strategy == "scaffold_hint"
                else None
            )
            if scaffold_rejection:
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": scaffold_rejection,
                    "raw_response": llm_raw_response,
                })
                print(f"  [llm] Rejected — {scaffold_rejection}")
                continue

            if is_strategy_prompt and _has_answer_clue_leak(text):
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": "answer_clue_leak",
                    "raw_response": llm_raw_response,
                })
                print("  [llm] Rejected — answer clue leak")
                continue

            if is_strategy_prompt and "screen" in text.lower():
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": "screen_reference",
                    "raw_response": llm_raw_response,
                })
                print("  [llm] Rejected — screen reference")
                continue

            strategy_rejection = (
                _strategy_rejection_reason(text, prompt_strategy)
                if is_strategy_prompt
                else None
            )
            if strategy_rejection:
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": strategy_rejection,
                    "raw_response": llm_raw_response,
                })
                print(f"  [llm] Rejected — {strategy_rejection}")
                continue

            if _is_too_similar(text):
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": "too_similar_to_previous",
                    "raw_response": llm_raw_response,
                })
                continue

            if text.strip().upper() == "NONE":
                print(f"  [llm] LLM said NONE — nothing needs attention ({latency:.3f}s)")
                return {"response_text": "NONE", "llm_used": True,
                        "llm_latency_sec": latency,
                        "llm_prompt_sent": llm_prompt_sent,
                        "llm_raw_response": llm_raw_response,
                        "llm_retry_log": retry_log,
                        "llm_backend": LLM_BACKEND}

            words = text.split()
            if len(words) < MIN_WORDS:
                retry_log.append({
                    "attempt": retry + 1,
                    "temperature": temp,
                    "rejection_reason": f"too_few_words ({len(words)})",
                    "raw_response": llm_raw_response,
                })
                print(f"  [llm] Too few words ({len(words)}), retrying...")
                continue
            if len(words) > effective_max_words:
                import re as _re
                sentences = _re.split(r'(?<=[.!?])\s+', text)
                kept = ""
                for sent in sentences:
                    candidate = (kept + " " + sent).strip() if kept else sent
                    if len(candidate.split()) <= effective_max_words:
                        kept = candidate
                    else:
                        break
                if kept:
                    text = kept
                    print(f"  [llm] Trimmed to last complete sentence "
                          f"({len(text.split())} words)")
                else:
                    retry_log.append({
                        "attempt": retry + 1,
                        "temperature": temp,
                        "rejection_reason": (
                            f"too_many_words ({len(words)} > {effective_max_words})"
                        ),
                        "raw_response": llm_raw_response,
                    })
                    print(f"  [llm] Too many words ({len(words)}), retrying...")
                    continue

            _add_to_history(text)

            retry_log.append({
                "attempt": retry + 1,
                "temperature": temp,
                "rejection_reason": None,
                "raw_response": llm_raw_response,
            })

            print(f"  [llm] Generated ({latency:.3f}s, mode={active_mode}): "
                  f"'{text}'")
            return {"response_text": text, "llm_used": True,
                    "llm_latency_sec": latency,
                    "llm_prompt_sent": llm_prompt_sent,
                    "llm_raw_response": llm_raw_response,
                    "llm_retry_log": retry_log,
                    "llm_backend": LLM_BACKEND}

        latency = round(time.time() - t0, 3)
        print(f"  [llm] All {MAX_RETRIES} attempts failed, using fallback")
        return {"response_text": fallback, "llm_used": False,
                "llm_latency_sec": latency,
                "llm_prompt_sent": llm_prompt_sent,
                "llm_raw_response": llm_raw_response,
                "llm_retry_log": retry_log,
                "llm_backend": LLM_BACKEND}

    except requests.exceptions.ConnectionError:
        print("  [llm] Ollama not running, using fallback")
        return {"response_text": fallback, "llm_used": False,
                "llm_latency_sec": 0.0,
                "llm_prompt_sent": llm_prompt_sent,
                "llm_raw_response": llm_raw_response,
                "llm_retry_log": retry_log,
                "llm_backend": LLM_BACKEND}

    except requests.exceptions.Timeout:
        print(f"  [llm] Timeout ({OLLAMA_TIMEOUT}s), using fallback")
        return {"response_text": fallback, "llm_used": False,
                "llm_latency_sec": OLLAMA_TIMEOUT,
                "llm_prompt_sent": llm_prompt_sent,
                "llm_raw_response": llm_raw_response,
                "llm_retry_log": retry_log,
                "llm_backend": LLM_BACKEND}

    except Exception as e:
        print(f"  [llm] Error: {e}, using fallback")
        return {"response_text": fallback, "llm_used": False,
                "llm_latency_sec": 0.0,
                "llm_prompt_sent": llm_prompt_sent,
                "llm_raw_response": llm_raw_response,
                "llm_retry_log": retry_log,
                "llm_backend": LLM_BACKEND}


if __name__ == "__main__":
    print("=" * 60)
    print("  llm_responder.py — self-test (all 9 triggers, Mode A + B)")
    print("=" * 60)

    sample_obs = {
        "face_detected": True,
        "gaze_on_robot": 0.2,
        "head_yaw_deg": -15.3,
        "head_pitch_deg": -30.0,
        "head_moving": False,
        "body_detected": True,
        "no_movement_sec": 12.5,
        "speech_energy": "low",
        "hand_raised": False,
        "already_prompted": False,
        "hand_raise_side": "none",
    }

    triggers = [
        ("no_movement", None),
        ("no_answer", None),
        ("head_turned", None),
        ("face_absent", None),
        ("hand_raised_opening", None),
        ("hand_raised_followup", None),
        ("hand_raised_closing_asked", "What does multiply mean?"),
        ("hand_raised_closing_no", None),
        ("hand_raised_closing_silence", None),
    ]

    for test_mode_label, test_llm_mode in [("Mode A", "a"), ("Mode B", "b")]:
        print(f"\n{'-' * 60}")
        print(f"  Testing {test_mode_label}")
        print(f"{'-' * 60}")

        import llm_responder as _self
        _orig_mode = _self.LLM_MODE
        _self.LLM_MODE = test_llm_mode
        _self._history.clear()

        passed = 0
        for trigger, extra in triggers:
            print(f"\n[{trigger}]")
            result = generate(sample_obs, trigger, extra_context=extra)
            text = result["response_text"]
            used = result["llm_used"]
            lat = result["llm_latency_sec"]
            engine = "LLM" if used else "FALLBACK"
            print(f"  Response: '{text}'")
            print(f"  Engine: {engine}, Latency: {lat:.3f}s")
            if text and len(text) > 2:
                passed += 1
                print(f"  PASS")
            else:
                print(f"  FAIL (empty response)")

        if test_llm_mode == "b":
            print(f"\n[auto — Mode B autonomous detection]")
            result = generate(sample_obs, "auto", extra_context=None)
            text = result["response_text"]
            used = result["llm_used"]
            print(f"  Response: '{text}'")
            print(f"  Engine: {'LLM' if used else 'FALLBACK'}")
            print(f"  (NONE means LLM decided nothing needs attention — valid)")
            if text and len(text) > 2:
                passed += 1
            total_triggers = len(triggers) + 1
        else:
            total_triggers = len(triggers)

        print(f"\n  {test_mode_label} Result: {passed}/{total_triggers} PASS")
        _self.LLM_MODE = _orig_mode

    print("=" * 60)
    print("  SELF-TEST COMPLETE")
    print("=" * 60)
