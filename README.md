# Policy-Guided, Model-Generated Robot Tutoring

This repository contains the research implementation associated with **“Policy-Guided, Model-Generated: A Strategy-Selected Robot Tutoring Pipeline.”** The project compares three LLM-based tutoring conditions for a beginner programming-concepts quiz. Its central design separates pedagogical decision-making from language generation: in the strategy-selected condition, a deterministic Python policy chooses the teaching move, and the LLM is limited to verbalizing that move under response and safety constraints.

> [!IMPORTANT]
> The reviewed code snapshot is **not yet runnable from a fresh clone**. Before publishing, add the source file `hand_raise_manager.py`. The full live camera-and-speech tutor also requires `learner_state.py`, `speech_input.py`, `speech_output.py`, and a top-level application runner. Do not rely on or publish the supplied `.pyc` files; they are Python-version-specific generated artifacts.

## Experimental conditions

| Condition | Name | Information available to the LLM | Teaching-move authority |
|---|---|---|---|
| **C1** | Behavior-only | Observable behavioral signals | LLM |
| **C2** | Contextual-state | Behavior, interaction state, and learner state | LLM |
| **C3** | Strategy-selected | A policy-selected strategy and constrained context | Deterministic Python policy |

The policy defines eleven strategies:

`praise`, `confidence_boost`, `challenge`, `slow_down`, `scaffold_hint`, `metacognitive_prompt`, `reassure`, `give_example`, `elaborate_concept`, `re_engage`, and `stillness_check`.

The paper’s main response evaluation uses the nine answer-related strategies. `re_engage` and `stillness_check` are behavior-only engagement strategies and are logged separately.

## Repository hierarchy

Arrange the repository as follows. The experiment scripts must remain exactly one directory below the project root because they calculate the root with `Path(__file__).resolve().parents[1]`.

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE                         # Add before public release
├── CITATION.cff                    # Add/update when publication details are final
├── config/
│   ├── settings.json               # Local file; copy from settings.example.json
│   ├── settings.example.json
│   ├── prompts_mode_a.json
│   ├── prompts_mode_b.json
│   └── prompt_assembler.json
├── models/
│   ├── face_landmarker.task
│   └── pose_landmarker_lite.task
├── experiments/
│   ├── run_c1_behavior_only_scenarios.py
│   ├── run_c2_contextual_state_scenarios.py
│   └── run_c3_strategy_pipeline_scenarios.py
├── outputs/                        # Created automatically; normally not committed
│   ├── C1_behavior_only/
│   ├── C2_contextual_state/
│   └── C3_strategy_pipeline/
├── paper/
│   └── CSCE2026.pdf                # Optional manuscript copy
├── engagement_manager.py
├── hand_raise_manager.py           # REQUIRED; source file is currently missing
├── interaction_state.py
├── learner_state.py                # REQUIRED for the live quiz; currently missing
├── llm_responder.py
├── logger.py
├── mediapipe_processor.py
├── pedagogical_policy.py
├── prompt_assembler.py
├── question_bank_programming.py
├── quiz_manager.py
├── speech_input.py                 # REQUIRED for the live quiz; currently missing
├── speech_output.py                # REQUIRED for the live quiz; currently missing
└── main.py                         # Recommended top-level live-system entry point
```

Do not keep duplicate files ending in `(2)`, root-level model/config copies, `fontlist-v390.json`, `.pyc` files, or `__pycache__/` directories.

## Requirements

### Software

- **Python 3.12 is recommended** because the original compiled artifacts were produced with CPython 3.12. The reviewed source syntax requires Python 3.10 or newer, but the public repository should contain source files rather than `.pyc` artifacts.
- **Ollama** running locally or at a reachable HTTP endpoint.
- The Ollama model **`llama3.1:8b`**, matching the model named in the paper and configuration.
- Git and a terminal.

### Python packages

The currently visible source requires:

- `requests` for Ollama HTTP requests;
- `opencv-python` for camera capture, overlays, Haar cascades, and the macOS fallback backend;
- `mediapipe` for Face Landmarker and Pose Landmarker on the MediaPipe Tasks backend.

The full speech pipeline may require additional microphone, speech-to-text, and text-to-speech packages. Those dependencies cannot be specified accurately until `speech_input.py` and `speech_output.py` are added.

### Hardware

- The scripted C1/C2/C3 scenario generators do **not** require a camera or microphone.
- A webcam is required for `mediapipe_processor.py`.
- A microphone and speakers are required for the live spoken tutor.
- A GPU is recommended for responsive local generation. The paper reports an NVIDIA RTX 3070, but Ollama can also be configured on another reachable machine.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<organization-or-user>/<repository-name>.git
cd <repository-name>
```

### 2. Create and activate a virtual environment

macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The supplied `requirements.txt` is a minimal compatibility file, not an exact lock file. For archival reproducibility, record the original working environment with:

```bash
python -m pip freeze > requirements-lock.txt
```

### 4. Install Ollama and pull the model

Install Ollama for your operating system, then run:

```bash
ollama pull llama3.1:8b
ollama serve
```

In another terminal, confirm that the model is available:

```bash
ollama list
```

The code uses Ollama’s non-streaming generate endpoint at `http://localhost:11434/api/generate` by default.

### 5. Create the local configuration

macOS or Linux:

```bash
cp config/settings.example.json config/settings.json
```

Windows PowerShell:

```powershell
Copy-Item config/settings.example.json config/settings.json
```

The C3 runner imports `llm_responder.py`, which expects `config/settings.json`, `config/prompts_mode_a.json`, and `config/prompts_mode_b.json` to exist at import time.

## Configuration

Edit `config/settings.json`:

```json
{
  "ollama_url": "http://localhost:11434/api/generate",
  "ollama_model": "llama3.1:8b",
  "llm_timeout_sec": 120,
  "llm_mode": "a",
  "llm_history_length": 3,
  "llm_min_words": 8,
  "llm_max_words": 35,
  "thresholds": {
    "no_movement_sec": 30,
    "no_answer_sec": 15,
    "face_absent_sec": 3,
    "gaze_away_thresh": 0.3
  },
  "cooldown_sec": 15,
  "llm_backend": "ollama",
  "condition": "C3"
}
```

| Setting | Purpose |
|---|---|
| `ollama_url` | Full Ollama `/api/generate` endpoint. Use a reachable URL or tunnel when the model runs remotely. |
| `ollama_model` | Ollama model tag; the paper uses `llama3.1:8b`. |
| `llm_timeout_sec` | HTTP timeout for one request. |
| `llm_mode` | `a` uses trigger-specific prompts. `b` lets the LLM inspect observation context and return `NONE` when no intervention is needed. |
| `llm_history_length` | Number of recent tutor responses used by `llm_responder.py` to reduce repetition. |
| `llm_min_words`, `llm_max_words` | General response-length limits. Strategy prompts are additionally limited to 18 words in code. |
| `thresholds` | Engagement-manager thresholds for no answer, absent face, gaze away, and no movement. |
| `cooldown_sec` | Minimum interval before repeating an engagement trigger. |
| `llm_backend` | Logged as metadata. The reviewed code routes requests through `ollama_url`; it does not implement a separate `gpu_ssh` transport. |
| `condition` | Label used by the live session logger. The experiment scripts define their own condition labels. |

### Vision configuration

`mediapipe_processor.py` defaults to:

- `opencv` on macOS;
- `mediapipe_tasks` on other platforms.

Override the backend before starting Python:

macOS or Linux:

```bash
export PIPELINE4_VISION_BACKEND=mediapipe_tasks
```

Windows PowerShell:

```powershell
$env:PIPELINE4_VISION_BACKEND="mediapipe_tasks"
```

Use `opencv` or `mediapipe_tasks`. In the current implementation, any other value falls through to the OpenCV branch rather than raising a configuration error.

The task models belong in `models/`. If they are absent, the code attempts to download them automatically from Google-hosted MediaPipe model URLs. For exact paper reproducibility, either version the reviewed model files or publish their hashes and fixed download locations instead of relying only on a `latest` URL.

Reviewed model hashes:

```text
face_landmarker.task
SHA-256: 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

pose_landmarker_lite.task
SHA-256: 59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a
```

`CAMERA_INDEX`, `OBSERVATION_INTERVAL_SEC`, `SHOW_WINDOW`, and the vision module’s 10-second stillness threshold are currently hard-coded near the top of `mediapipe_processor.py`; they are not read from `settings.json`.

## Generate the three-condition response data

Run commands from the repository root **after adding `hand_raise_manager.py`**.

### C1: Behavior-only

```bash
python experiments/run_c1_behavior_only_scenarios.py --runs 4 --seed 42
```

### C2: Contextual-state

```bash
python experiments/run_c2_contextual_state_scenarios.py --runs 4 --seed 42
```

### C3: Strategy-selected

```bash
python experiments/run_c3_strategy_pipeline_scenarios.py \
  --runs 4 \
  --seed 42 \
  --force-llm-for-content
```

> [!WARNING]
> `--force-llm-for-content` is required to generate LLM responses for C3. Without it, the current script records the placeholder text `Simulated C3 tutor response.` for strategy-conditioned content.

Optional C3 flag:

```bash
--no-hand-raise-llm
```

This makes the hand-raise response branch deterministic. It does not remove the import-time requirement for `hand_raise_manager.py`.

Each run contains nine target answer-related scenarios. Four runs across all three conditions produce 108 target responses before evaluation sampling.

### Scope of reproducibility

The uploaded scenario runners generate the three-condition JSON response sets. They do **not** include the full paper-evaluation workflow for:

- constructing the stratified 48-response sample;
- anonymizing and randomizing condition labels;
- collecting the three human-rater scores;
- running the GPT-4o secondary evaluation;
- calculating inter-rater agreement; or
- reproducing the paper’s final tables.

Add the sampling, rubric, scoring, and analysis scripts—or provide the de-identified evaluation dataset—before claiming one-command reproduction of the paper’s reported results.

## Run individual components

### Question-bank self-test

```bash
python question_bank_programming.py
```

Expected result: 60 questions, with 15 questions each for variables, data types, loops, and conditionals.

### Interaction-state self-test

```bash
python interaction_state.py
```

### Pedagogical-policy self-test

```bash
python pedagogical_policy.py
```

### Prompt-assembly inspection

```bash
python prompt_assembler.py
```

### LLM response-layer self-test

```bash
python llm_responder.py
```

This script can report passing fallback tests even when Ollama is unavailable. Confirm `llm_used: true` in generated experiment events when verifying real model execution.

### Camera observation

```bash
python mediapipe_processor.py
```

Press `Q` or `Esc` to stop. Observations are saved as `observations_<timestamp>.json` in the current working directory.

### Live spoken quiz

`quiz_manager.py`, `engagement_manager.py`, and `logger.py` are components, not a complete public entry point. Add the missing source modules and a top-level runner before documenting a live-system command.

## Test the installation

Validate JSON configuration:

```bash
python -m json.tool config/settings.json
python -m json.tool config/prompts_mode_a.json
python -m json.tool config/prompts_mode_b.json
python -m json.tool config/prompt_assembler.json
```

Compile the source tree:

```bash
python -m compileall -q .
```

Run source self-tests:

```bash
python question_bank_programming.py
python interaction_state.py
python pedagogical_policy.py
python prompt_assembler.py
python logger.py
```

Confirm Ollama directly:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Reply with the word ready.",
  "stream": false
}'
```

After `hand_raise_manager.py` is present, run a one-session smoke test:

```bash
python experiments/run_c1_behavior_only_scenarios.py --runs 1 --seed 42
python experiments/run_c2_contextual_state_scenarios.py --runs 1 --seed 42
python experiments/run_c3_strategy_pipeline_scenarios.py --runs 1 --seed 42 --force-llm-for-content
```

Check that the events intended to use the model contain:

```json
{
  "llm_used": true,
  "response_source": "llm"
}
```

A completed script can still contain deterministic fallback events, so a zero exit code alone does not prove that Ollama generated every response.

## Output JSON files

```text
outputs/C1_behavior_only/C1_behavior_only_pipeline_<timestamp>.json
outputs/C2_contextual_state/C2_contextual_state_<timestamp>.json
outputs/C3_strategy_pipeline/C3_strategy_pipeline_<timestamp>.json
```

Common content includes:

- condition and run metadata;
- target and setup events;
- observation, interaction, and learner-state snapshots;
- prompts and generated responses;
- LLM latency and retry logs;
- hand-raise and engagement events; and
- aggregate summary fields.

C1 and C2 use the top-level key `target_events`. C3 uses `target_strategy_events`.

The setup events establish the learner history required to activate strategies such as reassurance, example generation, and concept elaboration. Keep them separate from the target responses used in the main evaluation.

Do not commit identifiable participant data. Store only de-identified reproducibility artifacts in a tracked directory such as `results/paper/`; keep raw session output under the ignored `outputs/` directory.

## Validation and retry behavior

The code includes checks for combinations of:

- answer or answer-clue leakage;
- unrelated code-context references;
- raw sensor/internal-state exposure;
- third-person learner references;
- strategy-label leakage;
- response length;
- strategy-specific violations; and
- similarity to recent responses.

However, the reviewed implementation does **not** apply an identical validator in all three conditions. C1 and C2 use condition-local validation functions, whereas C3 uses `llm_responder.py` plus an additional outer code-context check. This should be reconciled with the paper’s description of a shared validation pipeline before archival release.

C1 and C2 each use a five-temperature retry sequence: `0.7`, `0.9`, `1.0`, `1.1`, and `1.2`. C3 calls a responder that already retries up to five times from inside another loop that can also repeat up to five times for code-context hallucinations. Consequently, C3 can potentially issue more than five model generations for one event. Align this behavior with the paper’s stated retry protocol or document the difference explicitly.

## Required corrections before publishing

1. Add the portable source file `hand_raise_manager.py`; the experiment runners cannot import the current root-level CPython 3.12 `.pyc` filename.
2. Add `learner_state.py`, `speech_input.py`, `speech_output.py`, and a top-level runner for the live tutoring path.
3. Fix `engagement_manager.py`: `select_strategy()` returns a decision dictionary, but the manager currently treats it as a strategy string. Use `decision["strategy"]`, `decision["strategy_description"]`, and `decision["reason"]`.
4. Make validation behavior consistent across C1, C2, and C3—or revise the manuscript and documentation to describe condition-specific validators.
5. Remove the nested C3 retry behavior if the intended limit is five model generations per event.
6. Add `hint_request` to `HAND_RAISE_HELP_ACTIONS` in the C3 summary metadata; the scenario sequence tests it but the list omits it.
7. Decide whether the 10-second vision stillness flag and 30-second engagement intervention intentionally represent different stages. Rename or consolidate them to avoid confusion.
8. Move all configuration files into `config/`, model files into `models/`, and experiment scripts into `experiments/`.
9. Remove duplicates ending in `(2)`, `.pyc` files, `__pycache__/`, and `fontlist-v390.json`.
10. Add the response-sampling and evaluation-analysis code needed to reproduce the paper’s reported 48-response evaluation and score tables.
11. Add a license and finalized citation metadata before public release.

## Privacy and research-data handling

The live pipeline can process camera frames, speech transcripts, learner answers, prompts, inferred states, and response timing. De-identify logs before sharing them, and do not publish recordings, identifiable transcripts, or participant-level outputs without the applicable consent and institutional approvals.

## Associated paper

**Nayanika Reddy Rajoli, Abhishek Jariwala, Vennela Akula, Daniela Marghitu, and Richard Chapman.** “Policy-Guided, Model-Generated: A Strategy-Selected Robot Tutoring Pipeline.” Short research paper manuscript, 2026.

Temporary BibTeX entry; update after publication:

```bibtex
@misc{rajoli2026policyguided,
  title  = {Policy-Guided, Model-Generated: A Strategy-Selected Robot Tutoring Pipeline},
  author = {Rajoli, Nayanika Reddy and Jariwala, Abhishek and Akula, Vennela and Marghitu, Daniela and Chapman, Richard},
  year   = {2026},
  note   = {Short research paper manuscript}
}
```

## License

No license was included in the reviewed files. Select a license, add the corresponding `LICENSE` file, and replace this section before describing the repository as open source.
