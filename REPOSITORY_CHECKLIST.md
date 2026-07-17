# Repository Verification Checklist

## Verified from the supplied source

- The paper title, authors, three conditions, eleven strategy names, nine target answer-related strategies, four-run protocol, and 108 pre-sampling target responses match the manuscript.
- The canonical configuration JSON files parse successfully.
- Duplicate files ending in `(2)` are byte-for-byte identical to their canonical counterparts.
- `question_bank_programming.py` self-test passes: 60 questions, 15 per concept.
- `interaction_state.py` self-test passes: 5/5.
- `pedagogical_policy.py` self-test passes: 14/14.
- `logger.py` self-test passes: 4/4.
- `prompt_assembler.py` loads and emits prompts from `config/prompt_assembler.json`.
- C1 and C2 each contain nine target events per run and write to the documented output directories.
- C3 selects all nine answer strategies and all three behavior strategies correctly in a smoke test using a temporary hand-raise stub.

## Not fully verified

- Real Ollama generation was not tested because no Ollama server/model was available in the verification environment.
- Camera processing was not executed with a physical camera.
- The complete C1/C2/C3 scripts cannot run from the supplied snapshot because `hand_raise_manager.py` is missing.
- The live spoken tutor cannot run because `learner_state.py`, `speech_input.py`, `speech_output.py`, and the top-level controller are missing.
- The paper’s stratified sampling, human evaluation, GPT-4o evaluation, agreement calculation, and final analysis scripts were not supplied.

## Blocking fixes

1. Restore `hand_raise_manager.py` source.
2. Fix the dictionary/string interface mismatch in `engagement_manager.py`.
3. Reconcile validation differences across conditions.
4. Reconcile C3 nested retries with the paper’s five-attempt description.
5. Add `hint_request` to the C3 hand-raise action metadata.
6. Add missing evaluation/reproduction scripts.
7. Add a license and final citation metadata.
