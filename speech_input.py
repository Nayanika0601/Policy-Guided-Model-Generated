
import time
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


SAMPLE_RATE        = 16000
CHANNELS           = 1
BLOCK_DURATION_SEC = 0.3
BLOCK_SAMPLES      = int(SAMPLE_RATE * BLOCK_DURATION_SEC)

RMS_THRESHOLD      = 0.015
SILENCE_AFTER_SEC  = 1.2

_WHISPER_MODEL_SIZE = "small.en"

print(f"Loading faster-whisper '{_WHISPER_MODEL_SIZE}' model ...")
_model = WhisperModel(_WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print("faster-whisper model loaded and ready.")


def listen_for_answer(timeout_sec: float = 15.0,
                      stop_event: threading.Event | None = None) -> dict:
    speech_started  = False
    silence_start: float | None = None
    interrupted     = False
    audio_chunks: list[np.ndarray] = []

    mic_start = time.time()
    deadline  = mic_start + timeout_sec

    print(f"  [mic] Listening (timeout {timeout_sec}s) ...")

    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                           dtype="int16", blocksize=BLOCK_SAMPLES) as stream:
        while time.time() < deadline:
            if stop_event is not None and stop_event.is_set():
                print("  [mic] Interrupted by event.")
                interrupted = True
                break

            raw, _overflowed = stream.read(BLOCK_SAMPLES)
            raw_bytes = bytes(raw)

            audio_f32 = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            energy = float(np.sqrt(np.mean(audio_f32 ** 2)))

            if speech_started:
                audio_chunks.append(audio_f32)

            if energy >= RMS_THRESHOLD:
                if not speech_started:
                    print("  [mic] Speech detected — recording ...")
                    speech_started = True
                    audio_chunks.append(audio_f32)
                silence_start = None
            elif speech_started:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_AFTER_SEC:
                    print("  [mic] Silence detected — done recording.")
                    break

    mic_listen_sec = round(time.time() - mic_start, 3)

    if not speech_started:
        print("  [mic] Nothing heard.")
        return {
            "transcript":             "",
            "mic_listen_sec":         mic_listen_sec,
            "whisper_transcribe_sec": 0.0,
            "interrupted":            interrupted,
        }

    audio = np.concatenate(audio_chunks).astype(np.float32)

    t0 = time.time()
    segments, _info = _model.transcribe(audio, language="en", beam_size=1)
    transcript = "".join(seg.text for seg in segments).strip()
    whisper_sec = round(time.time() - t0, 3)

    print(f"  [mic] Transcript: '{transcript}'")
    print(f"  [mic] Latency — mic: {mic_listen_sec:.3f}s, whisper: {whisper_sec:.3f}s")

    return {
        "transcript":             transcript,
        "mic_listen_sec":         mic_listen_sec,
        "whisper_transcribe_sec": whisper_sec,
        "interrupted":            interrupted,
    }


if __name__ == "__main__":
    print("=" * 55)
    print("  speech_input.py — self-test (faster-whisper)")
    print("=" * 55)

    tests_passed = 0
    total_tests  = 3

    print("\n[Test 1] faster-whisper model loaded?")
    try:
        assert _model is not None
        print("  PASS")
        tests_passed += 1
    except AssertionError:
        print("  FAIL — model is None")

    print("\n[Test 2] Mic stream opens?")
    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                               dtype="int16", blocksize=BLOCK_SAMPLES) as s:
            data, _ = s.read(BLOCK_SAMPLES)
            assert len(bytes(data)) == BLOCK_SAMPLES * 2
        print("  PASS")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")

    print("\n[Test 3] Live listen — say a number or wait 5s for timeout ...")
    try:
        result = listen_for_answer(timeout_sec=5)
        ok = isinstance(result, dict) and "transcript" in result
        if ok:
            t  = result["transcript"]
            ml = result["mic_listen_sec"]
            vt = result["whisper_transcribe_sec"]
            if t:
                print(f"  Heard: '{t}' (mic={ml:.3f}s, whisper={vt:.3f}s) — PASS")
            else:
                print(f"  Nothing heard (mic={ml:.3f}s, timeout works) — PASS")
            tests_passed += 1
        else:
            print(f"  FAIL — unexpected return: {result}")
    except Exception as e:
        print(f"  FAIL — {e}")

    print("-" * 55)
    print(f"  Result: {tests_passed}/{total_tests} PASS")
    if tests_passed == total_tests:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 55)
