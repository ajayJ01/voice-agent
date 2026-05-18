# AI Voice Agent — Language Detection Issue Fix Documentation

This note captures strategies for **Hindi + English + Hinglish** STT stability.  
**This repo’s default runtime** remains: LiveKit + Deepgram (see `agent.py`) + post-filters.  
Alternative stacks (Faster-Whisper local, Flux `language_hint`, etc.) are **optional** migration paths — do not flip them without testing latency and turn-taking.

---

## Problem Statement

Language detection can feel inconsistent when:

- Hindi is mis-tagged (e.g. as `es`)
- Hinglish is unstable
- English fragments drop or garble
- Multilingual mode adds junk on quiet audio
- Mixed conversations feel slower or less accurate

**Target:** Hindi + English + Hinglish, real-time behaviour, cost-aware.

---

## Root Cause (typical)

Many cloud STT setups use **multilingual / auto-detection**. Hinglish is not a single ISO locale, so models may spread probability across `hi`, `en`, `ur`, etc. **Weak mic / noise** makes this worse.

---

## Principles

### Avoid (for many cloud streaming APIs)

- `detect_language=True` when your integration does not support it cleanly (this project keeps Deepgram `detect_language=False`).
- Relying on “unconstrained multi” without **server-side** hi/en policy and junk stripping.

### Prefer

- **Stable capture:** mic level, VAD, optional push-to-talk / wake word.
- **Explicit product policy:** only pass to the LLM what you consider valid hi/en text (this repo does remapping + drops + glue stripping in `agent.py`).
- **Architecture fit:** local Faster-Whisper vs cloud streaming are **different products** — not a drop-in config swap for LiveKit+Deepgram.

---

## Alternative architecture (local / laptop assistant)

| Component | Example |
|-----------|---------|
| Audio | `sounddevice` |
| VAD | Silero VAD |
| STT | Faster-Whisper |
| Brain | GPT / Groq / local |
| TTS | Piper / cloud |
| Commands | Intent router (optional) |

### Faster-Whisper (illustrative)

Install (see upstream repo for current instructions): `faster-whisper`, `torch`, FFmpeg on PATH, etc.

**Wrong pattern (for strict control):** relying on implicit auto language in all cases.

**Biased pattern (often used for Hinglish-heavy audio):** lock Whisper to Hindi and use VAD:

```python
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cpu", compute_type="int8")

segments, info = model.transcribe(
    "audio.wav",
    language="hi",
    beam_size=1,
    vad_filter=True,
)

for segment in segments:
    print(segment.text)
```

Caveat: `language="hi"` can **hurt pure English** sentences; tune per product.

### Model size tradeoff

| Model | Speed | Accuracy |
|-------|-------|----------|
| tiny | Fastest | Lowest |
| small | Good | Good |
| medium | Medium | Strong |
| large | Slower | Strongest |

---

## Deepgram (this repo’s default cloud path)

- **Nova-3 + `language=multi`:** good for codeswitching; may still emit non–hi/en tags or junk on noise — mitigated in `agent.py` (strip, remap, drop policy).
- **Restricting `multi` to exactly `hi`+`en`:** not a single documented query flag for Nova-3 the way **Flux** uses `language_hint`; see Deepgram’s Flux / v2 docs if you migrate.
- **`detect_language`:** keep off for this LiveKit streaming integration unless Deepgram + plugin fully support your flow.

---

## Product patterns that help (any stack)

1. **Wake word or push-to-talk** — fewer false transcripts.
2. **Intent router** for fixed commands (“Chrome kholo”) — don’t send everything raw to a general LLM if you don’t need to.
3. **Noise:** reasonable mic gain; optional NS tradeoffs (this project’s client often runs NS/AGC off for STT clarity — test your room).

---

## Checklist

- [ ] Language / model choice documented in `.env` (`DEEPGRAM_LANGUAGE`, `DEEPGRAM_MODEL`).
- [ ] `detect_language` aligned with your provider + plugin.
- [ ] VAD + endpointing tuned for your room.
- [ ] Server-side hi/en policy matches product (drops, remaps, strips).
- [ ] If switching STT engine — re-measure latency and turn completion end-to-end.

---

## References (external)

- [Faster Whisper (SYSTRAN)](https://github.com/SYSTRAN/faster-whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [FFmpeg](https://ffmpeg.org/download.html)
- [Deepgram docs](https://developers.deepgram.com/)

---

## Note on **this** repository

Current worker: **`agent.py`** + **Deepgram** streaming via `livekit.plugins.deepgram`.  
Changing to Faster-Whisper or Flux requires **new STT integration** and regression testing — not a one-line `.env` change.
