# Hinglish Voice Assistant (LiveKit)

Real-time voice agent optimized for **Hindi**, **English**, and **codeswitched Hinglish**. Audio goes over **LiveKit** (WebRTC); the worker runs **`agent.py`** using **livekit-agents** with **Deepgram** streaming STT, optional **Groq** or **OpenAI-compatible** LLM, and optional **Cartesia** TTS.

## What this repo does

| Layer | Technology | Notes |
|--------|------------|--------|
| Transport | LiveKit | Browser or any LiveKit client publishes mic audio; worker joins the same room. |
| VAD | Silero (`livekit-plugins-silero`) | End-of-speech and interruption hints; tuned via `VAD_*` env vars. |
| STT | Deepgram (`livekit-plugins-deepgram`) | Default **Nova-3**, `DEEPGRAM_LANGUAGE=multi` for codeswitching. `detect_language` is forced **off** in code for stable streaming. |
| Language policy | `MirroringLanguageAgent` in `agent.py` | Post-processes transcripts: drops or scrubs non–hi/en noise common on quiet audio, normalizes Hinglish tokens, remaps some mis-hears. Only **hi/en** are intended to reach the LLM. |
| LLM | Groq (default) or OpenAI API | `LLM_PROVIDER`, keys, and `LLM_MODEL` in `.env`. If credentials are missing, pipeline can fall back to **`stt_only`**. |
| TTS | Cartesia (optional) | Used when `VOICE_PIPELINE=stt_llm_tts` and `CARTESIA_VOICE_ID` + `CARTESIA_API_KEY` are set. |
| Browser demo | `index.html` | Fetches a short-lived JWT from **`token_server.py`**, connects to LiveKit, shows transcripts and mic level. Display-only strips mirror some server-side cleanup. |

For deeper notes on multilingual STT tradeoffs and optional architectures, see [`docs/LANGUAGE_DETECTION_NOTES.md`](docs/LANGUAGE_DETECTION_NOTES.md).

## Repository layout

| Path | Role |
|------|------|
| `agent.py` | LiveKit worker entrypoint: STT/LLM/TTS wiring, `entrypoint`, turn handling, stuck-pipeline recovery, transcript filtering. |
| `token_server.py` | FastAPI app: `/health`, `/token` — creates room if needed, **cleans stale participants and dispatches**, returns `{ url, token, room, identity }`. |
| `index.html` | Static test UI: mic selection, level meter, LiveKit client, transcript merge helpers. |
| `requirements.txt` | Python dependencies (pinned major line for LiveKit 1.x). |
| `.env.example` | Documented environment template (copy to `.env`). |

## Prerequisites

- Python **3.10+** (3.11 recommended).
- Accounts and API keys: **LiveKit Cloud** (or self-hosted), **Deepgram**, and for full replies **Groq** or **OpenAI**. For spoken replies: **Cartesia**.
- **FFmpeg** on `PATH` if your LiveKit / agents setup expects it (common for media pipelines).

## Setup

```powershell
cd "c:\path\to\New Ai Assistant"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set at least:

- **LiveKit:** `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- **Deepgram:** `DEEPGRAM_API_KEY`
- **LLM:** `GROQ_API_KEY` (if `LLM_PROVIDER=groq`) or `OPENAI_API_KEY` (if `LLM_PROVIDER=openai`)
- **Full voice (optional):** `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID`, and `VOICE_PIPELINE=stt_llm_tts`

In the [LiveKit Cloud](https://cloud.livekit.io/) project, register an agent whose name matches **`AGENT_NAME`** (default `hinglish-voice-agent`). The token server creates an **agent dispatch** for that name when the browser requests a token.

## Running locally

You need **two** processes: the **worker** and the **token server**. The browser loads `index.html` over HTTP (any static server).

### 1) Agent worker

```powershell
python agent.py dev
```

Wait until logs show the worker registered. If the browser connected while the worker was down, LiveKit may queue dispatches; the worker rejects jobs for a short window after start (`STALE_DISPATCH_DROP_WINDOW_S`) so you do not get duplicate agents—**refresh and connect again** if your first connect was rejected.

### 2) Token server

```powershell
uvicorn token_server:app --host 127.0.0.1 --port 8787 --reload
```

Uses `TOKEN_SERVER_HOST` / `TOKEN_SERVER_PORT` when run as `python token_server.py` instead.

### 3) Static site + open UI

```powershell
python -m http.server 5500
```

Open `http://127.0.0.1:5500/index.html`.

Set the token server URL in the page header:

```html
<meta name="token-server-base" content="http://127.0.0.1:8787">
```

Production: serve `index.html` from your real origin, set **`TOKEN_SERVER_CORS_ORIGINS`** to that exact origin (comma-separated), and point `token-server-base` to your deployed token API. Never embed provider API keys in the frontend.

## Voice pipelines (`VOICE_PIPELINE`)

| Value | Behavior |
|-------|----------|
| `stt_llm` | Default: transcribe → LLM text reply (no TTS). |
| `stt_llm_tts` | Transcribe → LLM → Cartesia speech. Requires Cartesia env vars and plugin import OK. |
| `stt_only` | STT only (no LLM). Used automatically when `STT_TEST_MODE=true` (forces this mode). |

If LLM credentials are missing, `_build_agent` falls back to **`stt_only`** even if you asked for `stt_llm`.

## Important environment groups

- **`DEPLOYMENT`**: `development` vs `production` / `prod`. Production enforces stricter defaults (for example `TOKEN_SERVER_CORS_ORIGINS` required on token server; `STT_TEST_MODE` cleared unless `STT_ALLOW_TEST_IN_PRODUCTION=true`).
- **`STT_QUALITY_MODE`**: `latency` | `balanced` | `accuracy` — one knob that adjusts Deepgram endpointing, VAD silence, turn endpointing, and STT recovery quiet windows unless overridden explicitly.
- **`TURN_DETECTION`**: default `stt` (phrase boundaries + endpointing). Alternatives: `vad`, `manual`, `realtime_llm` (see LiveKit agents docs).
- **`DEEPGRAM_KEYTERM`**: comma-separated terms; **only applied when the STT language is English-prefixed**, otherwise dropped in code to avoid breaking non-English models.
- **Barge-in / echo**: `AEC_WARMUP_DURATION_S`, `INTERRUPTION_*` tune how quickly user speech can interrupt TTS.

Full variable list and defaults live in **`.env.example`**.

## Logging and latency

- Console and file: **`LOG_LEVEL`**, **`LOG_FILE`** (default `voice-agent.log`).
- Search logs for **`STT_TURN_TIMING`** per turn when `LOG_STT_TURN_TIMING=true`.
- **`PIPE_STEP`** lines trace job lifecycle timing from `entrypoint`.

## Troubleshooting

| Symptom | Things to check |
|---------|-------------------|
| No agent in room / wrong agent | `AGENT_NAME` matches LiveKit agent registration; worker is running; after a reject, refresh browser for a new token + dispatch. |
| `REJECT_STALE_DISPATCH` right after start | Expected if old dispatches fired in the first ~1s of worker uptime; reconnect. Adjust `STALE_DISPATCH_DROP_WINDOW_S` only if you understand duplicate-job tradeoffs. |
| Token server 500 in production | Set `TOKEN_SERVER_CORS_ORIGINS` to real browser origins (no `*` with credentials). |
| Empty or garbage STT | Mic level in UI (green/yellow); Windows “Communications” device lowers gain—use a normal mic device; see help panel in `index.html`. |
| `stt_llm` but no LLM replies | Missing `GROQ_API_KEY` / `OPENAI_API_KEY`; check logs for pipeline fallback to `stt_only`. |
| No speech from agent | `VOICE_PIPELINE=stt_llm_tts`, `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID`, and Cartesia package installed; check `TTS_INIT_FAILED` in logs. Look for `TTS_LANG_MIRROR` (per-turn `en`/`hi`). Log `no audio frames were pushed` often means Cartesia returned no audio — check [Cartesia credits/subscription](https://play.cartesia.ai/subscription) (HTTP 402) and that the voice supports the reply language. |
| Duplicate user messages / stuck turn | Code includes **stuck-pipeline recovery** and STT idle guards (`STUCK_PIPELINE_*`, `STT_RECOVERY_*`); tune only after reading log patterns. |

## Security

- Keep **all API secrets** in `.env` on the server / worker machine only.
- The browser should only receive **short-lived LiveKit tokens** from `token_server` (optional `TOKEN_SERVER_API_KEY` / `X-API-Key`).
- Do not commit `.env`.

## License

Add a `LICENSE` file if you open-source this project; none is set in this template.
