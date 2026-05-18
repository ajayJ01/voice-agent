import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit.agents import JobContext, JobRequest, StopResponse, WorkerOptions, cli
from livekit.agents import llm as lk_llm
from livekit.agents.llm import ChatMessage
from livekit.agents.voice import Agent, AgentSession, room_io
from livekit.plugins import deepgram, openai, silero

# Existing imports ke baad ye line add kar do
try:
    from livekit.plugins import cartesia
except ImportError:
    cartesia = None

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("voice-agent")

_log_file = os.getenv("LOG_FILE", "voice-agent.log")
_file_handler = logging.FileHandler(Path(_log_file), encoding="utf-8")
_file_handler.setLevel(os.getenv("LOG_LEVEL", "INFO"))
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
)
logger.addHandler(_file_handler)


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning("CONFIG %s=%r invalid using_default=%s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning("CONFIG %s=%r invalid using_default=%s", name, raw, default)
        return default


def _env_optional_float(name: str, default: float | None) -> float | None:
    """Parse float env; empty → default. ``none``/``off`` → None (disable feature)."""
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    s = str(raw).strip().lower()
    if s in {"none", "null", "off", "disable"}:
        return None
    try:
        return float(s)
    except ValueError:
        logger.warning("CONFIG %s=%r invalid using_default=%s", name, raw, default)
        return default


def _env_choice(name: str, allowed: set[str], default: str) -> str:
    raw = (os.getenv(name) or default).strip().lower()
    if raw not in allowed:
        logger.warning("CONFIG %s=%r invalid fallback=%r allowed=%s", name, raw, default, sorted(allowed))
        return default
    return raw


def _stt_quality_preset() -> tuple[str, dict[str, float | int]]:
    """Single knob: latency ↔ accuracy for Hindi+English streaming STT.

    Override any value with explicit DEEPGRAM_ENDPOINTING_MS, STT_RECOVERY_*, TURN_ENDPOINTING_*.
    """
    raw_mode = (os.getenv("STT_QUALITY_MODE") or "balanced").strip().lower()
    presets: dict[str, dict[str, float | int]] = {
        "latency": {
            "dg_endpointing_ms": 560,
            "recover_quiet_s": 0.70,
            "recover_frag_extra_s": 0.38,
            "turn_ep_min_s": 0.20,
            "turn_ep_max_s": 1.52,
            "vad_min_silence_s": 0.36,
        },
        "balanced": {
            "dg_endpointing_ms": 680,
            "recover_quiet_s": 0.88,
            "recover_frag_extra_s": 0.48,
            "turn_ep_min_s": 0.24,
            "turn_ep_max_s": 1.78,
            "vad_min_silence_s": 0.42,
        },
        "accuracy": {
            "dg_endpointing_ms": 840,
            "recover_quiet_s": 1.12,
            "recover_frag_extra_s": 0.62,
            "turn_ep_min_s": 0.30,
            "turn_ep_max_s": 2.28,
            "vad_min_silence_s": 0.52,
        },
    }
    if raw_mode not in presets:
        logger.warning("STT_QUALITY_MODE=%r invalid using_balanced", raw_mode)
        return "balanced", presets["balanced"]
    return raw_mode, presets[raw_mode]


def _devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    dev = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    return dev / max(len(text), 1)


def _hindi_likely_incomplete_phrase(text: str) -> bool:
    """Heuristic: phrase-final landed on a connective (e.g. … का) — more words often follow."""
    t = (text or "").strip().rstrip("?.! …\u0964\u0965")
    if not t or _devanagari_ratio(t) < 0.2:
        return False
    if t.endswith("क्या"):
        return False
    # Postpositions / glue words at end often mean "X का <noun clause>" was cut early.
    if t.endswith(("का", "की", "के")):
        return True
    if len(t) < 12 and "?" not in text and "।" not in text:
        if t.endswith(("में", "से", "को", "पर", "और", "या")):
            return True
    return False


# Languages we explicitly support. Anything else from Deepgram `multi` mode on
# quiet audio is almost always a hallucination ("¿Quién me tomará...", "Ya por",
# "Llamo a tomar el nombre de Amsit"). Reject these turns silently rather than
# letting the LLM try to reply to gibberish.
_SUPPORTED_LANGS = {"en", "hi"}
_HALLUCINATION_LANGS = {"es", "pt", "fr", "de", "it", "ru", "ja", "nl"}
# If Devanagari share is at least this, treat transcript as Hindi for tag recovery / drop policy.
_STT_DEVANAGARI_HI_SIGNAL = 0.06


_SPANISH_MARKERS = set("áéíóúñÁÉÍÓÚÑ¿¡")
# Very common Spanish/Portuguese stopwords that almost never appear in real
# English speech. If we see one of these in a non-English-tagged transcript we
# treat it as hallucinated.
_HALLUCINATION_VOCAB = {
    # Spanish stopwords / pronouns / particles
    "ya", "y", "que", "qué", "ajá", "hasta", "tomó", "tomo", "sombra",
    "programó", "programo", "su", "lo", "la", "el", "del", "los", "las",
    "tomar", "para", "como", "está", "estaba", "esto", "eso", "esta",
    "sí", "pero", "porque", "tambien", "también", "cierto", "ciertos",
    "señor", "señora", "deje", "digamos", "vamos", "aunque", "se",
    "orden", "menu", "dirigamos", "dirigir", "siga", "sigamos",
    # Spanish verb-ending fragments that real English never produces
    "amos", "emos", "imos", "ando", "iendo",
    # Portuguese
    "obrigado", "obrigada", "tudo", "bem", "voce", "você",
    "estamos", "estão", "já", "bara", "toca", "perder", "antio",
    # French
    "merci", "bonjour", "oui", "non",
    # German
    "danke", "ja", "nein", "guten",
}

# Common English-only stopwords. If a transcript flagged as a hallucinated
# language has many of these, it's probably real English mis-tagged.
_ENGLISH_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "should", "could",
    "may", "might", "must", "can", "to", "of", "in", "on", "at", "with",
    "for", "from", "by", "about", "i", "you", "he", "she", "it", "we",
    "they", "them", "us", "this", "that", "these", "those", "what",
    "where", "when", "why", "how", "who", "whom", "if", "then", "and",
    "or", "but", "so", "because", "as", "while", "until", "though",
    "tell", "tomorrow", "today", "yesterday", "please", "name", "your",
}

# For transcripts tagged `en` by Deepgram: EU diacritics or clearly Romance/German
# tokens in a short phrase are almost never real English speech-to-text here.
_EU_ACCENT_CHARS = frozenset(
    "áéíóúñÁÉÍÓÚÑçÇàèìòùÀÈÌÒÙâêîôûÂÊÎÔÛäëïöüÄËÏÖÜœŒæÆ"
)
_EN_TAG_HALLUCINATION_VOCAB = frozenset(
    {
        "qué",
        "sí",
        "pero",
        "porque",
        "está",
        "estaba",
        "también",
        "tambien",
        "señor",
        "señora",
        "obrigado",
        "obrigada",
        "você",
        "voce",
        "merci",
        "bonjour",
        "oui",
        "danke",
        "guten",
        # Common `multi` hallucination glue (“… el nombre …”, “no lo …”)
        "llamo",
        "nombre",
        "perdido",
        "perdida",
        "jugo",
        "aló",
        "hola",
        "ola",
    }
)


def _looks_like_english_latin(text: str) -> bool:
    """Heuristic: Latin text is probably English (mis-tagged as es/pt/…) — allow remap to en."""
    t = (text or "").strip()
    if not t or _devanagari_ratio(t) >= 0.08:
        return False
    words = [w.lower().strip(".,?!¿¡;:'\"") for w in t.split()]
    words = [w for w in words if w]
    if not words:
        return False
    english_hits = sum(1 for w in words if w in _ENGLISH_STOPWORDS)
    if english_hits >= 2:
        return True
    if english_hits >= 1 and len(words) <= 5:
        return True
    return False


def _should_drop_unsupported_stt_language(text: str, stt_lang: str | None) -> bool:
    """STT policy: only Hindi + English. Drop anything else (after remap in note_stt_language)."""
    sl = (stt_lang or "").strip().lower()
    if not sl:
        return False
    base = sl.split("-", 1)[0]
    if base in _SUPPORTED_LANGS:
        return False
    t = (text or "").strip()
    if _devanagari_ratio(t) >= _STT_DEVANAGARI_HI_SIGNAL:
        return False
    if _looks_like_english_latin(t):
        return False
    return True


def _latin_words(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z]+(?:['’][a-zA-Z]+)?", text or "")]


def _assistant_echo_word_overlap(user: str, assistant: str) -> float:
    """Share of user words that appear in `assistant` (TTS echo / room pickup)."""
    uw = _latin_words(user)
    aw = frozenset(_latin_words(assistant))
    if not uw or not aw:
        return 0.0
    meaningful = [w for w in uw if len(w) > 1]
    if not meaningful:
        meaningful = uw
    hits = sum(1 for w in meaningful if w in aw)
    return hits / len(meaningful)


def _is_probably_assistant_echo(user: str, assistant: str) -> bool:
    u = (user or "").strip().lower()
    a = (assistant or "").strip().lower()
    if not u or not a:
        return False
    # Long echoed fragment (e.g. user picked up full TTS clause).
    if len(u) >= 10 and u in a:
        return True
    u_words = _latin_words(user)
    if not u_words:
        return False
    # Single-token lines like "Hello?" / "Hello." match the assistant's "Hello."
    # with 100% overlap but are often a real follow-up — need 2+ words before
    # using overlap ratio.
    meaningful = [w for w in u_words if len(w) > 1]
    if not meaningful:
        meaningful = u_words
    if len(meaningful) < 2:
        return False
    overlap = _assistant_echo_word_overlap(user, assistant)
    if overlap >= 0.55 and len(u_words) <= 10:
        return True
    return False


_ROME_HI_ONE_WORD = frozenset(
    {
        "aló",
        "alo",
        "hola",
        "ola",
        "bueno",
        "gracias",
        "adiós",
        "adios",
        "vale",
    }
)


def _token_is_misheard_spanish_alo_phone(w: str) -> bool:
    """True for STT tokens like Aló / ¿Aló? that Nova-3 often tags ``es`` for English *hello*."""
    x = w.lower().strip(".,?!¿¡;:'\"")
    if not x:
        return False
    if bool(re.fullmatch(r"al[oó]\??", x)):
        return True
    if x in ("alo", "alo?"):
        return True
    return False


def _remap_misheard_hello_from_spanish_tag(
    transcript: str, language: str | None
) -> tuple[str, str | None]:
    """English ``Hello`` is often decoded as Spanish phone ``Aló`` with ``lang=es`` in ``multi`` mode."""
    t_raw = (transcript or "").strip()
    if not t_raw:
        return transcript, language
    sl = (language or "").strip().lower()
    if not sl.startswith("es"):
        return transcript, language
    parts = re.split(r"\s+", t_raw)
    tokens = [p for p in parts if p.strip(".,?!¿¡;:'\"")]
    if not tokens or len(tokens) > 4:
        return transcript, language
    if not all(_token_is_misheard_spanish_alo_phone(p) for p in tokens):
        return transcript, language
    return "Hello?", "en"


def _coerce_short_alo_transcript_to_hello(transcript: str) -> str:
    """If text is only Alo-like tokens, rewrite to English (``lang`` may be missing in some paths)."""
    t, _ = _remap_misheard_hello_from_spanish_tag(transcript, "es")
    return t


def _is_hi_tagged_romance_latin_noise(text: str, stt_lang: str | None) -> bool:
    """Deepgram sometimes tags a Spanish phone-opening as `hi` — we only want hi/en."""
    sl = (stt_lang or "").strip().lower()
    if not sl.startswith("hi"):
        return False
    t = (text or "").strip()
    if not t or _devanagari_ratio(t) >= 0.05:
        return False
    if any(c in _SPANISH_MARKERS for c in t):
        return True
    words = [w.lower().strip(".,?!¿¡;:'\"") for w in t.split()]
    words = [w for w in words if w]
    if 1 <= len(words) <= 2 and words and all(w in _ROME_HI_ONE_WORD for w in words):
        return True
    return False


def _is_en_tagged_noise(text: str) -> bool:
    """Garbage that Deepgram tags as English in `multi` mode (not user intent)."""
    t = (text or "").strip()
    if not t:
        return True
    if _devanagari_ratio(t) >= 0.05:
        return False
    if any(ch in _EU_ACCENT_CHARS for ch in t):
        return True
    words = [w.lower().strip(".,?!¿¡;:'\"") for w in t.split()]
    words = [w for w in words if w]
    if not words:
        return True
    if len(words) <= 5:
        hits = sum(1 for w in words if w in _EN_TAG_HALLUCINATION_VOCAB)
        if hits >= 1:
            en_stop = sum(1 for w in words if w in _ENGLISH_STOPWORDS)
            if en_stop <= hits:
                return True
    return False


# Nova-3 `multi` often prepends PT/ES-looking glue before real hi/en (room tone / weak audio).
_MULTI_NOVA_ROMANCE_GLUE_FRAGMENTS: frozenset[str] = frozenset(
    {
        "tioto",
        "tahara",
        "dermato",
        "no seu",
        "seu cute",
        "por tahara",
        "dermatolog",
        "verdião",
        "verdiao",
        "tom de tow",
        "una ver",
        # Nova-3 `multi` phone-noise / ES-PT-Latin mash (weak mic)
        "panilla",
        "surtillo",
        "queda tu",
        "merely panilla",
        "solo batao",
        "bharatunga",
        "purdanum",
        "purdanum mantri",
        "sungry",
        "tu mujer",
    }
)


def _strip_spanish_artifacts(text: str) -> str:
    """Remove Spanish-only punctuation/word leaks from an otherwise-valid transcript.

    Deepgram `multi` mode occasionally prepends a tiny Spanish phrase
    ("¿Qué ha", "Perdón,", "Hola,") to a real Hindi/English utterance when
    the first ~200ms of audio is just breath / room tone. Common pattern:
        "¿Qué ha tum sun रहे हो मुझे?"  (lang=hi, real Hindi after the leak)
        "Perdón, agar body me calcium kam ho..."

    Strategy:
      1. Drop the Spanish-only inverted punctuation `¿` / `¡` outright — these
         characters cannot legitimately appear in any Hindi or English text.
      2. If the leftover starts with a known Spanish leader word AND the text
         contains Devanagari further in, drop everything up to the first
         Devanagari character (recovers the real Hindi tail).
      3. Otherwise just return the punctuation-cleaned text.
    """
    if not text:
        return text
    cleaned = text.replace("¿", "").replace("¡", "").lstrip()
    if not cleaned:
        return cleaned
    first_word = cleaned.split(maxsplit=1)[0].strip(".,?!;:'\"").lower()
    spanish_leaders = {
        "qué", "que", "perdón", "perdon", "hola", "sí", "si", "vamos",
        "deje", "menu", "ya", "señor", "señora", "já",
        # Short leaders Deepgram glues before Devanagari / next clause.
        "toma", "solo", "queda",
    }
    if first_word in spanish_leaders:
        for i, ch in enumerate(cleaned):
            if "\u0900" <= ch <= "\u097F":
                tail = cleaned[i:].strip()
                if tail:
                    return tail
                break
    return cleaned


def _is_embedded_romance_noise_clause(s: str) -> bool:
    """True for a Latin fragment that looks like PT/ES `multi` hallucination (not real English)."""
    t = (s or "").strip()
    if not t:
        return False
    if _devanagari_ratio(t) >= 0.04:
        return False
    if any(c in t for c in "¿¡"):
        return True
    eu = sum(1 for c in t if c in _EU_ACCENT_CHARS)
    words = [w.lower().strip(".,?!¿¡;:'\"") for w in t.split()]
    words = [w for w in words if w]
    en_hits = sum(1 for w in words if w in _ENGLISH_STOPWORDS)
    rom_hits = sum(1 for w in words if w in _HALLUCINATION_VOCAB)
    if "já" in words:
        rom_hits += 3
    joined = " ".join(words).lower()
    if "tricóndo" in joined or "tricondo" in joined or "jantio" in joined:
        rom_hits += 3
    if eu >= 2 and en_hits <= 2 and not _looks_like_english_latin(t):
        return True
    if rom_hits >= 2 and rom_hits > en_hits:
        return True
    if rom_hits >= 1 and eu >= 2 and en_hits < 4:
        return True
    jl = joined.lower()
    if any(frag in jl for frag in _MULTI_NOVA_ROMANCE_GLUE_FRAGMENTS):
        return True
    return False


def _latin_prefix_is_pt_es_noise(prefix: str) -> bool:
    """True if Latin text before the first Devanagari char looks like PT/ES `multi` glue."""
    p = (prefix or "").strip()
    if not p or _devanagari_ratio(p) > 0.05:
        return False
    if _looks_like_english_latin(p):
        return False
    low = p.lower()
    if any(frag in low for frag in _MULTI_NOVA_ROMANCE_GLUE_FRAGMENTS):
        return True
    if re.search(r"(?i)^por\b", low) and len(p) < 72:
        return True
    if re.search(r"(?i)\ba\s+dermato\b", low):
        return True
    if re.search(r"(?i)tu\s+mujer", low):
        return True
    if re.search(r"(?i)^toma\b", low) and len(p) < 56:
        return True
    if re.search(r"(?i)^solo\s+batao\b", low) and len(p) < 120:
        return True
    return False


def _strip_glued_pt_latin_prefix_before_devanagari(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    for i, ch in enumerate(t):
        if "\u0900" <= ch <= "\u097F":
            prefix = t[:i].strip()
            if prefix and _latin_prefix_is_pt_es_noise(prefix):
                tail = t[i:].lstrip()
                return tail if tail else t
            break
    return t


def _whole_line_is_pt_es_multi_noise(text: str) -> bool:
    """Latin-only line (no Devanagari) that is almost always Nova-3 `multi` junk."""
    s = (text or "").strip()
    if not s or _devanagari_ratio(s) >= 0.02:
        return False
    if _looks_like_english_latin(s):
        return False
    low = s.lower()
    if any(x in low for x in _MULTI_NOVA_ROMANCE_GLUE_FRAGMENTS):
        return True
    if re.search(r"(?i)^por\b", low) and len(s) < 88:
        words = re.findall(r"[A-Za-zÀ-ÿ]+", s)
        if 1 <= len(words) <= 7:
            en_h = sum(1 for w in words if w.lower() in _ENGLISH_STOPWORDS)
            if en_h <= max(1, len(words) // 2):
                return True
    return False


def _strip_embedded_romance_clauses(text: str) -> str:
    """Drop Portuguese/Spanish sentence chunks glued between Hindi/English in one STT turn."""
    t = (text or "").strip()
    if not t or _devanagari_ratio(t) < 0.015:
        return t
    chunks = re.split(r"(?<=[\.\?\!\u0964\u0965])\s+", t)
    if len(chunks) <= 1:
        return t
    kept: list[str] = []
    dropped_any = False
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        if _is_embedded_romance_noise_clause(ch):
            dropped_any = True
            continue
        kept.append(ch)
    out = " ".join(kept)
    out = re.sub(r"\s{2,}", " ", out).strip()
    if dropped_any and not out:
        return t
    return out if out else t


def _strip_stt_romance_leaks(text: str) -> str:
    """Spanish prefix leaks + PT/ES glue + embedded clauses (Deepgram `multi`)."""
    t = _strip_spanish_artifacts(text)
    t = _strip_glued_pt_latin_prefix_before_devanagari(t)
    t = _strip_embedded_romance_clauses(t)
    t = _strip_glued_pt_latin_prefix_before_devanagari(t)
    if _whole_line_is_pt_es_multi_noise(t):
        return ""
    return t


# Roman tokens users often keep inside otherwise-Devanagari turns — soften strict Hindi-only LLM directive.
_HINDI_MIXED_ROMAN_LOAN_ALLOWS = frozenset(
    {
        "plus",
        "intelligence",
        "intelligent",
        "computer",
        "mobile",
        "internet",
        "hello",
        "hi",
        "ok",
        "okay",
        "love",
        "sorry",
        "thanks",
        "thank",
        "happy",
        "birthday",
        "google",
        "youtube",
    }
)


def _transcript_has_obvious_roman_loanwords(t: str) -> bool:
    for w in re.findall(r"[A-Za-z]+", t or ""):
        if w.lower() in _HINDI_MIXED_ROMAN_LOAN_ALLOWS:
            return True
    return False


# Whole-word Latin tokens Nova-3 often leaves inside Hindi; applied only when
# ``_devanagari_ratio`` is high (see ``_normalize_hinglish_stt_tokens``).
_HINGLISH_LATIN_TOKEN_FIXES: tuple[tuple[str, str], ...] = (
    (r"(?<![A-Za-z'])sun(?![A-Za-z'])", "सुन"),
    (r"(?<![A-Za-z'])motion(?![A-Za-z'])", "मौसम"),
)


def _extract_transcript_delay_s(ev: Any) -> float | None:
    """Parse transcript delay (seconds) if the SDK exposes it on the event.

    Note: ``UserInputTranscribedEvent`` (livekit-agents 1.5.x) does not include
    ``transcript_delay``; the framework only logs it in debug ``extra``. Callers
    should fall back to VAD timing (``since_end_ms``).
    """
    for attr in ("transcript_delay", "transcription_delay"):
        v = getattr(ev, attr, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    md = getattr(ev, "metrics", None)
    if md is not None:
        nested: Any
        if isinstance(md, dict):
            nested = md.get("transcription_delay", md.get("transcript_delay"))
        elif hasattr(md, "model_dump"):
            try:
                dump = md.model_dump()
                nested = dump.get("transcription_delay") or dump.get(
                    "transcript_delay"
                )
            except Exception:
                nested = None
        else:
            nested = getattr(md, "transcription_delay", None) or getattr(
                md, "transcript_delay", None
            )
        if nested is not None:
            try:
                return float(nested)
            except (TypeError, ValueError):
                pass
    if hasattr(ev, "model_dump"):
        try:
            d = ev.model_dump()
            for k in ("transcript_delay", "transcription_delay"):
                if d.get(k) is not None:
                    return float(d[k])
        except Exception:
            pass
    return None


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    lim = min(len(a), len(b))
    while n < lim and a[n] == b[n]:
        n += 1
    return n


def _merge_stt_display_chunks(acc: str, chunk: str) -> str:
    """Accumulate STT interims/finals into one display string for logging / STT test preview.

    Nova-3 often emits (1) a short or comma-variant prefix, (2) a disjoint Hindi tail,
    then (3) refinements of that tail (e.g. ``…सकती`` → ``…सकता हूं?``). Taking the
    single longest chunk drops the English/Hindi preamble — merge instead.
    """
    a = (acc or "").strip()
    b = (chunk or "").strip()
    if not a:
        return b
    if not b:
        return a
    if a in b:
        return b
    if b in a:
        return a
    a_loose = a.rstrip(".,;:!?")
    if b.startswith(a_loose) or b.startswith(a):
        return b
    b_loose = b.rstrip(".,;:!?")
    if a.startswith(b_loose) or a.startswith(b):
        return a
    max_k = min(len(a), len(b))
    for k in range(max_k, 0, -1):
        if a.endswith(b[:k]):
            return (a + b[k:]).strip()
    best_i = -1
    best_cpl = -1
    for i in range(len(a)):
        suf = a[i:]
        if not suf:
            continue
        cpl = _common_prefix_len(suf, b)
        if cpl > best_cpl or (cpl == best_cpl and i > best_i):
            best_cpl = cpl
            best_i = i
    if best_i >= 0 and best_cpl > 0:
        suf = a[best_i:]
        if best_cpl == len(suf) and len(suf) <= len(b):
            return (a[:best_i] + b).strip()
        thr = max(8, int(0.35 * min(len(suf), len(b))))
        if best_cpl >= thr:
            return (a[:best_i] + b).strip()
    return (a + " " + b).replace("  ", " ").strip()


def _normalize_hinglish_stt_tokens(text: str) -> str:
    """Fix Latin spell-outs Nova-3 `multi` often leaves inside Hindi finals.

    Examples: ``मुझे sun रहे हो`` → ``मुझे सुन रहे हो``;
    ``जयपुर का motion`` → ``जयपुर का मौसम``.
    Only runs when Devanagari is already present to avoid touching pure English.
    """
    t = (text or "").strip()
    if not t or _devanagari_ratio(t) < 0.12:
        return t
    out = t
    for pat, repl in _HINGLISH_LATIN_TOKEN_FIXES:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    # कोण (angle/homophone) vs कौन (who) — common Nova glitch in PM / president questions.
    out = re.sub(r"(प्रधानमंत्री|राष्ट्रपति)\s+कोण", r"\1 कौन", out)
    return out


def _dedupe_repeated_stt_tail(text: str) -> str:
    """Collapse duplicate suffixes Nova-3 ``multi`` sometimes appends to the same turn.

    Example: ``Hello, तुम्हारा नाम क्या है? है?`` → ``Hello, तुम्हारा नाम क्या है?``
    Also trims a stray second ``?`` (e.g. ``...कौन है? ?``) from endpointing tails.
    """
    t = (text or "").strip()
    if len(t) < 2:
        return t
    out = re.sub(r"(है\?)(\s*\1)+$", r"\1", t)
    out = re.sub(r"(\?)(\s*\1)+$", r"\1", out)
    return out.strip()


def _is_hallucinated_transcript(text: str, stt_lang: str | None) -> bool:
    """True if the transcript looks like Deepgram-multi noise instead of real Hindi/English speech.

    Strategy: trust Deepgram's language tag. We only support `hi` and `en` (and
    their regional variants). If `multi` mode produces a non-supported language
    tag, the transcript is almost always either (a) the agent's own TTS
    echoing back into the mic, or (b) random noise being shaped into a Spanish
    sentence by the multi-language model. The ONLY exception: real Hindi audio
    sometimes gets mis-tagged as `pt`/`es` while the text itself is correct
    Devanagari — those we keep.
    """
    sl = (stt_lang or "").strip().lower()
    base_lang = sl.split("-", 1)[0]
    t = (text or "").strip()
    if not t:
        return True

    if base_lang not in _HALLUCINATION_LANGS:
        return False

    # Devanagari → real Hindi mis-tagged. Keep.
    if _devanagari_ratio(t) >= _STT_DEVANAGARI_HI_SIGNAL:
        return False

    # Spanish-specific punctuation/diacritics → confirmed hallucination.
    if any(c in _SPANISH_MARKERS for c in t):
        return True

    words = [w.lower().strip(".,?!¿¡;:'\"") for w in t.split()]
    words = [w for w in words if w]
    spanish_hits = sum(1 for w in words if w in _HALLUCINATION_VOCAB)
    english_hits = sum(1 for w in words if w in _ENGLISH_STOPWORDS)

    # ANY Spanish vocab match in a non-English-tagged transcript → drop.
    # Only override if English stopwords clearly dominate (real English).
    if spanish_hits >= 1 and english_hits <= spanish_hits:
        return True

    # Short non-English-tagged ASCII (< 20 chars, fewer than 4 words) is almost
    # never real speech — when a user actually speaks English Deepgram tags it
    # `en`, not `es`/`pt`. Drop it.
    if len(t) < 20 or len(words) < 4:
        return True

    # Long ASCII text with strong English-stopword density → likely mis-tagged
    # English. Keep.
    if english_hits >= 2 and english_hits > spanish_hits:
        return False

    # Default for non-English-tagged ASCII: drop unless we have strong English signal.
    return True


def _user_message_language_directive(transcript: str, stt_lang: str | None) -> str:
    """One-line language directive to prepend to the user message itself.

    System-prompt rules alone aren't enough for stubborn creative prompts
    ("Tell me a story") on llama-3.3-70b when the conversation history is in
    another language — the model anchors to the history. Prefixing the user
    message with a directive embeds the rule in the very text the model is
    responding to, which is much harder to ignore.

    Returns "" if we can't determine a clear language for THIS turn.
    """
    t = (transcript or "").strip()
    sl = (stt_lang or "").strip().lower()
    base_lang = sl.split("-", 1)[0] if sl else ""
    dev_ratio = _devanagari_ratio(t)

    if dev_ratio > 0.12:
        if _transcript_has_obvious_roman_loanwords(t):
            return (
                "[Reply in Devanagari Hindi; keep brief Roman words the user actually used "
                "(e.g. plus, Intelligence); do not add extra English.]"
            )
        return "[Reply in Devanagari Hindi only — no English/Roman letters.]"
    if base_lang == "en":
        return "[Reply in pure English only — no Hindi, no Devanagari.]"
    if base_lang == "hi" and dev_ratio < 0.05:
        return "[Reply in Roman Hinglish only — no Devanagari.]"
    return ""


def _dynamic_reply_language_rule(transcript: str, stt_lang: str | None) -> str:
    """Tight per-turn rule so the LLM answers in the user's language (runs before generate_reply)."""
    t = (transcript or "").strip()
    sl = (stt_lang or "").strip().lower() or None
    dev_ratio = _devanagari_ratio(t)

    bits: list[str] = []
    bits.append("=== STRICT REPLY RULE FOR THIS TURN (override anything in the system prompt) ===")
    if sl:
        bits.append(f'STT language tag: "{sl}".')

    if dev_ratio > 0.12:
        if _transcript_has_obvious_roman_loanwords(t):
            bits.append(
                'The user wrote mainly in Devanagari script (देवनागरी) but included a few '
                'Roman/English words (often loanwords like plus, OK, intelligence). '
                'Reply in Devanagari Hindi for the whole answer; mirror those few Roman words '
                'exactly as the user wrote them when needed — do NOT wrap the reply in extra English.\n'
                'CRITICAL: IGNORE the language of earlier turns; THIS turn\'s Devanagari mix wins.\n'
                'Example: user="मुझे intelligence plus करना है" → reply in Devanagari while keeping '
                '"intelligence" / "plus" in Roman if you echo them.'
            )
        else:
            bits.append(
                'The user wrote in Devanagari script (देवनागरी) in THIS TURN. '
                'You MUST reply ENTIRELY in Devanagari Hindi.\n'
                'CRITICAL: IGNORE the language of any earlier turns. Even if the previous '
                'reply was in English, you MUST switch to Devanagari RIGHT NOW for this reply. '
                'The language of THIS user message overrides everything else.\n'
                'Do NOT use Roman/English letters at all. Do NOT transliterate.\n'
                'Few-shot example showing the required language switch:\n'
                '  [history] user: "What is your name?"\n'
                '  [history] assistant: "My name is Nyra."\n'
                '  [current] user: "तुम्हारा नाम क्या है?"\n'
                '  [correct] assistant: "मेरा नाम न्यरा है।"  ← Devanagari, ignoring English history\n'
                '  [WRONG]   assistant: "My name is Nyra."  ← do NOT do this\n'
                'Example: user="आज का मौसम कैसा है?" → reply="आज मौसम सुहाना है।" '
                '(NOT "Aaj mausam suhana hai")'
            )
    elif sl and (sl == "en" or sl.startswith("en-")):
        bits.append(
            'User spoke English in THIS TURN. You MUST reply in pure English ONLY.\n'
            'CRITICAL: IGNORE the language of any earlier turns. Even if the previous '
            'reply was in Hindi/Devanagari, you MUST switch to English RIGHT NOW for this '
            'reply. The language of THIS user message overrides everything else.\n'
            'NO Hindi words. NO Hinglish. NO Devanagari letters. NO parenthetical translations.\n'
            'Few-shot example showing the required language switch:\n'
            '  [history] user: "मेरा नाम क्या है?"\n'
            '  [history] assistant: "मुझे नहीं पता।"\n'
            '  [current] user: "What is your name?"\n'
            '  [correct] assistant: "My name is Nyra."  ← English, ignoring Hindi history\n'
            '  [WRONG]   assistant: "मेरा नाम न्यरा है।"  ← do NOT do this\n'
            'If the user asked your name, just answer it directly in English '
            '(e.g. "My name is Nyra."). Do not pad with extra phrases.'
        )
    elif sl and (sl == "hi" or sl.startswith("hi")):
        # No Devanagari but lang=hi → Roman-Hindi/Hinglish.
        bits.append(
            "User spoke Hindi/Hinglish in Roman letters. "
            "Reply in natural Roman-Hindi/Hinglish (not Devanagari)."
        )
    else:
        bits.append(
            "Mirror the user's language and script exactly. "
            "If they used Devanagari, reply in Devanagari. "
            "If they used Roman letters, reply in Roman letters."
        )

    bits.append("Keep the reply short (1-2 sentences) and natural like a friend.")
    return " ".join(bits)


def _mirroring_agent_drops_user_transcript(agent: Agent, transcript: str) -> bool:
    """True iff MirroringLanguageAgent would raise StopResponse for this final transcript."""
    if not isinstance(agent, MirroringLanguageAgent):
        return False
    text = _dedupe_repeated_stt_tail(
        _normalize_hinglish_stt_tokens(_strip_stt_romance_leaks((transcript or "").strip()))
    )
    sl = agent._last_stt_language
    if _should_drop_unsupported_stt_language(text, sl):
        return True
    if _is_hi_tagged_romance_latin_noise(text, sl):
        return True
    if _is_hallucinated_transcript(text, sl):
        return True
    if agent._recent_assistant_plain and _is_probably_assistant_echo(
        text, agent._recent_assistant_plain
    ):
        return True
    if sl and (sl == "en" or sl.startswith("en-")) and _is_en_tagged_noise(text):
        return True
    return False


class MirroringLanguageAgent(Agent):
    """Updates system instructions each turn before the LLM runs (see on_user_turn_completed)."""

    def __init__(
        self,
        *,
        base_instructions: str,
        stuck_watch_ref: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(instructions=base_instructions, **kwargs)
        self._base_instructions = base_instructions
        self._last_stt_language: str | None = None
        self._recent_assistant_plain: str = ""
        self._stuck_watch_ref: dict[str, Any] | None = stuck_watch_ref

    def _mark_stuck_watch_serviced_if_same_final(self, text: str) -> None:
        """Stuck-pipeline watchdog shares last_user_final; StopResponse skips user_turn_committed."""
        ref = self._stuck_watch_ref
        if not ref:
            return
        t = (text or "").strip()
        if not t:
            return
        if ref.get("text") == t and not ref.get("serviced", True):
            ref["serviced"] = True
            logger.info(
                "STUCK_WATCH_MARK_SERVICED reason=user_turn_dropped_before_llm chars=%s",
                len(t),
            )

    def note_assistant_spoke(self, text: str | None) -> None:
        t = (text or "").strip()
        if not t:
            return
        self._recent_assistant_plain = t[-1500:] if len(t) > 1500 else t

    def note_stt_language(self, lang: str | None, *, transcript_snippet: str = "") -> None:
        if not lang:
            return
        sl = str(lang).strip().lower()
        t = transcript_snippet or ""
        base = sl.split("-", 1)[0]
        if base in _SUPPORTED_LANGS:
            self._last_stt_language = sl
            return
        # Mis-tag recovery toward hi/en only (Deepgram `multi` emits es/pt/zh/…).
        # Lower Devanagari threshold so short questions (e.g. "कौन है?") still map to `hi`
        # when Deepgram returns `es` on noise or code-switch artifacts.
        if _devanagari_ratio(t) >= _STT_DEVANAGARI_HI_SIGNAL:
            self._last_stt_language = "hi"
            return
        if _looks_like_english_latin(t):
            self._last_stt_language = "en"
            return
        self._last_stt_language = str(lang)

    def reset_turn_language(self) -> None:
        self._last_stt_language = None

    async def on_user_turn_completed(
        self, turn_ctx: lk_llm.ChatContext, new_message: ChatMessage
    ) -> None:
        raw_text = (new_message.text_content or "").strip()
        sl = self._last_stt_language

        # Strip Spanish-only punctuation/leader-word leaks before any other
        # processing. This covers the case where lang=hi but a fake Spanish
        # prefix was glued onto real Hindi content (e.g.
        # "¿Qué ha tum sun रहे हो मुझे?" → "tum sun रहे हो मुझे?").
        text = _strip_stt_romance_leaks(raw_text)
        if text != raw_text:
            logger.info(
                "STRIPPED_STT_ROMANCE_LEAKS lang=%s before=%r after=%r",
                sl, raw_text, text,
            )
            try:
                new_message.content = [text]
            except Exception:
                pass

        hn = _normalize_hinglish_stt_tokens(text)
        if hn != text:
            logger.info("STT_HINGLISH_FIX lang=%s before=%r after=%r", sl, text, hn)
            text = hn
            try:
                new_message.content = [text]
            except Exception:
                pass

        alo_fix = _coerce_short_alo_transcript_to_hello(text)
        if alo_fix != text:
            logger.info(
                "COERCED_STT_ALO_TO_HELLO in_user_turn lang=%s before=%r after=%r",
                sl,
                text,
                alo_fix,
            )
            text = alo_fix
            self._last_stt_language = "en"
            sl = "en"
            try:
                new_message.content = [text]
            except Exception:
                pass

        ded = _dedupe_repeated_stt_tail(text)
        if ded != text:
            logger.info("STT_DEDUPE_TAIL lang=%s before=%r after=%r", sl, text, ded)
            text = ded
            try:
                new_message.content = [text]
            except Exception:
                pass

        if _should_drop_unsupported_stt_language(text, sl):
            logger.warning(
                "DROP_STT_LANG_POLICY lang=%s chars=%s text=%r reason=only_hi_en_supported",
                sl, len(text), text,
            )
            try:
                new_message.content = []
            except Exception:
                pass
            try:
                if turn_ctx.items and turn_ctx.items[-1] is new_message:
                    turn_ctx.items.pop()
            except Exception as e:
                logger.debug("DROP_STT_LANG_HISTORY_TRIM_FAILED err=%r", e)
            self._mark_stuck_watch_serviced_if_same_final(text)
            raise StopResponse()

        if _is_hi_tagged_romance_latin_noise(text, sl):
            logger.warning(
                "DROP_STT_LANG_POLICY lang=%s chars=%s text=%r reason=hi_tagged_romance_latin_noise",
                sl, len(text), text,
            )
            try:
                new_message.content = []
            except Exception:
                pass
            try:
                if turn_ctx.items and turn_ctx.items[-1] is new_message:
                    turn_ctx.items.pop()
            except Exception as e:
                logger.debug("DROP_STT_LANG_HISTORY_TRIM_FAILED err=%r", e)
            self._mark_stuck_watch_serviced_if_same_final(text)
            raise StopResponse()

        # Drop Spanish/Portuguese/French hallucinations from Deepgram `multi`
        # mode on quiet audio. Without this filter the LLM tries to reply to
        # gibberish like "¿Quién me tomará el nombre de un cierto?" and
        # produces useless responses that pollute conversation history.
        if _is_hallucinated_transcript(text, sl):
            logger.warning(
                "DROP_HALLUCINATION lang=%s chars=%s text=%r reason=non_supported_language_no_devanagari_no_english",
                sl, len(text), text,
            )
            # Belt-and-suspenders cleanup: blank the user message AND remove
            # it from the turn context, so even if StopResponse isn't honored
            # by the framework version, the LLM sees an empty user turn
            # instead of Spanish gibberish on subsequent turns.
            try:
                new_message.content = []
            except Exception:
                pass
            try:
                if turn_ctx.items and turn_ctx.items[-1] is new_message:
                    turn_ctx.items.pop()
            except Exception as e:
                logger.debug("HALLUCINATION_HISTORY_TRIM_FAILED err=%r", e)
            self._mark_stuck_watch_serviced_if_same_final(text)
            raise StopResponse()

        if self._recent_assistant_plain and _is_probably_assistant_echo(
            text, self._recent_assistant_plain
        ):
            logger.warning(
                "DROP_HALLUCINATION lang=%s chars=%s text=%r reason=tts_or_room_echo_overlap",
                sl, len(text), text,
            )
            try:
                new_message.content = []
            except Exception:
                pass
            try:
                if turn_ctx.items and turn_ctx.items[-1] is new_message:
                    turn_ctx.items.pop()
            except Exception as e:
                logger.debug("HALLUCINATION_HISTORY_TRIM_FAILED err=%r", e)
            self._mark_stuck_watch_serviced_if_same_final(text)
            raise StopResponse()

        if sl and (sl == "en" or sl.startswith("en-")) and _is_en_tagged_noise(text):
            logger.warning(
                "DROP_HALLUCINATION lang=%s chars=%s text=%r reason=en_tagged_multimodel_noise",
                sl, len(text), text,
            )
            try:
                new_message.content = []
            except Exception:
                pass
            try:
                if turn_ctx.items and turn_ctx.items[-1] is new_message:
                    turn_ctx.items.pop()
            except Exception as e:
                logger.debug("HALLUCINATION_HISTORY_TRIM_FAILED err=%r", e)
            self._mark_stuck_watch_serviced_if_same_final(text)
            raise StopResponse()

        rule = _dynamic_reply_language_rule(text, sl)
        try:
            # Put the per-turn language rule FIRST. LLMs attend more strongly
            # to the top of the system prompt; with the rule at the bottom,
            # llama-3.1-8b ignored it ~30% of the time on short inputs like
            # "Hello" (replied "Namaste!" instead of "Hi!").
            await self.update_instructions(f"{rule}\n\n{self._base_instructions}")
            logger.info("MIRROR_LANG updated instructions stt_lang=%s", sl)
        except Exception as exc:
            logger.error("MIRROR_LANG_INSTRUCTIONS_FAILED error=%r", exc)

        # Belt-and-suspenders: also inject a tiny language directive INTO the
        # user message itself. System-prompt rules alone aren't enough on
        # llama-3.3-70b for creative prompts ("Tell me a story") when the
        # conversation history is in another language — the model defaults to
        # the history's language no matter what the system prompt says.
        # Prefixing the user message with a one-line directive gives the
        # model an instruction it can't ignore (it's part of the very text
        # it's reacting to).
        directive = _user_message_language_directive(text, sl)
        if directive:
            decorated = f"{directive}\n{text}"
            try:
                new_message.content = [decorated]
                logger.info(
                    "INJECTED_USER_LANG_DIRECTIVE stt_lang=%s directive=%r",
                    sl, directive,
                )
            except Exception as e:
                logger.debug("INJECT_USER_DIRECTIVE_FAILED err=%r", e)

        try:
            await super().on_user_turn_completed(turn_ctx, new_message)
        except StopResponse:
            raise
        except Exception as e:
            logger.error("SUPER_CALL_FAILED in on_user_turn_completed: %s", e)


def _is_usable_transcript(text: str) -> bool:
    """Filter transcripts that should not arm stuck-recovery or STT_FINAL summaries."""
    t_raw = (text or "").strip()
    if not t_raw:
        return False
    t = t_raw.lower()
    filler = {
        "uh",
        "um",
        "hmm",
        "huh",
        "so",
        "and",
        "yeah",
        "yes",
        "okay",
        "ok",
        "h",
        "uh,",
        "um,",
        "yeah.",
        "okay.",
        "ok.",
    }
    if t in filler:
        return False
    words = [w for w in t.replace(",", " ").replace(".", " ").split() if w]
    if len(words) >= 2:
        return True
    if len(t_raw) >= 5:
        return True
    dev_chars = sum(1 for c in t_raw if "\u0900" <= c <= "\u097f")
    if dev_chars >= 2:
        return True
    return False


def _normalize_deepgram_language(language: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized in {"hi-en", "hien", "hinglish"}:
        logger.warning("STT_CONFIG language=%r unsupported_for_streaming fallback=multi", language)
        return "multi"
    return language


def _build_llm_from_env(*, provider: str | None = None) -> Any:
    provider = provider or _env_choice("LLM_PROVIDER", {"openai", "groq"}, default="groq")
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.4"))

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY (or OPENAI_API_KEY) for LLM_PROVIDER=groq")
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        return openai.LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for LLM_PROVIDER=openai")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    kwargs: dict[str, Any] = {"model": model, "api_key": api_key, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.LLM(**kwargs)


def _llm_credentials_present(provider: str) -> bool:
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


def _resolve_llm_provider() -> str:
    requested = _env_choice("LLM_PROVIDER", {"openai", "groq"}, default="groq")
    if _llm_credentials_present(requested):
        return requested
    alt = "openai" if requested == "groq" else "groq"
    if _llm_credentials_present(alt):
        logger.warning(
            "LLM_CONFIG requested_provider=%s missing_credentials fallback_provider=%s",
            requested,
            alt,
        )
        return alt
    return requested

def _build_agent(
    *, pipeline: str, stuck_watch_ref: dict[str, Any] | None = None
) -> tuple[Agent, str]:
    _required_env("DEEPGRAM_API_KEY")

    stt_quality_mode, stt_qp = _stt_quality_preset()
    stt_model = os.getenv("DEEPGRAM_MODEL", "nova-3")
    stt_language = _normalize_deepgram_language(os.getenv("DEEPGRAM_LANGUAGE", "multi"))
    endpointing_ms = _env_int("DEEPGRAM_ENDPOINTING_MS", int(stt_qp["dg_endpointing_ms"]))
    sample_rate = int(os.getenv("STT_SAMPLE_RATE", "48000"))
    keyterm_raw = os.getenv("DEEPGRAM_KEYTERM", "").strip()
    keyterm = [p.strip() for p in keyterm_raw.split(",") if p.strip()] if keyterm_raw else None
    # Deepgram's `keyterm` (advanced semantic biasing) is ONLY supported for
    # English models. With `language=multi` or any non-en language, passing
    # keyterms either silently breaks recognition entirely (zero interims, zero
    # finals — exactly the symptom we hit) or returns an obscure 400. Auto-drop
    # them for non-English languages to make .env safe to share across configs.
    if keyterm and not stt_language.lower().startswith("en"):
        logger.info(
            "STT_CONFIG dropping keyterm (count=%d) for language=%s — only supported on English models",
            len(keyterm), stt_language,
        )
        keyterm = None

    if _env_truthy("DEEPGRAM_DETECT_LANGUAGE", default=False):
        logger.warning(
            "STT_CONFIG DEEPGRAM_DETECT_LANGUAGE=true unsupported_with_streaming "
            "ignored — lang fixed from DEEPGRAM_LANGUAGE"
        )
    dg_numerals = _env_truthy("DEEPGRAM_NUMERALS", default=True)
    dg_mip_opt = _env_truthy("DEEPGRAM_MIP_OPT_OUT", default=True)
    dg_profanity = _env_truthy("DEEPGRAM_PROFANITY_FILTER", default=False)

    resolved_pipeline = pipeline
    llm = None
    tts = None
    instructions = os.getenv("AGENT_INSTRUCTIONS", "You are a friendly Hinglish voice assistant. Mirror user's language.")
    _nyra_hi_agree = (
        " When answering as Nyra in Devanagari Hindi, use consistent feminine verb agreement "
        "for yourself (e.g. करती हूँ, बोलती हूँ), not masculine (करूँगा, बोलूँगा)."
    )
    if "feminine verb agreement" not in instructions.lower():
        instructions = instructions.rstrip() + _nyra_hi_agree
    _in_gov_roles = (
        " For India facts: in English, \"President of India\" = राष्ट्रपति; "
        "\"Prime Minister\" = प्रधानमंत्री — answer exactly the office the user asked for "
        "(they are different people)."
    )
    if "राष्ट्रपति" not in instructions and "president of india" not in instructions.lower():
        instructions = instructions.rstrip() + _in_gov_roles

    if pipeline in ["stt_llm", "stt_llm_tts"]:
        provider = _resolve_llm_provider()
        if _llm_credentials_present(provider):
            llm = _build_llm_from_env(provider=provider)
        else:
            resolved_pipeline = "stt_only"

    # === FINAL FIXED CARTESIA TTS ===
    if pipeline == "stt_llm_tts" and cartesia:
        try:
            tts = cartesia.TTS(
                model="sonic-3",                                      # ← yeh zaroori hai
                voice=_required_env("CARTESIA_VOICE_ID"),             # .env se exact voice ID
                language=os.getenv("CARTESIA_LANGUAGE", "hi"),        # ← Hindi ke liye
                # speed float bhi support karta hai (optional)
            )

            logger.info("TTS_FINAL_CONFIG model=sonic-3 language=%s voice=%s",
                os.getenv("CARTESIA_LANGUAGE", "hi"),
                os.getenv("CARTESIA_VOICE_ID"))
            
        except Exception as e:
            logger.error("TTS_INIT_FAILED error=%s → tts=disabled", e)
            tts = None
    # =====================

    logger.info(
        "VOICE_AGENT_CONFIG pipeline=%s stt_quality_mode=%s stt=deepgram model=%s language=%s "
        "endpointing_ms=%s numerals=%s mip_opt_out=%s profanity_filter=%s "
        "llm=%s tts=%s stt_allowed_langs=hi,en_only (post-filter); "
        "note: Deepgram language=multi can still decode non-hi/en — we remap/drop server-side",
        resolved_pipeline,
        stt_quality_mode,
        stt_model,
        stt_language,
        endpointing_ms,
        dg_numerals,
        dg_mip_opt,
        dg_profanity,
        "enabled" if llm else "disabled",
        "enabled" if tts else "disabled",
    )

    stt = deepgram.STT(
        model=stt_model,
        language=stt_language,
        detect_language=False,
        interim_results=True,
        endpointing_ms=endpointing_ms,
        sample_rate=sample_rate,
        vad_events=True,
        keyterm=keyterm,
        smart_format=True,
        punctuate=True,
        filler_words=True,
        no_delay=True,
        numerals=dg_numerals,
        mip_opt_out=dg_mip_opt,
        profanity_filter=dg_profanity,
    )

    _vad_min_speech = _env_float("VAD_MIN_SPEECH_DURATION_S", 0.18)
    _vad_min_silence = _env_float(
        "VAD_MIN_SILENCE_DURATION_S", float(stt_qp["vad_min_silence_s"])
    )
    _vad_threshold = _env_float("VAD_ACTIVATION_THRESHOLD", 0.50)
    logger.info(
        "VAD_CONFIG min_speech_s=%s min_silence_s=%s activation_threshold=%s sample_rate=16000",
        _vad_min_speech,
        _vad_min_silence,
        _vad_threshold,
    )
    vad = silero.VAD.load(
        min_speech_duration=_vad_min_speech,
        min_silence_duration=_vad_min_silence,
        activation_threshold=_vad_threshold,
        sample_rate=16000,
    )

    # MirroringLanguageAgent updates the system prompt per turn with a strict
    # "reply in same language as user" rule and drops Deepgram-multi
    # hallucinations (Spanish/Portuguese gibberish on quiet audio) before they
    # reach the LLM. Plain Agent ignored the system prompt's mirror rule and
    # let hallucinations through, so always use the custom class.
    logger.info("AGENT_INIT class=MirroringLanguageAgent")
    agent = MirroringLanguageAgent(
        base_instructions=instructions,
        stuck_watch_ref=stuck_watch_ref,
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
        allow_interruptions=True,
    )
    logger.info("AGENT_INIT_OK class=MirroringLanguageAgent")

    return agent, resolved_pipeline

async def entrypoint(ctx: JobContext) -> None:
    pipeline_t0 = asyncio.get_event_loop().time()
    job_id = getattr(ctx.job, "id", "unknown")

    def _step(name: str, **fields: Any) -> None:
        now = asyncio.get_event_loop().time()
        since_start_ms = int((now - pipeline_t0) * 1000)
        field_str = " ".join(f"{k}={v!r}" for k, v in fields.items())
        if field_str:
            logger.info(
                "PIPE_STEP job_id=%s step=%s t_ms=%s %s",
                job_id,
                name,
                since_start_ms,
                field_str,
            )
        else:
            logger.info("PIPE_STEP job_id=%s step=%s t_ms=%s", job_id, name, since_start_ms)

    room_name = getattr(ctx.room, "name", "unknown-room")
    _deployment = (os.getenv("DEPLOYMENT") or "").strip().lower() or "development"
    _is_prod = _deployment in {"production", "prod"}
    stt_test_mode = _env_truthy("STT_TEST_MODE", default=False)
    requested_pipeline = _env_choice(
        "VOICE_PIPELINE",
        {"stt_only", "stt_llm", "stt_llm_tts"},
        default="stt_llm",
    )
    if stt_test_mode and _is_prod and not _env_truthy(
        "STT_ALLOW_TEST_IN_PRODUCTION", default=False
    ):
        logger.warning(
            "PRODUCTION_SAFE job_id=%s room=%s STT_TEST_MODE cleared — not for end-user traffic "
            "(set STT_ALLOW_TEST_IN_PRODUCTION=true only for deliberate canary probes)",
            job_id,
            room_name,
        )
        stt_test_mode = False
    elif stt_test_mode and _is_prod:
        logger.error(
            "STT_TEST_MODE in production job_id=%s room=%s — first-final.disconnect "
            "(review STT_ALLOW_TEST_IN_PRODUCTION)",
            job_id,
            room_name,
        )
    if stt_test_mode:
        if requested_pipeline != "stt_only":
            logger.warning(
                "STT_TEST_MODE job_id=%s forcing VOICE_PIPELINE=stt_only (was %s)",
                job_id,
                requested_pipeline,
            )
        requested_pipeline = "stt_only"
    stop_after_first_final = _env_truthy("STT_STOP_AFTER_FIRST_FINAL", default=False) or stt_test_mode
    # False = keep session open for more turns after each LLM reply (normal chat).
    # Set PIPELINE_HALT_AFTER_LLM=true only for one-shot probe / tests.
    halt_after_llm = _env_truthy("PIPELINE_HALT_AFTER_LLM", default=False)
    if _is_prod and halt_after_llm:
        logger.warning(
            "DEPLOYMENT=production with PIPELINE_HALT_AFTER_LLM=true closes the session after the "
            "first assistant reply; use false for end-user voice chat."
        )
    _step(
        "entrypoint_start",
        room=room_name,
        requested_pipeline=requested_pipeline,
        halt_after_llm=halt_after_llm,
        continuous_listen=not stop_after_first_final,
        stt_test_mode=stt_test_mode,
    )
    logger.info(
        "VOICE_AGENT_CONNECTING job_id=%s room=%s deployment=%s requested_pipeline=%s "
        "stt_test_mode=%s",
        job_id,
        room_name,
        _deployment,
        requested_pipeline,
        stt_test_mode,
    )
    await ctx.connect()
    _step("room_connected", room=room_name)

    last_user_final: dict[str, Any] = {
        "turn": 0,
        "text": "",
        "ts": 0.0,
        "serviced": True,
    }
    agent, pipeline = _build_agent(
        pipeline=requested_pipeline, stuck_watch_ref=last_user_final
    )
    _step("agent_configured", pipeline=pipeline, requested_pipeline=requested_pipeline)

    # "stt" = end-of-turn driven by Deepgram phrase boundaries + endpointing (fewer 20s mega-turns).
    # "vad" = end-of-turn mainly on silence (older behaviour). Override with TURN_DETECTION=vad.
    turn_detection = _env_choice(
        "TURN_DETECTION",
        {"stt", "vad", "manual", "realtime_llm"},
        default="stt",
    )
    stt_quality_mode, stt_qp = _stt_quality_preset()
    ep_min = _env_float("TURN_ENDPOINTING_MIN_DELAY_S", float(stt_qp["turn_ep_min_s"]))
    ep_max = _env_float("TURN_ENDPOINTING_MAX_DELAY_S", float(stt_qp["turn_ep_max_s"]))
    logger.info(
        "VOICE_TURN_CONFIG job_id=%s stt_quality_mode=%s turn_detection=%s endpointing_min_s=%s "
        "endpointing_max_s=%s (override with TURN_ENDPOINTING_*; tune preset with STT_QUALITY_MODE)",
        job_id,
        stt_quality_mode,
        turn_detection,
        ep_min,
        ep_max,
    )

    # Force VAD-based interruption detection. The default `adaptive` mode hits
    # LiveKit's interruption-inference cloud, which routinely times out (408
    # after 0.7s) and floods logs with VOICE_SESSION_ERROR. The framework
    # falls back to VAD anyway, so set it explicitly and skip the noise.
    from livekit.agents.voice.turn import (
        EndpointingOptions,
        InterruptionOptions,
    )

    # Barge-in: stop TTS when the user speaks, then process only their latest
    # turn. LiveKit drops in-flight speech on real interruption; we shorten
    # AEC warmup and lower VAD thresholds vs the old echo-safe defaults.
    aec_warmup_s = _env_optional_float("AEC_WARMUP_DURATION_S", 1.0)
    intr_min_dur = _env_float("INTERRUPTION_MIN_DURATION_S", 0.35)
    intr_min_words = _env_int("INTERRUPTION_MIN_WORDS", 1)
    intr_false_timeout = _env_float("INTERRUPTION_FALSE_TIMEOUT_S", 3.5)
    intr_resume = _env_truthy("INTERRUPTION_RESUME_AFTER_FALSE", default=True)
    logger.info(
        "BARGE_IN_CONFIG aec_warmup_s=%s interruption_min_duration_s=%s "
        "interruption_min_words=%s false_interrupt_timeout_s=%s resume_after_false=%s",
        aec_warmup_s,
        intr_min_dur,
        intr_min_words,
        intr_false_timeout,
        intr_resume,
    )

    session = AgentSession(
        aec_warmup_duration=aec_warmup_s,
        turn_handling={
            "turn_detection": turn_detection,
            "endpointing": EndpointingOptions(
                mode="dynamic",
                min_delay=ep_min,
                max_delay=ep_max,
            ),
            "interruption": InterruptionOptions(
                enabled=True,
                mode="vad",
                discard_audio_if_uninterruptible=True,
                min_duration=intr_min_dur,
                min_words=intr_min_words,
                resume_false_interruption=intr_resume,
                false_interruption_timeout=intr_false_timeout,
            ),
            "preemptive_generation": {"enabled": False},
        },
    )
    _step("session_created")
    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            close_on_disconnect=False,
            audio_input=True,
            audio_output=True,
            text_input=False,
            text_output=True,
        ),
    )
    _step("session_started", audio_input=True, audio_output=True)

    last_any_transcript_ts = {"t": 0.0}
    last_speaking_start_ts = {"t": 0.0}
    last_speaking_end_ts = {"t": 0.0}
    last_ended_turn_id = {"v": 0}
    agent_phase_ts = {"thinking": 0.0}
    last_interim_transcript = {"text": "", "ts": 0.0}
    user_is_speaking = {"value": False}
    user_state_seen = {"value": False}
    agent_is_speaking = {"value": False}
    turn = {"id": 0, "interims": 0, "finals": 0, "first_interim_ts": 0.0}
    session_ready_ts: dict[str, float] = {"t": 0.0}
    stt_turn_profile: dict[str, Any] = {
        "speech_ms": -1,
        "first_interim_ms": -1,
        "final_from_start_ms": -1,
        "final_after_listen_ms": -1,
        "last_final_lang": None,
        "ms_since_session_ready": -1,
        "lk_transcript_delay_s": None,
    }
    log_stt_interim_all = _env_truthy("LOG_STT_INTERIM_ALL", default=True)
    log_stt_turn_timing = _env_truthy("LOG_STT_TURN_TIMING", default=True)
    stt_latest_final_text: dict[str, str] = {"text": ""}
    stt_longest_final_text: dict[str, str] = {"text": ""}
    # Longest seen this turn (interims + finals); Nova often emits a short STT_FINAL
    # while longer text was already present in interims.
    stt_longest_stream_text: dict[str, str] = {"text": ""}
    first_final_stop_armed = {"value": False}
    one_shot_done = {"value": False}
    logged_continuous_listen = {"value": False}
    halt_done = {"value": False}
    session_active = {"value": True}
    watch_tasks: set[asyncio.Task[Any]] = set()
    no_user_state_task: asyncio.Task[Any] | None = None

    # Stuck-pipeline watchdog state. STT_FINAL arrives → record it. When
    # the user turn is committed (on_user_turn_completed → conversation
    # item added with role=user), we mark it as serviced. If a final
    # is recorded but never serviced within ~3s we manually fire
    # generate_reply() because LiveKit's internal turn state has been
    # corrupted by a stray late interim that fired AFTER the framework
    # already closed the turn boundary.
    # Watchdog: only fire AFTER the user-state machine leaves "speaking".
    # A 3s blind sleep was wrong — STT_FINAL often arrives seconds before
    # VAD closes the turn, so we called generate_reply() early and duplicated
    # the natural on_user_turn_completed path ("Hello." + "Hi.").
    STUCK_PIPELINE_MAX_WAIT_S = _env_float("STUCK_PIPELINE_MAX_WAIT_S", 45.0)
    STUCK_PIPELINE_GRACE_AFTER_LISTEN_S = _env_float(
        "STUCK_PIPELINE_GRACE_AFTER_LISTEN_S", 0.75
    )
    # Nova-3 `multi` often emits STT_FINAL *after* VAD is already "listening".
    # The agent then commits the user turn on an async tail; 750ms was too
    # short and spuriously triggered generate_reply() (duplicate / wrong metrics).
    STUCK_PIPELINE_MIN_WAIT_AFTER_FINAL_S = _env_float(
        "STUCK_PIPELINE_MIN_WAIT_AFTER_FINAL_S", 2.25
    )
    # After the above waits, require this much *silence on the STT stream*
    # (no interim/final events) before stuck recovery runs. Prevents answering
    # "आज जयपुर का" while Deepgram is still emitting the rest of the sentence.
    STT_RECOVERY_MIN_QUIET_S = _env_float(
        "STT_RECOVERY_MIN_QUIET_S", float(stt_qp["recover_quiet_s"])
    )
    STT_RECOVERY_FRAGMENT_EXTRA_S = _env_float(
        "STT_RECOVERY_FRAGMENT_EXTRA_S", float(stt_qp["recover_frag_extra_s"])
    )
    stt_last_activity_ts: dict[str, float] = {"t": 0.0}
    logger.info(
        "STT_STUCK_GUARD mode=%s quiet_s=%s fragment_extra_s=%s min_wait_after_final_s=%s grace_after_listen_s=%s",
        stt_quality_mode,
        STT_RECOVERY_MIN_QUIET_S,
        STT_RECOVERY_FRAGMENT_EXTRA_S,
        STUCK_PIPELINE_MIN_WAIT_AFTER_FINAL_S,
        STUCK_PIPELINE_GRACE_AFTER_LISTEN_S,
    )

    def _track_watch_task(task: asyncio.Task[Any]) -> asyncio.Task[Any]:
        watch_tasks.add(task)

        def _done(t: asyncio.Task[Any]) -> None:
            watch_tasks.discard(t)

        task.add_done_callback(_done)
        return task

    async def _cancel_watch_tasks() -> None:
        """Cancel background STT/session helpers and wait so asyncio won't warn on shutdown."""
        current = asyncio.current_task()
        pending = [t for t in list(watch_tasks) if t is not current]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _stop_after_first_final(final_text: str) -> None:
        # Only one closer; extra finals from the same utterance must not start
        # duplicate shutdown tasks.
        if first_final_stop_armed["value"]:
            return
        first_final_stop_armed["value"] = True

        immediate = _env_truthy("STT_TEST_IMMEDIATE_CLOSE_ON_FIRST_FINAL", default=False)
        if not immediate:
            # Nova-3 often finalizes a phrase boundary while the user is still
            # talking. Closing on that first final detaches the mic early and
            # loses the rest of the sentence (see STT_CAPTURE_STOP vs late finals).
            t0 = asyncio.get_event_loop().time()
            max_wait = _env_float("STT_TEST_MAX_WAIT_USER_SILENCE_S", 45.0)
            while session_active["value"] and user_is_speaking["value"]:
                if asyncio.get_event_loop().time() - t0 > max_wait:
                    logger.warning(
                        "STT_TEST_STOP_USER_SPEAKING_TIMEOUT waited_s=%s latest_final=%r",
                        max_wait,
                        (stt_latest_final_text["text"] or final_text)[:160],
                    )
                    break
                await asyncio.sleep(0.08)
            if not session_active["value"]:
                return
            drain = _env_float(
                "STT_TEST_POST_LISTENING_DRAIN_S",
                STUCK_PIPELINE_GRACE_AFTER_LISTEN_S,
            )
            if drain > 0:
                logger.info(
                    "STT_CAPTURE_STOP_DRAIN post_listening_s=%s hint=trailing_stt_finals",
                    drain,
                )
                await asyncio.sleep(drain)
            # Fixed drain still closes before Nova-3 sends a duplicate suffix final
            # (e.g. a second ``है?``); require STT idle like stuck-recovery quiet.
            quiet_need = _env_float("STT_TEST_TRAILING_QUIET_S", STT_RECOVERY_MIN_QUIET_S)
            max_tail = _env_float("STT_TEST_MAX_TRAILING_WAIT_S", 4.0)
            if quiet_need > 0 and session_active["value"]:
                logger.info(
                    "STT_CAPTURE_STOP_TRAILING_QUIET need_s=%s max_wait_s=%s hint=no_stt_events",
                    quiet_need,
                    max_tail,
                )
                tail_deadline = asyncio.get_event_loop().time() + max_tail
                while session_active["value"]:
                    idle = asyncio.get_event_loop().time() - float(
                        stt_last_activity_ts.get("t") or 0.0
                    )
                    if idle >= quiet_need:
                        break
                    if asyncio.get_event_loop().time() >= tail_deadline:
                        logger.warning(
                            "STT_TEST_TRAILING_QUIET_TIMEOUT idle_s=%.3f need_s=%s",
                            idle,
                            quiet_need,
                        )
                        break
                    sl_e = quiet_need - idle
                    await asyncio.sleep(min(0.08, max(0.02, sl_e + 0.02)))

        if not session_active["value"]:
            return
        if one_shot_done["value"]:
            return
        one_shot_done["value"] = True
        preview = (
            stt_longest_stream_text["text"]
            or stt_longest_final_text["text"]
            or stt_latest_final_text["text"]
            or final_text
        ).strip()
        _step("stop_triggered_after_stt", text=preview)
        logger.info(
            "STT_CAPTURE_STOP reason=%s pipeline=%s text=%r",
            "first_final_immediate" if immediate else "first_final_after_user_listening",
            pipeline,
            preview,
        )
        try:
            if pipeline == "stt_only":
                grace = _env_float("STT_TEST_DISCONNECT_DELAY_S", 0.35)
                if grace > 0:
                    logger.info(
                        "STT_CAPTURE_STOP ui_grace_s=%s hint=time_for_client_to_show_transcript",
                        grace,
                    )
                    await asyncio.sleep(grace)
                # Let the user turn finish committing before pausing speech scheduling;
                # otherwise LiveKit logs ``skipping user input, speech scheduling is paused``.
                flush = _env_float("STT_TEST_COMMIT_FLUSH_S", 0.45)
                if flush > 0:
                    logger.info(
                        "STT_CAPTURE_STOP commit_flush_s=%s hint=user_turn_pipeline_before_close",
                        flush,
                    )
                    await asyncio.sleep(flush)
            session_active["value"] = False
            await _cancel_watch_tasks()
            # Close the voice session before tearing down the room, otherwise
            # LiveKit may still try to flush transcription/text streams and log
            # benign "engine is closed" warnings.
            await session.aclose()
            _step("session_closed_after_stt")
            disconnect_result = ctx.room.disconnect()
            if asyncio.iscoroutine(disconnect_result):
                await disconnect_result
            _step("room_disconnected_after_stt")
        except Exception as exc:
            logger.error("STT_CAPTURE_STOP_FAILED error=%r", exc)

    async def _halt_after_llm(reason: str, assistant_text: str | None) -> None:
        if halt_done["value"]:
            return
        halt_done["value"] = True
        _step("halt_after_llm", reason=reason)
        logger.info(
            "PIPELINE_HALT_AFTER_LLM reason=%s assistant_preview=%r",
            reason,
            (assistant_text or "")[:240],
        )
        try:
            session_active["value"] = False
            await _cancel_watch_tasks()
            await session.aclose()
            _step("session_closed_after_llm_halt")
            disconnect_result = ctx.room.disconnect()
            if asyncio.iscoroutine(disconnect_result):
                await disconnect_result
            _step("room_disconnected_after_llm_halt")
        except Exception as exc:
            logger.error("PIPELINE_HALT_AFTER_LLM_FAILED error=%r", exc)

    async def _warn_if_no_stt_after_speaking() -> None:
        snapshot = last_speaking_end_ts["t"]
        turn_id = turn["id"]
        # Capture agent state at the moment the "speech" started — if it was
        # already mid-TTS, this VAD trigger is almost certainly TTS echo
        # leaking back through the mic, NOT real user speech. We log it as
        # debug instead of warning to keep real STT failures visible.
        was_agent_speaking = agent_is_speaking["value"]
        await asyncio.sleep(2.5)
        if not session_active["value"]:
            return
        if last_speaking_end_ts["t"] != snapshot:
            return
        # Finals often arrive just before VAD sets "listening", so last_any_transcript_ts can be
        # slightly *before* last_speaking_end_ts even though the turn had STT output — do not warn.
        if last_any_transcript_ts["t"] < snapshot and turn["finals"] == 0:
            if last_speaking_start_ts["t"] > 0:
                speech_ms = max(0, int((snapshot - last_speaking_start_ts["t"]) * 1000))
            else:
                speech_ms = -1
            if was_agent_speaking:
                logger.debug(
                    "STT_NO_TRANSCRIPT_ECHO turn=%s speech_ms=%s "
                    "(suppressed: agent was speaking — likely TTS echo) "
                    "interims=%s finals=%s",
                    turn_id, speech_ms, turn["interims"], turn["finals"],
                )
            else:
                logger.warning(
                    "STT_NO_TRANSCRIPT turn=%s speech_ms=%s wait_ms=2500 "
                    "interims=%s finals=%s last_interim=%r",
                    turn_id, speech_ms, turn["interims"], turn["finals"],
                    last_interim_transcript["text"],
                )

    async def _warn_if_no_user_state_seen() -> None:
        # Grace period after session start so mic publish / first VAD do not trip a false warning.
        await asyncio.sleep(_env_float("USER_STATE_WATCH_DELAY_S", 1.25))
        await asyncio.sleep(8.0)
        if not session_active["value"]:
            return
        if user_state_seen["value"]:
            return
        total_ms = int((_env_float("USER_STATE_WATCH_DELAY_S", 1.25) + 8.0) * 1000)
        logger.warning(
            "STT_NO_USER_STATE job_id=%s wait_ms=%s hint=likely_silent_or_low_audio_from_client",
            job_id,
            total_ms,
        )
        _step("no_user_state_timeout", wait_ms=total_ms)

    @session.on("user_state_changed")
    def _on_user_state_changed(ev) -> None:
        user_state_seen["value"] = True
        t = no_user_state_task
        if t and not t.done():
            t.cancel()
        state_text = str(getattr(ev, "new_state", ev))
        now = asyncio.get_event_loop().time()
        user_is_speaking["value"] = (state_text == "speaking")
        logger.info("STT_USER_STATE state=%s turn=%s", state_text, turn["id"])

        if state_text == "speaking":
            turn["id"] += 1
            turn["interims"] = 0
            turn["finals"] = 0
            turn["first_interim_ts"] = 0.0
            if isinstance(agent, MirroringLanguageAgent):
                agent.reset_turn_language()
            last_speaking_start_ts["t"] = now
            # Reset end marker so per-turn since_end_ms doesn't carry over huge gaps
            # from previous turns/silence windows.
            last_speaking_end_ts["t"] = 0.0
            last_interim_transcript["text"] = ""
            last_interim_transcript["ts"] = 0.0
            stt_longest_final_text["text"] = ""
            stt_longest_stream_text["text"] = ""
            stt_turn_profile["speech_ms"] = -1
            stt_turn_profile["first_interim_ms"] = -1
            stt_turn_profile["final_from_start_ms"] = -1
            stt_turn_profile["final_after_listen_ms"] = -1
            stt_turn_profile["last_final_lang"] = None
            stt_turn_profile["ms_since_session_ready"] = (
                int((now - session_ready_ts["t"]) * 1000) if session_ready_ts["t"] else -1
            )
            stt_turn_profile["lk_transcript_delay_s"] = None
            logger.info("STT_TURN_START turn=%s", turn["id"])
            _step("user_speaking_started", turn=turn["id"])
        elif state_text == "listening":
            last_speaking_end_ts["t"] = now
            last_ended_turn_id["v"] = turn["id"]
            # If "speaking" never fired (low-volume mic / VAD missed), skip the
            # bogus speech_ms (would otherwise be unix-epoch * 1000) and just
            # record the end. Useful signal: turn["id"]==0 means no start ever.
            if last_speaking_start_ts["t"] > 0 and turn["id"] > 0:
                speech_ms = max(0, int((now - last_speaking_start_ts["t"]) * 1000))
            else:
                speech_ms = -1
                logger.warning(
                    "STT_TURN_END_WITHOUT_START hint=vad_missed_or_low_audio turn=%s "
                    "interims=%s finals=%s",
                    turn["id"], turn["interims"], turn["finals"],
                )
            logger.info(
                "STT_TURN_END turn=%s speech_ms=%s interims=%s finals=%s",
                turn["id"], speech_ms, turn["interims"], turn["finals"],
            )
            stt_turn_profile["speech_ms"] = speech_ms
            _step("user_speaking_ended", turn=turn["id"], speech_ms=speech_ms)
            _track_watch_task(asyncio.create_task(_warn_if_no_stt_after_speaking()))

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev) -> None:
        state = str(getattr(ev, "new_state", ev))
        logger.info("VOICE_AGENT_STATE job_id=%s state=%s", job_id, state)
        agent_is_speaking["value"] = (state == "speaking")
        if state == "thinking":
            agent_phase_ts["thinking"] = asyncio.get_event_loop().time()

    async def _kick_stuck_pipeline(stuck_turn_id: int, stuck_text: str) -> None:
        """Last-resort LLM kick if a finalized transcript never commits.

        Must NOT run while ``user_is_speaking`` is still true — that only
        means VAD has not closed the current utterance yet; the framework
        will commit via on_user_turn_completed once ``listening`` fires.
        Firing generate_reply() mid-utterance duplicates turns and starves
        STT (second greeting / missing transcripts).
        """
        t0 = asyncio.get_event_loop().time()
        saw_user_listening = False
        try:
            while asyncio.get_event_loop().time() - t0 < STUCK_PIPELINE_MAX_WAIT_S:
                await asyncio.sleep(0.15)
                if not session_active["value"]:
                    return
                if last_user_final["serviced"]:
                    return
                if last_user_final["turn"] != stuck_turn_id:
                    return
                if last_user_final["text"] != stuck_text:
                    return
                if user_is_speaking["value"]:
                    continue
                saw_user_listening = True
                break
            else:
                logger.warning(
                    "STUCK_PIPELINE_GIVE_UP turn=%s text_preview=%r "
                    "reason=user_still_speaking_after_max_wait_s=%s",
                    stuck_turn_id,
                    (stuck_text or "")[:80],
                    STUCK_PIPELINE_MAX_WAIT_S,
                )
                return
        except asyncio.CancelledError:
            return
        if not saw_user_listening:
            return
        try:
            loop = asyncio.get_event_loop()
            now = loop.time()
            final_ts = float(last_user_final.get("ts") or 0.0)
            wait_deadline = max(
                now + STUCK_PIPELINE_GRACE_AFTER_LISTEN_S,
                final_ts + STUCK_PIPELINE_MIN_WAIT_AFTER_FINAL_S,
            )
            delay = max(0.0, wait_deadline - loop.time())
            if delay > 0:
                logger.debug(
                    "STUCK_WATCH_SLEEP delay_s=%.3f final_age_s=%s",
                    delay,
                    (now - final_ts) if final_ts else None,
                )
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if not session_active["value"]:
            return
        if last_user_final["serviced"]:
            return
        if last_user_final["turn"] != stuck_turn_id:
            return
        if last_user_final["text"] != stuck_text:
            return
        if user_is_speaking["value"] or agent_is_speaking["value"]:
            return
        need_quiet = STT_RECOVERY_MIN_QUIET_S + (
            STT_RECOVERY_FRAGMENT_EXTRA_S
            if _hindi_likely_incomplete_phrase(stuck_text)
            else 0.0
        )
        try:
            while session_active["value"]:
                if last_user_final["serviced"]:
                    return
                if last_user_final["turn"] != stuck_turn_id:
                    return
                if last_user_final["text"] != stuck_text:
                    return
                if user_is_speaking["value"]:
                    return
                loop = asyncio.get_event_loop()
                idle = loop.time() - float(stt_last_activity_ts.get("t") or 0.0)
                if idle >= need_quiet:
                    break
                await asyncio.sleep(min(0.12, need_quiet - idle + 0.02))
        except asyncio.CancelledError:
            return
        if not session_active["value"]:
            return
        if last_user_final["serviced"]:
            return
        if last_user_final["turn"] != stuck_turn_id:
            return
        if last_user_final["text"] != stuck_text:
            return
        if user_is_speaking["value"] or agent_is_speaking["value"]:
            return
        if turn["id"] > stuck_turn_id:
            logger.warning(
                "STUCK_PIPELINE_RECOVERY_SKIPPED turn=%s reason=newer_user_turn_started "
                "active_turn=%s",
                stuck_turn_id,
                turn["id"],
            )
            last_user_final["serviced"] = True
            return
        if _mirroring_agent_drops_user_transcript(agent, stuck_text):
            logger.warning(
                "STUCK_PIPELINE_RECOVERY_SKIPPED turn=%s text=%r "
                "reason=transcript_would_be_dropped_by_hallucination_gate",
                stuck_turn_id,
                stuck_text,
            )
            last_user_final["serviced"] = True
            return
        if pipeline == "stt_only":
            logger.info(
                "STUCK_PIPELINE_RECOVERY_SKIPPED turn=%s reason=stt_only_no_llm",
                stuck_turn_id,
            )
            last_user_final["serviced"] = True
            return
        logger.warning(
            "STUCK_PIPELINE_RECOVERY turn=%s text=%r — no user_turn_committed "
            "after VAD listening; manually invoking generate_reply()",
            stuck_turn_id, stuck_text,
        )
        try:
            last_user_final["serviced"] = True
            session.generate_reply(user_input=stuck_text)
        except Exception as exc:
            logger.error("STUCK_PIPELINE_RECOVERY_FAILED err=%r", exc)
            last_user_final["serviced"] = False

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev) -> None:
        transcript = (getattr(ev, "transcript", "") or "").strip()
        is_final = bool(getattr(ev, "is_final", False))
        language = getattr(ev, "language", None)
        if is_final and transcript:
            transcript = _normalize_hinglish_stt_tokens(_strip_stt_romance_leaks(transcript))
        if transcript:
            _pre_t, _pre_lang = transcript, language
            transcript, language = _remap_misheard_hello_from_spanish_tag(transcript, language)
            if transcript != _pre_t or language != _pre_lang:
                logger.info(
                    "REMAPPED_STT_ES_ALO_AS_HELLO before=%r lang_before=%s after=%r lang_after=%s",
                    _pre_t,
                    _pre_lang,
                    transcript,
                    language,
                )
        if isinstance(agent, MirroringLanguageAgent):
            agent.note_stt_language(language, transcript_snippet=transcript)
        now = asyncio.get_event_loop().time()
        stt_last_activity_ts["t"] = now
        last_any_transcript_ts["t"] = now
        if transcript:
            stt_longest_stream_text["text"] = _merge_stt_display_chunks(
                stt_longest_stream_text["text"], transcript
            )

        since_start_ms = (
            int((now - last_speaking_start_ts["t"]) * 1000)
            if last_speaking_start_ts["t"]
            else -1
        )
        since_end_ms = (
            int((now - last_speaking_end_ts["t"]) * 1000)
            if last_speaking_end_ts["t"]
            else -1
        )
        if is_final:
            logger.info(
                "STT_FINAL turn=%s lang=%s chars=%d since_start_ms=%s since_end_ms=%s text=%r",
                turn["id"], language, len(transcript), since_start_ms, since_end_ms, transcript,
            )
        else:
            _emit = logger.info if log_stt_interim_all else logger.debug
            _emit(
                "STT_INTERIM turn=%s lang=%s chars=%d since_start_ms=%s since_end_ms=%s text=%r",
                turn["id"], language, len(transcript), since_start_ms, since_end_ms, transcript,
            )

        if not is_final and transcript:
            turn["interims"] += 1
            if not turn["first_interim_ts"]:
                turn["first_interim_ts"] = now
                stt_turn_profile["first_interim_ms"] = since_start_ms
                logger.info(
                    "STT_FIRST_INTERIM turn=%s latency_ms=%s text=%r",
                    turn["id"], since_start_ms, transcript,
                )
                _step("stt_first_interim", turn=turn["id"], latency_ms=since_start_ms)
            last_interim_transcript["text"] = transcript
            last_interim_transcript["ts"] = now

        if is_final:
            if transcript:
                stt_latest_final_text["text"] = transcript
                if len(transcript) > len(stt_longest_final_text["text"]):
                    stt_longest_final_text["text"] = transcript
            turn["finals"] += 1
            first_interim_latency_ms = (
                int((turn["first_interim_ts"] - last_speaking_start_ts["t"]) * 1000)
                if turn["first_interim_ts"] and last_speaking_start_ts["t"]
                else -1
            )
            logger.info(
                "STT_FINAL_SUMMARY turn=%s usable=%s interims=%s finals=%s "
                "first_interim_latency_ms=%s final_latency_from_start_ms=%s text=%r",
                turn["id"], _is_usable_transcript(transcript), turn["interims"],
                turn["finals"], first_interim_latency_ms, since_start_ms, transcript,
            )
            _step(
                "stt_final_received",
                turn=turn["id"],
                latency_ms=since_start_ms,
                chars=len(transcript),
            )
            stt_turn_profile["final_from_start_ms"] = since_start_ms
            stt_turn_profile["final_after_listen_ms"] = since_end_ms
            stt_turn_profile["last_final_lang"] = str(language) if language is not None else None
            _td = _extract_transcript_delay_s(ev)
            if _td is None and since_end_ms >= 0:
                _td = since_end_ms / 1000.0
            if _td is not None:
                stt_turn_profile["lk_transcript_delay_s"] = _td

            # Record this final as the latest unserviced one. The watchdog
            # waits until VAD reports ``listening`` (utterance finished),
            # then grants a short grace window for the natural
            # on_user_turn_completed path. Only if nothing commits does it
            # call generate_reply() (true wedged state).
            if _is_usable_transcript(transcript) and pipeline != "stt_only":
                last_user_final["turn"] = turn["id"]
                last_user_final["text"] = transcript
                last_user_final["ts"] = now
                last_user_final["serviced"] = False
                _track_watch_task(
                    asyncio.create_task(
                        _kick_stuck_pipeline(turn["id"], transcript)
                    )
                )

            if stop_after_first_final:
                asyncio.create_task(_stop_after_first_final(transcript))
            else:
                if not logged_continuous_listen["value"]:
                    logged_continuous_listen["value"] = True
                    logger.info(
                        "STT_LISTEN_CONTINUES reason=continuous_probe first_final=%r",
                        transcript,
                    )

    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev) -> None:
        item = getattr(ev, "item", None)
        role = getattr(item, "role", None)

        # LiveKit may commit the user ChatMessage and fire this handler before
        # ``MirroringLanguageAgent.on_user_turn_completed`` can clear the same
        # transcript. Apply the same hi/en + hallucination gate here so dropped
        # Spanish (and similar) never appears in context or logs as user text.
        if (
            isinstance(item, ChatMessage)
            and role == "user"
            and isinstance(agent, MirroringLanguageAgent)
        ):
            raw_ut = (item.text_content or "").strip()
            ut = _dedupe_repeated_stt_tail(
                _normalize_hinglish_stt_tokens(_strip_stt_romance_leaks(raw_ut))
            )
            ut_alo = _coerce_short_alo_transcript_to_hello(ut)
            if ut_alo != ut:
                logger.info(
                    "CONVERSATION_COERCED_ALO job_id=%s before=%r after=%r",
                    job_id,
                    ut,
                    ut_alo,
                )
                ut = ut_alo
            if ut != raw_ut:
                logger.info(
                    "CONVERSATION_STT_SCRUB job_id=%s before_chars=%s after_chars=%s",
                    job_id,
                    len(raw_ut),
                    len(ut),
                )
                try:
                    item.content = [ut]
                except Exception as e:
                    logger.debug("CONVERSATION_STT_SCRUB_APPLY_FAILED err=%r", e)
            if not ut and raw_ut.strip():
                try:
                    item.content = []
                except Exception as e:
                    logger.debug("CONVERSATION_STT_SCRUB_EMPTY_CLEAR_FAILED err=%r", e)
                try:
                    agent._mark_stuck_watch_serviced_if_same_final(raw_ut)
                except Exception:
                    pass
                logger.warning(
                    "DROP_USER_CONVERSATION_ITEM job_id=%s chars=0 text= reason=stt_scrub_empty",
                    job_id,
                )
            elif ut and _mirroring_agent_drops_user_transcript(agent, ut):
                logger.warning(
                    "DROP_USER_CONVERSATION_ITEM job_id=%s chars=%s text=%r",
                    job_id,
                    len(ut),
                    ut,
                )
                try:
                    item.content = []
                except Exception as e:
                    logger.debug("DROP_USER_CONVERSATION_ITEM_CLEAR_FAILED err=%r", e)
                try:
                    agent._mark_stuck_watch_serviced_if_same_final(ut)
                except Exception:
                    pass

        content = getattr(item, "content", None)
        logger.info(
            "VOICE_CONVERSATION_ITEM job_id=%s role=%s content=%s",
            job_id,
            role,
            content,
        )

        if isinstance(agent, MirroringLanguageAgent) and role == "assistant":
            tx = getattr(item, "text_content", None)
            if isinstance(tx, str) and tx.strip():
                try:
                    agent.note_assistant_spoke(tx)
                except Exception as e:
                    logger.debug("NOTE_ASSISTANT_SPOKE_FAILED err=%r", e)

        if isinstance(item, ChatMessage):
            raw_m = getattr(item, "metrics", None)
            if raw_m is None:
                md = {}
            elif isinstance(raw_m, dict):
                md = dict(raw_m)
            elif hasattr(raw_m, "model_dump"):
                md = raw_m.model_dump()
            else:
                md = {}
            if role == "user":
                last_user_final["serviced"] = True
                utext = (item.text_content or "").strip()
                preview = (utext[:100] + "…") if len(utext) > 100 else utext
                td_coalesced = md.get("transcription_delay")
                if td_coalesced is None:
                    td_coalesced = stt_turn_profile.get("lk_transcript_delay_s")
                logger.info(
                    "PIPE_LATENCY job_id=%s phase=user_turn_committed turn=%s chars=%s "
                    "transcription_delay_s=%s end_of_turn_delay_s=%s on_user_turn_completed_delay_s=%s",
                    job_id,
                    last_ended_turn_id["v"],
                    len(utext),
                    td_coalesced,
                    md.get("end_of_turn_delay"),
                    md.get("on_user_turn_completed_delay"),
                )
                if log_stt_turn_timing:
                    sl = (
                        agent._last_stt_language
                        if isinstance(agent, MirroringLanguageAgent)
                        else None
                    )
                    logger.info(
                        "STT_TURN_TIMING job_id=%s turn=%s deployment=%s pipeline=%s "
                        "stt_quality_mode=%s stt_lang=%s chars=%s "
                        "vad_speech_ms=%s first_interim_ms=%s final_from_speech_start_ms=%s "
                        "final_after_listen_ms=%s interims=%s finals=%s "
                        "ms_since_session_ready=%s lk_transcript_delay_s=%s "
                        "msg_transcription_delay_s=%s fw_end_of_turn_delay_s=%s text_preview=%r",
                        job_id,
                        last_ended_turn_id["v"],
                        _deployment,
                        pipeline,
                        stt_quality_mode,
                        sl,
                        len(utext),
                        stt_turn_profile.get("speech_ms", -1),
                        stt_turn_profile.get("first_interim_ms", -1),
                        stt_turn_profile.get("final_from_start_ms", -1),
                        stt_turn_profile.get("final_after_listen_ms", -1),
                        turn["interims"],
                        turn["finals"],
                        stt_turn_profile.get("ms_since_session_ready", -1),
                        stt_turn_profile.get("lk_transcript_delay_s"),
                        md.get("transcription_delay"),
                        md.get("end_of_turn_delay"),
                        preview,
                    )
            elif role == "assistant":
                now_item = asyncio.get_event_loop().time()
                t0 = agent_phase_ts.get("thinking") or 0.0
                think_to_commit_ms = int((now_item - t0) * 1000) if t0 else -1
                logger.info(
                    "PIPE_LATENCY job_id=%s phase=assistant_message turn=%s chars=%s interrupted=%s "
                    "llm_ttft_s=%s e2e_user_done_to_agent_start_s=%s think_state_to_commit_ms=%s",
                    job_id,
                    last_ended_turn_id["v"],
                    len((item.text_content or "")),
                    getattr(item, "interrupted", None),
                    md.get("llm_node_ttft"),
                    md.get("e2e_latency"),
                    think_to_commit_ms,
                )

        if pipeline == "stt_llm" and halt_after_llm and isinstance(item, ChatMessage):
            if role == "assistant":
                text = item.text_content
                if not (text or "").strip():
                    return
                _step(
                    "llm_assistant_message",
                    chars=len(text or ""),
                    interrupted=getattr(item, "interrupted", None),
                )
                logger.info(
                    "LLM_ASSISTANT_REPLY chars=%s interrupted=%s text=%r",
                    len(text or ""),
                    getattr(item, "interrupted", None),
                    text,
                )
                asyncio.create_task(_halt_after_llm("assistant_message", text))

    @session.on("metrics_collected")
    def _on_metrics_collected(ev) -> None:
        metrics = getattr(ev, "metrics", None)
        if metrics is None:
            return
        metrics_type = getattr(metrics, "type", "")
        if metrics_type == "stt_metrics":
            logger.info(
                "STT_METRICS job_id=%s request_id=%s audio_duration=%s streamed=%s reused=%s model=%s",
                job_id,
                getattr(metrics, "request_id", ""),
                getattr(metrics, "audio_duration", None),
                getattr(metrics, "streamed", None),
                getattr(metrics, "connection_reused", None),
                getattr(getattr(metrics, "metadata", None), "model_name", None),
            )
        elif metrics_type == "eou_metrics":
            meta = getattr(metrics, "metadata", None)
            td_label = None
            if meta is not None:
                td_label = getattr(meta, "model_provider", None) or getattr(meta, "model_name", None)
            logger.info(
                "PIPE_LATENCY job_id=%s phase=eou turn_detection=%s eou_delay_ms=%s "
                "transcript_after_speech_end_ms=%s on_user_turn_completed_ms=%s speech_id=%s",
                job_id,
                td_label,
                int((getattr(metrics, "end_of_utterance_delay", 0) or 0) * 1000),
                int((getattr(metrics, "transcription_delay", 0) or 0) * 1000),
                int((getattr(metrics, "on_user_turn_completed_delay", 0) or 0) * 1000),
                getattr(metrics, "speech_id", None),
            )
        elif metrics_type == "llm_metrics":
            logger.info(
                "PIPE_LATENCY job_id=%s phase=llm_inference ttft_ms=%s duration_ms=%s tokens_per_s=%s "
                "prompt_tokens=%s completion_tokens=%s cancelled=%s request_id=%s",
                job_id,
                int((getattr(metrics, "ttft", 0) or 0) * 1000),
                int((getattr(metrics, "duration", 0) or 0) * 1000),
                getattr(metrics, "tokens_per_second", None),
                getattr(metrics, "prompt_tokens", None),
                getattr(metrics, "completion_tokens", None),
                getattr(metrics, "cancelled", None),
                getattr(metrics, "request_id", None),
            )
        elif os.getenv("LOG_VAD_METRICS", "").strip().lower() in {"1", "true", "yes", "on"}:
            logger.info("VAD_METRICS metrics=%s", metrics)

    @session.on("error")
    def _on_session_error(ev) -> None:
        logger.error("VOICE_SESSION_ERROR job_id=%s event=%s", job_id, ev)
        err = getattr(ev, "error", None)
        if err is not None:
            em = str(err).lower()
            if "429" in em and (
                "rate limit" in em
                or "tokens per day" in em
                or "tpd" in em
                or "rate_limit_exceeded" in em
            ):
                logger.error(
                    "LLM_QUOTA_EXHAUSTED job_id=%s hint=use_smaller_model_in_dotenv "
                    "e.g. LLM_MODEL=llama-3.1-8b-instant or wait_for_daily_reset",
                    job_id,
                )

    tts_status = "enabled" if getattr(agent, "tts", None) is not None else "disabled"
    logger.info(
        "VOICE_AGENT_READY job_id=%s room=%s pipeline=%s speak_now=true text_output=true tts=%s",
        job_id,
        room_name,
        pipeline,
        tts_status,
    )
    _step("ready_for_stt")
    session_ready_ts["t"] = asyncio.get_event_loop().time()
    no_user_state_task = _track_watch_task(asyncio.create_task(_warn_if_no_user_state_seen()))

    try:
        while ctx.room.isconnected():
            await asyncio.sleep(1)
    finally:
        session_active["value"] = False
        await _cancel_watch_tasks()
        try:
            await session.aclose()
        except Exception as exc:
            logger.debug("session.aclose after disconnect job_id=%s err=%r", job_id, exc)
        _step("entrypoint_room_loop_exited")


import time as _time_mod

_WORKER_START_MONOTONIC = _time_mod.monotonic()
# Shorter default: first ~1s still drops the usual stale burst right after worker
# registration; a 5s window often rejects a legitimate first connect soon after `dev` start.
_STALE_DISPATCH_DROP_WINDOW_S = _env_float("STALE_DISPATCH_DROP_WINDOW_S", 1.25)


async def _request_fnc(req: JobRequest) -> None:
    """Reject stale dispatches queued before this worker started.

    When the user opens the page while `python agent.py` is NOT running,
    `token_server` still creates a LiveKit dispatch. The dispatch sits
    in LiveKit's queue. When the agent finally starts, ALL queued
    dispatches fire within milliseconds of worker registration — which:
      1. Spawns 2-3 agent sessions for the same room.
      2. Saturates the worker so the *legitimate* fresh dispatch
         (the one from the user's current browser tab) can't establish
         its room connection within 10s and dies with
         "wait_pc_connection timed out".

    Drop anything that arrives in the first STALE_DISPATCH_DROP_WINDOW_S seconds of worker uptime
    (default ~1.25s; override via env). Bursts from stale pre-start dispatches land there; a
    window that is too wide rejects a normal first browser connect right after ``agent.py dev``.
    If a dispatch was rejected, disconnect and connect again (new token + dispatch), or increase
    the window only while debugging duplicate-agent issues.
    """
    age = _time_mod.monotonic() - _WORKER_START_MONOTONIC
    room_name = req.room.name if req.room else "?"
    if age < _STALE_DISPATCH_DROP_WINDOW_S:
        logger.warning(
            "REJECT_STALE_DISPATCH job_id=%s room=%s age_s=%.2f "
            "(queued before worker start; refresh browser to retry)",
            req.id, room_name, age,
        )
        try:
            await req.reject()
        except Exception as exc:
            logger.error("REJECT_STALE_DISPATCH_FAILED err=%r", exc)
        return
    logger.info(
        "ACCEPT_DISPATCH job_id=%s room=%s age_s=%.2f",
        req.id, room_name, age,
    )
    await req.accept()


if __name__ == "__main__":
    _required_env("LIVEKIT_URL")
    _required_env("LIVEKIT_API_KEY")
    _required_env("LIVEKIT_API_SECRET")
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=_request_fnc,
            agent_name=os.getenv("AGENT_NAME", "hinglish-voice-agent"),
        )
    )
