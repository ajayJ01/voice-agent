import hashlib
import hmac
import logging
import os
import re
from datetime import timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
from livekit.api.twirp_client import TwirpError

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [token-server] %(message)s",
)
logger = logging.getLogger("token-server")

app = FastAPI(title="LiveKit Token Server", version="1.0.0")

_IDENTITY_RE = re.compile(r"^[-a-zA-Z0-9_@.]{1,128}$")
_ROOM_RE = re.compile(r"^[-a-zA-Z0-9_]{1,128}$")


def _deployment() -> str:
    return (os.getenv("DEPLOYMENT") or "").strip().lower()


def _cors_origins() -> list[str]:
    raw = (os.getenv("TOKEN_SERVER_CORS_ORIGINS") or "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    if _deployment() in {"production", "prod"}:
        raise RuntimeError(
            "DEPLOYMENT=production requires TOKEN_SERVER_CORS_ORIGINS "
            "(comma-separated browser origins, e.g. https://app.example.com)"
        )
    logger.warning("TOKEN_SERVER_CORS_ORIGINS unset; allowing all origins (dev only)")
    return ["*"]


def _verify_token_api_key(request: Request) -> None:
    expected = (os.getenv("TOKEN_SERVER_API_KEY") or "").strip()
    if not expected:
        return
    got = (request.headers.get("X-API-Key") or request.query_params.get("api_key") or "").strip()
    if not hmac.compare_digest(
        hashlib.sha256(expected.encode("utf-8")).digest(),
        hashlib.sha256(got.encode("utf-8")).digest(),
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise HTTPException(status_code=500, detail=f"Missing env var: {name}")
    return value


_cors = _cors_origins()
# Browsers disallow credentials with wildcard origin; dev uses * without cookies.
_cors_allow_credentials = _cors != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["GET", "HEAD"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "livekit-token-server"}


@app.get("/token")
async def create_token(
    request: Request,
    identity: str = Query("web-user", max_length=128),
    room: str = Query("voice-room", max_length=128),
) -> dict[str, str]:
    _verify_token_api_key(request)

    if not _IDENTITY_RE.match(identity):
        raise HTTPException(
            status_code=400,
            detail="identity must be 1–128 chars: letters, digits, ._-@",
        )
    if not _ROOM_RE.match(room):
        raise HTTPException(
            status_code=400,
            detail="room must be 1–128 chars: letters, digits, underscore, hyphen",
        )

    livekit_url = _required_env("LIVEKIT_URL")
    api_key = _required_env("LIVEKIT_API_KEY")
    api_secret = _required_env("LIVEKIT_API_SECRET")
    agent_name = os.getenv("AGENT_NAME", "hinglish-voice-agent")

    empty_timeout = int(os.getenv("LIVEKIT_ROOM_EMPTY_TIMEOUT_S", "300"))
    max_participants = int(os.getenv("LIVEKIT_ROOM_MAX_PARTICIPANTS", "4"))

    async with api.LiveKitAPI(livekit_url, api_key, api_secret) as lkapi:
        try:
            rooms = await lkapi.room.list_rooms(api.ListRoomsRequest(names=[room]))
            if not rooms.rooms:
                await lkapi.room.create_room(
                    api.CreateRoomRequest(
                        name=room,
                        empty_timeout=empty_timeout,
                        max_participants=max_participants,
                    )
                )
        except TwirpError as e:
            if "already_exists" not in str(e.code) and "already exists" not in str(e.message):
                raise HTTPException(status_code=500, detail=str(e)) from e

        # Kick ALL stale participants (browsers AND agents) that share this
        # room. If we leave a stale agent attached, we end up with TWO agents
        # in the same room when a new dispatch creates a fresh worker — the
        # old agent's RoomIO captures the new browser's audio while the new
        # agent's STT idles, producing the classic "speech detected, zero
        # transcripts, 60 seconds of speech_ms" symptom in agent logs. Force
        # a clean room so the new dispatch (created below) gives us exactly
        # one agent + one browser.
        try:
            existing_participants = await lkapi.room.list_participants(
                api.ListParticipantsRequest(room=room)
            )
            for p in existing_participants.participants:
                p_identity = p.identity or ""
                if p_identity == identity:
                    continue
                try:
                    await lkapi.room.remove_participant(
                        api.RoomParticipantIdentity(room=room, identity=p_identity)
                    )
                    kind = "browser" if p_identity.startswith("web-") else "agent"
                    logger.info(
                        "kicked stale %s participant=%s room=%s",
                        kind, p_identity, room,
                    )
                except TwirpError as kick_err:
                    logger.warning(
                        "kick failed participant=%s err=%s",
                        p_identity, kick_err,
                    )
        except TwirpError as e:
            logger.warning("list_participants failed room=%s err=%s", room, e)

        # Always tear down stale dispatches and create a fresh one. A dispatch
        # is "consumed" by a worker when it joins a room. If that worker dies
        # (e.g. agent.py was Ctrl+C'd and restarted), the dispatch record may
        # still be in LiveKit's database, attached to the dead worker. The new
        # worker that just registered will NOT receive a fresh job request
        # unless we explicitly create a new dispatch — and creating without
        # cleanup can leave duplicates. So: delete all existing dispatches for
        # this (room, agent_name) pair, then create exactly one fresh dispatch.
        try:
            existing = await lkapi.agent_dispatch.list_dispatch(room)
            for d in existing:
                if d.agent_name == agent_name:
                    try:
                        await lkapi.agent_dispatch.delete_dispatch(d.id, room)
                        logger.info(
                            "deleted stale dispatch id=%s agent=%s room=%s",
                            d.id, agent_name, room,
                        )
                    except TwirpError as del_err:
                        logger.warning(
                            "delete_dispatch failed id=%s err=%s", d.id, del_err,
                        )
        except TwirpError as e:
            if "not_found" not in str(e.code) and "does not exist" not in str(e.message):
                logger.warning("list_dispatch failed room=%s err=%s", room, e)

        try:
            await lkapi.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=agent_name,
                    room=room,
                    metadata=f"dispatch:{identity}",
                )
            )
            logger.info("created dispatch agent=%s room=%s identity=%s",
                        agent_name, room, identity)
        except TwirpError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    ttl_hours = int(os.getenv("LIVEKIT_TOKEN_TTL_HOURS", "2"))
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_ttl(timedelta(hours=ttl_hours))
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    return {"url": livekit_url, "token": token, "room": room, "identity": identity}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("TOKEN_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("TOKEN_SERVER_PORT", "8787"))
    uvicorn.run(app, host=host, port=port)
