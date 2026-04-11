"""actuators/spotify_tools.py — Spotify playback control for Jarvis Phase 4.

Authentication
--------------
Uses Spotipy's SpotifyOAuth (PKCE) flow. On first launch, a browser tab
opens for the one-time OAuth consent. The token is cached at project root
in ``.spotify_cache`` (gitignored).

Required env vars (add to .env):
    SPOTIFY_CLIENT_ID=<your client id>
    SPOTIFY_CLIENT_SECRET=<your client secret>
    SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
    SPOTIFY_SCOPE=user-read-playback-state user-modify-playback-state user-read-currently-playing

All Spotipy calls are blocking → wrapped in asyncio.to_thread.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from app.actuators.registry import tool

logger = logging.getLogger(__name__)

_SPOTIFY_SCOPE = os.getenv(
    "SPOTIFY_SCOPE",
    "user-read-playback-state user-modify-playback-state user-read-currently-playing",
)
_CACHE_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / ".spotify_cache")
_VERIFY_TIMEOUT_SECONDS = 2.0
_VERIFY_POLL_INTERVAL = 0.4


def _get_spotify() -> Any:
    """Return an authenticated Spotipy client, or raise RuntimeError if unconfigured."""
    try:
        import spotipy  # type: ignore[import]
        from spotipy.oauth2 import SpotifyOAuth  # type: ignore[import]
    except ImportError as err:
        raise RuntimeError("spotipy is not installed. Run: pip install spotipy") from err

    client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback").strip()

    if not client_id or not client_secret:
        raise RuntimeError(
            "Spotify credentials are not configured. "
            "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your .env file."
        )

    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=_SPOTIFY_SCOPE,
        cache_path=_CACHE_PATH,
        open_browser=True,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def _safe_spotify_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Execute a Spotipy call and translate auth/connection errors to RuntimeError."""
    try:
        return fn(*args, **kwargs)
    except Exception as err:  # noqa: BLE001
        error_str = str(err)
        if "token" in error_str.lower() or "auth" in error_str.lower():
            raise RuntimeError(
                "Spotify authentication failed. Please re-run the OAuth flow "
                "or check your credentials in .env."
            ) from err
        if "no active device" in error_str.lower():
            raise RuntimeError(
                "No active Spotify device found. Open Spotify on any device first."
            ) from err
        raise RuntimeError(f"Spotify error: {err}") from err


def _read_playback_state(sp: Any) -> dict[str, Any]:
    current = _safe_spotify_call(sp.current_playback)
    if current is None:
        return {"playing": False, "device": None, "track": None, "volume_percent": None}

    item = current.get("item") or {}
    device = current.get("device") or {}
    return {
        "playing": bool(current.get("is_playing", False)),
        "device": device.get("name"),
        "device_id": device.get("id"),
        "track": item.get("name"),
        "track_id": item.get("id"),
        "volume_percent": device.get("volume_percent"),
    }


def _verify_playback_state(
    sp: Any,
    *,
    expect_playing: bool | None = None,
    expect_volume: int | None = None,
    expect_device_id: str | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    deadline = time.monotonic() + _VERIFY_TIMEOUT_SECONDS
    last_state: dict[str, Any] = {}

    while time.monotonic() < deadline:
        last_state = _read_playback_state(sp)

        checks: list[bool] = []
        if expect_playing is not None:
            checks.append(bool(last_state.get("playing")) == expect_playing)
        if expect_volume is not None:
            actual = last_state.get("volume_percent")
            checks.append(isinstance(actual, int) and abs(actual - expect_volume) <= 2)
        if expect_device_id is not None:
            checks.append(str(last_state.get("device_id") or "") == expect_device_id)

        if checks and all(checks):
            return True, "Spotify playback state matches the requested action.", last_state

        # Polling is intentionally short to keep command latency low.
        time_left = deadline - time.monotonic()
        if time_left > 0:
            sleep_for = _VERIFY_POLL_INTERVAL if time_left > _VERIFY_POLL_INTERVAL else time_left
            if sleep_for > 0:
                time.sleep(sleep_for)

    return False, "Spotify API did not confirm the requested playback state in time.", last_state


# ── play_spotify ──────────────────────────────────────────────────────────────

@tool(
    name="play_spotify",
    description="Resume Spotify playback on the active device.",
    parameters={"type": "object", "properties": {}, "required": []},
    risk_level="low",
    category="spotify",
    timeout_seconds=10,
)
async def play_spotify() -> dict[str, Any]:
    """Resume playback."""
    sp = await asyncio.to_thread(_get_spotify)
    await asyncio.to_thread(_safe_spotify_call, sp.start_playback)
    verified, evidence, state = await asyncio.to_thread(_verify_playback_state, sp, expect_playing=True)
    return {
        "status": "playing" if verified else "unverified",
        "verified": verified,
        "evidence": evidence,
        "playback": state,
    }


# ── pause_spotify ─────────────────────────────────────────────────────────────

@tool(
    name="pause_spotify",
    description="Pause Spotify playback.",
    parameters={"type": "object", "properties": {}, "required": []},
    risk_level="low",
    category="spotify",
    timeout_seconds=10,
)
async def pause_spotify() -> dict[str, Any]:
    """Pause playback."""
    sp = await asyncio.to_thread(_get_spotify)
    await asyncio.to_thread(_safe_spotify_call, sp.pause_playback)
    verified, evidence, state = await asyncio.to_thread(_verify_playback_state, sp, expect_playing=False)
    return {
        "status": "paused" if verified else "unverified",
        "verified": verified,
        "evidence": evidence,
        "playback": state,
    }


# ── next_track ────────────────────────────────────────────────────────────────

@tool(
    name="next_track",
    description="Skip to the next track on Spotify.",
    parameters={"type": "object", "properties": {}, "required": []},
    risk_level="low",
    category="spotify",
    timeout_seconds=10,
)
async def next_track() -> dict[str, Any]:
    """Skip to next track."""
    sp = await asyncio.to_thread(_get_spotify)
    before = await asyncio.to_thread(_read_playback_state, sp)
    await asyncio.to_thread(_safe_spotify_call, sp.next_track)
    await asyncio.sleep(0.6)
    after = await asyncio.to_thread(_read_playback_state, sp)
    verified = bool(after.get("track_id")) and after.get("track_id") != before.get("track_id")
    return {
        "status": "skipped_to_next" if verified else "unverified",
        "verified": verified,
        "evidence": (
            "Track id changed after next-track request."
            if verified
            else "Spotify API did not report a track change after next-track request."
        ),
        "before": before,
        "after": after,
    }


# ── previous_track ────────────────────────────────────────────────────────────

@tool(
    name="previous_track",
    description="Go back to the previous track on Spotify.",
    parameters={"type": "object", "properties": {}, "required": []},
    risk_level="low",
    category="spotify",
    timeout_seconds=10,
)
async def previous_track() -> dict[str, Any]:
    """Go to previous track."""
    sp = await asyncio.to_thread(_get_spotify)
    before = await asyncio.to_thread(_read_playback_state, sp)
    await asyncio.to_thread(_safe_spotify_call, sp.previous_track)
    await asyncio.sleep(0.6)
    after = await asyncio.to_thread(_read_playback_state, sp)
    verified = bool(after.get("track_id")) and after.get("track_id") != before.get("track_id")
    return {
        "status": "skipped_to_previous" if verified else "unverified",
        "verified": verified,
        "evidence": (
            "Track id changed after previous-track request."
            if verified
            else "Spotify API did not report a track change after previous-track request."
        ),
        "before": before,
        "after": after,
    }


# ── set_spotify_volume ────────────────────────────────────────────────────────

@tool(
    name="set_spotify_volume",
    description="Set Spotify playback volume to a percentage between 0 and 100.",
    parameters={
        "type": "object",
        "properties": {
            "level": {
                "type": "integer",
                "description": "Volume level from 0 to 100.",
            }
        },
        "required": ["level"],
    },
    risk_level="low",
    category="spotify",
    timeout_seconds=10,
)
async def set_spotify_volume(level: int) -> dict[str, Any]:
    """Set Spotify volume."""
    clamped = max(0, min(100, level))

    sp = await asyncio.to_thread(_get_spotify)
    await asyncio.to_thread(_safe_spotify_call, sp.volume, clamped)
    verified, evidence, state = await asyncio.to_thread(_verify_playback_state, sp, expect_volume=clamped)
    return {
        "status": "ok" if verified else "unverified",
        "volume_percent": clamped,
        "verified": verified,
        "evidence": evidence,
        "playback": state,
    }


# ── get_now_playing ───────────────────────────────────────────────────────────

@tool(
    name="get_now_playing",
    description=(
        "Get the currently playing track on Spotify: title, artist, album, "
        "and playback progress."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    risk_level="low",
    category="spotify",
    timeout_seconds=10,
)
async def get_now_playing() -> dict[str, Any]:
    """Return the currently playing track."""
    def _get() -> dict[str, Any]:
        sp = _get_spotify()
        current = _safe_spotify_call(sp.current_playback)
        if current is None or current.get("item") is None:
            return {"playing": False, "message": "Nothing is currently playing on Spotify."}

        item = current["item"]
        artists = ", ".join(a["name"] for a in item.get("artists", []))
        progress_ms = current.get("progress_ms", 0)
        duration_ms = item.get("duration_ms", 1)
        progress_pct = round((progress_ms / duration_ms) * 100, 1) if duration_ms else 0

        return {
            "playing": current.get("is_playing", False),
            "track": item.get("name", ""),
            "artist": artists,
            "album": item.get("album", {}).get("name", ""),
            "duration_ms": duration_ms,
            "progress_ms": progress_ms,
            "progress_percent": progress_pct,
            "device": current.get("device", {}).get("name", ""),
        }

    return await asyncio.to_thread(_get)


# ── search_and_play ───────────────────────────────────────────────────────────

@tool(
    name="search_and_play",
    description=(
        "Search Spotify for a track, artist, or playlist and immediately start playing "
        "the top result."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, e.g. 'Blinding Lights', 'The Weeknd', 'chill lofi playlist'.",
            },
            "search_type": {
                "type": "string",
                "description": "Type to search: 'track' (default), 'artist', 'playlist', or 'album'.",
            },
        },
        "required": ["query"],
    },
    risk_level="low",
    category="spotify",
    timeout_seconds=15,
)
async def search_and_play(query: str, search_type: str = "track") -> dict[str, Any]:
    """Search and play the top Spotify result."""
    valid_types = {"track", "artist", "playlist", "album"}
    stype = search_type.strip().lower() if search_type.strip().lower() in valid_types else "track"

    def _search_play(sp: Any) -> dict[str, Any]:
        results = _safe_spotify_call(sp.search, q=query, type=stype, limit=1)

        key = f"{stype}s"
        items = results.get(key, {}).get("items", [])
        if not items:
            return {"status": "not_found", "query": query, "error": f"No {stype} found for '{query}'."}

        top = items[0]
        uri = top.get("uri", "")
        name = top.get("name", query)

        if stype == "track":
            _safe_spotify_call(sp.start_playback, uris=[uri])
            artists = ", ".join(a["name"] for a in top.get("artists", []))
            return {"status": "playing", "track": name, "artist": artists, "uri": uri}
        elif stype == "artist":
            # Play the artist's top tracks
            top_tracks = _safe_spotify_call(sp.artist_top_tracks, top["id"])
            track_uris = [t["uri"] for t in top_tracks.get("tracks", [])[:10]]
            if not track_uris:
                return {"status": "not_found", "query": query, "error": "Artist has no top tracks."}
            _safe_spotify_call(sp.start_playback, uris=track_uris)
            return {"status": "playing", "artist": name, "tracks_queued": len(track_uris)}
        else:
            # Playlist or Album — play by context URI
            _safe_spotify_call(sp.start_playback, context_uri=uri)
            return {"status": "playing", "type": stype, "name": name, "uri": uri}

    sp = await asyncio.to_thread(_get_spotify)
    result = await asyncio.to_thread(_search_play, sp)
    if result.get("status") != "playing":
        return result

    verified, evidence, state = await asyncio.to_thread(_verify_playback_state, sp, expect_playing=True)
    result["verified"] = verified
    result["evidence"] = evidence
    result["playback"] = state
    if not verified:
        result["status"] = "unverified"
    return result


# ── get_spotify_devices ───────────────────────────────────────────────────────

@tool(
    name="get_spotify_devices",
    description="List all available Spotify Connect devices (PC, phone, speaker, etc.).",
    parameters={"type": "object", "properties": {}, "required": []},
    risk_level="low",
    category="spotify",
    timeout_seconds=10,
)
async def get_spotify_devices() -> dict[str, Any]:
    """List available Spotify devices."""
    def _get() -> dict[str, Any]:
        sp = _get_spotify()
        result = _safe_spotify_call(sp.devices)
        devices = [
            {
                "id": d.get("id"),
                "name": d.get("name"),
                "type": d.get("type"),
                "is_active": d.get("is_active"),
                "volume_percent": d.get("volume_percent"),
            }
            for d in result.get("devices", [])
        ]
        return {"devices": devices, "count": len(devices)}

    return await asyncio.to_thread(_get)


# ── transfer_playback ─────────────────────────────────────────────────────────

@tool(
    name="transfer_playback",
    description="Transfer Spotify playback to a specific device by name.",
    parameters={
        "type": "object",
        "properties": {
            "device_name": {
                "type": "string",
                "description": "The name of the target Spotify device.",
            }
        },
        "required": ["device_name"],
    },
    risk_level="low",
    category="spotify",
    timeout_seconds=10,
)
async def transfer_playback(device_name: str) -> dict[str, Any]:
    """Transfer Spotify playback to a named device."""
    def _transfer(sp: Any) -> dict[str, Any]:
        result = _safe_spotify_call(sp.devices)
        devices = result.get("devices", [])

        match = next(
            (d for d in devices if device_name.lower() in d.get("name", "").lower()),
            None,
        )
        if match is None:
            available = [d["name"] for d in devices]
            return {
                "status": "not_found",
                "device_name": device_name,
                "available_devices": available,
                "error": f"No device named '{device_name}' found.",
            }

        _safe_spotify_call(sp.transfer_playback, device_id=match["id"], force_play=True)
        return {"status": "transferred", "device_name": match["name"], "device_id": match["id"]}

    sp = await asyncio.to_thread(_get_spotify)
    result = await asyncio.to_thread(_transfer, sp)
    if result.get("status") != "transferred":
        return result

    verified, evidence, state = await asyncio.to_thread(
        _verify_playback_state,
        sp,
        expect_device_id=str(result.get("device_id") or ""),
    )
    result["verified"] = verified
    result["evidence"] = evidence
    result["playback"] = state
    if not verified:
        result["status"] = "unverified"
    return result
