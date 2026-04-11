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
from pathlib import Path
from typing import Any

from app.actuators.registry import tool

logger = logging.getLogger(__name__)

_SPOTIFY_SCOPE = os.getenv(
    "SPOTIFY_SCOPE",
    "user-read-playback-state user-modify-playback-state user-read-currently-playing",
)
_CACHE_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / ".spotify_cache")


def _get_spotify() -> Any:
    """Return an authenticated Spotipy client, or raise RuntimeError if unconfigured."""
    try:
        import spotipy  # type: ignore[import]
        from spotipy.oauth2 import SpotifyOAuth  # type: ignore[import]
    except ImportError as err:
        raise RuntimeError("spotipy is not installed. Run: pip install spotipy") from err

    client_id = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback").strip()

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
    def _play() -> dict[str, Any]:
        sp = _get_spotify()
        _safe_spotify_call(sp.start_playback)
        return {"status": "playing"}

    return await asyncio.to_thread(_play)


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
    def _pause() -> dict[str, Any]:
        sp = _get_spotify()
        _safe_spotify_call(sp.pause_playback)
        return {"status": "paused"}

    return await asyncio.to_thread(_pause)


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
    def _next() -> dict[str, Any]:
        sp = _get_spotify()
        _safe_spotify_call(sp.next_track)
        return {"status": "skipped_to_next"}

    return await asyncio.to_thread(_next)


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
    def _prev() -> dict[str, Any]:
        sp = _get_spotify()
        _safe_spotify_call(sp.previous_track)
        return {"status": "skipped_to_previous"}

    return await asyncio.to_thread(_prev)


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

    def _set_vol() -> dict[str, Any]:
        sp = _get_spotify()
        _safe_spotify_call(sp.volume, clamped)
        return {"status": "ok", "volume_percent": clamped}

    return await asyncio.to_thread(_set_vol)


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

    def _search_play() -> dict[str, Any]:
        sp = _get_spotify()
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

    return await asyncio.to_thread(_search_play)


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
    def _transfer() -> dict[str, Any]:
        sp = _get_spotify()
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

    return await asyncio.to_thread(_transfer)
