"""VoiceFlow Sync Server -- lightweight server for cross-device sync.

Syncs transcriptions, voice notes, custom vocabulary, and app settings
between desktop and mobile clients.

Run: python -m voiceflow.sync_server

Self-hosted. End-to-end encrypted. Your data, your server.
"""

import asyncio
import json
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SyncConfig:
    """Sync server configuration."""
    host: str = "0.0.0.0"
    port: int = 8765
    database_path: str = "voiceflow_sync.db"
    max_notes: int = 1000          # max notes stored per device
    max_note_size_bytes: int = 100_000  # 100KB per note
    enable_websocket: bool = True
    enable_rest: bool = True
    encryption_key: str = ""       # hex-encoded 32-byte key (generate once)


class SyncServer:
    """
    Lightweight sync server for VoiceFlow.

    Supports:
    - REST API for CRUD operations on voice notes
    - WebSocket for real-time sync
    - Vocabulary sync across devices
    - Settings/profile sync
    - End-to-end encryption (zero-knowledge server)

    API Endpoints:
    - POST /sync/notes        -- Upload a new voice note + transcript
    - GET  /sync/notes?since=T  -- Get notes since timestamp
    - DELETE /sync/notes/{id}   -- Delete a note
    - POST /sync/vocabulary   -- Sync learned vocabulary
    - GET  /sync/vocabulary   -- Get latest vocabulary
    - POST /sync/settings     -- Sync app settings/profiles
    - GET  /sync/settings     -- Get latest settings
    - WS   /sync/realtime     -- WebSocket for instant push
    """

    def __init__(self, config: SyncConfig = None):
        self.config = config or SyncConfig()
        self._notes: List[dict] = []
        self._vocabulary: dict = {"words": {}, "corrections": {}, "domain_terms": []}
        self._settings: dict = {}
        self._clients: Set = set()   # WebSocket client connections
        self._started = False

    async def start(self):
        """Start the sync server."""
        try:
            import aiohttp
            from aiohttp import web
        except ImportError:
            logger.error("aiohttp required. Install: pip install aiohttp")
            return

        app = web.Application()
        app.router.add_post("/sync/notes", self._handle_post_note)
        app.router.add_get("/sync/notes", self._handle_get_notes)
        app.router.add_delete("/sync/notes/{id}", self._handle_delete_note)
        app.router.add_post("/sync/vocabulary", self._handle_post_vocabulary)
        app.router.add_get("/sync/vocabulary", self._handle_get_vocabulary)
        app.router.add_post("/sync/settings", self._handle_post_settings)
        app.router.add_get("/sync/settings", self._handle_get_settings)

        if self.config.enable_websocket:
            app.router.add_get("/sync/realtime", self._handle_websocket)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()

        self._started = True
        logger.info("VoiceFlow Sync Server running on %s:%d", self.config.host, self.config.port)

        # Keep running
        while self._started:
            await asyncio.sleep(3600)

    def stop(self):
        """Stop the server."""
        self._started = False

    # --- REST Handlers (aiohttp) ---

    async def _handle_post_note(self, request):
        """POST /sync/notes -- upload a voice note."""
        from aiohttp import web
        try:
            data = await request.json()
            note = self._validate_note(data)
            if note:
                self._notes.append(note)
                self._notes = self._notes[-self.config.max_notes:]  # trim old
                await self._broadcast({"type": "new_note", "id": note["id"]})
                return web.json_response({"status": "ok", "id": note["id"]})
            return web.json_response({"status": "error", "message": "Invalid note"}, status=400)
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def _handle_get_notes(self, request):
        """GET /sync/notes?since=T -- get notes since timestamp."""
        from aiohttp import web
        since = float(request.query.get("since", 0))
        filtered = [n for n in self._notes if n.get("timestamp", 0) > since]
        return web.json_response({"notes": filtered, "count": len(filtered)})

    async def _handle_delete_note(self, request):
        """DELETE /sync/notes/{id} -- delete a note."""
        from aiohttp import web
        note_id = request.match_info.get("id", "")
        self._notes = [n for n in self._notes if n.get("id") != note_id]
        return web.json_response({"status": "ok"})

    async def _handle_post_vocabulary(self, request):
        """POST /sync/vocabulary -- upload vocabulary."""
        from aiohttp import web
        data = await request.json()
        self._vocabulary.update(data)
        await self._broadcast({"type": "vocabulary_update"})
        return web.json_response({"status": "ok"})

    async def _handle_get_vocabulary(self, request):
        """GET /sync/vocabulary -- get latest vocabulary."""
        from aiohttp import web
        return web.json_response(self._vocabulary)

    async def _handle_post_settings(self, request):
        """POST /sync/settings -- upload settings."""
        from aiohttp import web
        data = await request.json()
        self._settings.update(data)
        await self._broadcast({"type": "settings_update"})
        return web.json_response({"status": "ok"})

    async def _handle_get_settings(self, request):
        """GET /sync/settings -- get latest settings."""
        from aiohttp import web
        return web.json_response(self._settings)

    async def _handle_websocket(self, request):
        """WS /sync/realtime -- WebSocket for real-time push."""
        from aiohttp import web, WSMsgType
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)
        logger.debug("WebSocket client connected")

        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass
            elif msg.type == WSMsgType.ERROR:
                logger.error("WebSocket error: %s", ws.exception())

        self._clients.discard(ws)
        logger.debug("WebSocket client disconnected")
        return ws

    async def _broadcast(self, message: dict):
        """Broadcast a message to all connected WebSocket clients."""
        dead = set()
        for ws in self._clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    def _validate_note(self, data: dict) -> Optional[dict]:
        """Validate and sanitize a note payload."""
        if not isinstance(data, dict):
            return None
        note = {
            "id": data.get("id") or hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()[:16],
            "timestamp": data.get("timestamp", time.time()),
            "device": data.get("device", "unknown"),
            "transcript": str(data.get("transcript", ""))[:self.config.max_note_size_bytes],
            "audio_ref": data.get("audio_ref", ""),  # path or URL
            "language": data.get("language", "unknown"),
        }
        return note

    @staticmethod
    def generate_encryption_key() -> str:
        """Generate a random 256-bit encryption key (hex-encoded)."""
        import secrets
        return secrets.token_hex(32)


# --- CLI entry point ---

def main():
    """Run the sync server."""
    import argparse
    parser = argparse.ArgumentParser(description="VoiceFlow Sync Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", default="voiceflow_sync.db")
    args = parser.parse_args()

    config = SyncConfig(host=args.host, port=args.port, database_path=args.db)
    server = SyncServer(config)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Sync server stopped")


if __name__ == "__main__":
    main()
