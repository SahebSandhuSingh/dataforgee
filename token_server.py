"""
Lightweight token server for the Say That Sound web client.

Provides two things:
  1. GET /token?room=<room>&identity=<identity>  → LiveKit JWT access token
  2. Static file serving for the web/ directory

Usage:
    python token_server.py
"""

import os

from aiohttp import web
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

from config import TOKEN_SERVER_PORT

load_dotenv()

LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


async def handle_token(request: web.Request) -> web.Response:
    """Generate a LiveKit access token for the web client."""
    room = request.query.get("room", "say-that-sound")
    identity = request.query.get("identity", "web-user")

    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        return web.json_response(
            {"error": "LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set"},
            status=500,
        )

    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )

    jwt_token = token.to_jwt()

    return web.json_response(
        {
            "token": jwt_token,
            "url": LIVEKIT_URL,
            "room": room,
            "identity": identity,
        }
    )


async def handle_index(request: web.Request) -> web.FileResponse:
    """Serve the main index.html."""
    return web.FileResponse(os.path.join(WEB_DIR, "index.html"))


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/token", handle_token)
    app.router.add_get("/", handle_index)
    app.router.add_static("/static/", WEB_DIR, show_index=False)
    return app


if __name__ == "__main__":
    print(f"Token server starting on http://localhost:{TOKEN_SERVER_PORT}")
    print(f"  → Web client: http://localhost:{TOKEN_SERVER_PORT}/")
    print(f"  → Token endpoint: http://localhost:{TOKEN_SERVER_PORT}/token")
    print(f"  → LiveKit URL: {LIVEKIT_URL or '(not set)'}")
    web.run_app(create_app(), port=TOKEN_SERVER_PORT)
