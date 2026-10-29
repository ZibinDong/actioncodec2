"""Serve a policy over WebSocket.

    python scripts/serve.py --model <model directory or Hub repository ID>
"""

from actioncodec2.vla.runtime.server import main

if __name__ == "__main__":
    raise SystemExit(main())
