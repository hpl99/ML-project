"""URsafe Application Server Launcher.

Starts the Uvicorn ASGI server and prints the application URL.
Supports --port and --host CLI options, and automatically detects port conflicts.
"""
import argparse
import socket
import sys
import uvicorn


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a network port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_free_port(start_port: int = 8000, host: str = "127.0.0.1") -> int:
    """Find the next available port if starting port is busy."""
    port = start_port
    while port < start_port + 20:
        if not is_port_in_use(port, host):
            return port
        port += 1
    return start_port


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="URsafe Web Application Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--no-auto-port", dest="auto_port", action="store_false", help="Disable automatic free port fallback")
    parser.set_defaults(auto_port=True)
    args = parser.parse_args()

    port = args.port
    host = args.host

    if is_port_in_use(port, host):
        free_port = find_free_port(port + 1, host)
        print("=" * 70)
        print(f"  [NOTICE] Port {port} is already in use by another running process.")
        if args.auto_port:
            print(f"  --> Automatically switching to available port: {free_port}")
            port = free_port
        else:
            print(f"  Active instance at: http://{host}:{port}")
            print(f"  Or start on another port: python run_app.py --port {free_port}")
            print("=" * 70)
            sys.exit(0)
        print("=" * 70)

    url = f"http://{host}:{port}"
    print("=" * 70)
    print("  URsafe: Terrain Hazard Prediction & AI Navigation Companion")
    print(f"  Server starting at: {url}")
    print(f"  API Documentation:  {url}/docs")
    print(f"  ADR Documentation:  {url}/adr")
    print("=" * 70)

    uvicorn.run("backend.main:app", host=host, port=port, reload=False, log_level="info")
