"""URsafe Application Server Launcher.

Starts the Uvicorn ASGI server and prints the application URL.
"""
import sys
import uvicorn

if __name__ == "__main__":
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    print("=" * 70)
    print("  URsafe: Terrain Hazard Prediction & AI Navigation Companion")
    print(f"  Server starting at: {url}")
    print(f"  API Documentation:  {url}/docs")
    print(f"  ADR Documentation:  {url}/adr")
    print("=" * 70)
    uvicorn.run("backend.main:app", host=host, port=port, reload=False, log_level="info")
