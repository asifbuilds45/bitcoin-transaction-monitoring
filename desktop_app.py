"""
NTRO Bitcoin Forensic Workstation - Desktop Application Launcher
Seamlessly runs the offline monitoring system inside a dedicated native desktop window.
"""

import os
import sys
import time
import socket
import subprocess
import urllib.request


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def get_available_port(default_port: int = 8501) -> int:
    if not is_port_in_use(default_port):
        return default_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(url: str, proc: subprocess.Popen, timeout_seconds: int = 35) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        # Check if process crashed
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status in (200, 304):
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def launch_in_app_mode(url: str) -> subprocess.Popen:
    """Launch in standalone frameless app mode using Edge or Chrome."""
    edge_paths = [
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        "msedge.exe"
    ]
    chrome_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        "chrome.exe"
    ]

    for exe in edge_paths + chrome_paths:
        try:
            return subprocess.Popen([
                exe,
                f"--app={url}",
                "--window-size=1440,900"
            ])
        except (FileNotFoundError, OSError):
            continue

    import webbrowser
    webbrowser.open(url)
    return None


def main():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    app_py_path = os.path.join(workspace_dir, "app.py")

    if not os.path.exists(app_py_path):
        print(f"[ERROR] Could not find app.py at {app_py_path}")
        sys.exit(1)

    port = 8501
    url = f"http://127.0.0.1:{port}"

    print("=" * 65)
    print("  NTRO BITCOIN TRANSACTION FORENSIC WORKSTATION")
    print("  Mode: Fully Offline Air-Gapped Desktop Application")
    print(f"  Target Endpoint: {url}")
    print("=" * 65)

    # Clean Streamlit CLI arguments
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_py_path,
        "--server.port", str(port),
        "--server.headless", "true"
    ]

    print("[1/2] Initializing offline analytics engine...")
    server_proc = subprocess.Popen(
        cmd,
        cwd=workspace_dir
    )

    try:
        print("[2/2] Waiting for local server to load pipeline...")
        ready = wait_for_server(url, server_proc, timeout_seconds=40)
        
        if not ready:
            if server_proc.poll() is not None:
                print(f"[ERROR] Analytics engine exited with code {server_proc.returncode}.")
            else:
                print("[ERROR] Analytics engine timed out while starting.")
            sys.exit(1)

        print("\nOpening Dedicated Desktop Application Window...")

        use_pywebview = False
        try:
            import webview
            use_pywebview = True
        except ImportError:
            use_pywebview = False

        if use_pywebview:
            window = webview.create_window(
                title="NTRO - AI Bitcoin Forensic Workstation [Offline Air-Gapped]",
                url=url,
                width=1440,
                height=920,
                min_size=(1100, 720),
                resizable=True
            )
            webview.start()
        else:
            ui_proc = launch_in_app_mode(url)
            if ui_proc:
                ui_proc.wait()
            else:
                input("\nPress ENTER to close the forensic workstation...")

    finally:
        print("\n[SHUTDOWN] Terminating background analytics engine...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("[SHUTDOWN] Application closed cleanly.")


if __name__ == "__main__":
    main()
