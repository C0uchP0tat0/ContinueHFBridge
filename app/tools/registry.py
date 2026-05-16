"""Server-side tool implementations (optional — Continue runs tools in IDE)."""
import os
import subprocess


def create_file(path: str, content: str = ""):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "ok", "action": "create_file", "path": path}


def read_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return {"status": "ok", "content": f.read()}


def edit_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"status": "ok", "action": "edit_file", "path": path}


def run_shell(cmd: str):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return {"stdout": result.stdout, "stderr": result.stderr, "code": result.returncode}


TOOLS = {
    "create_file": create_file,
    "read_file": read_file,
    "edit_file": edit_file,
    "run_shell": run_shell,
}
