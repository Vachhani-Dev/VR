"""
Automatic Python Environment Resolver & Dependency Bootstrapper for AeroTouch 3D / AeroDrive.

Handles all common Windows / PowerShell / multi-Python setups automatically:
  1. Fast path (<0.1 ms): Returns immediately if required modules already exist in `sys.executable`.
  2. Auto-Discovery (100% Offline capable): Scans `.venv` / `venv` folders in the repo & parent
     directories, `pip`'s own Python interpreter, Conda environments, and Windows `py -0p`
     installations. If any interpreter already has the packages installed, re-launches cleanly!
  3. Self-Healing Install: Installs missing packages via `sys.executable -m pip install ...`, and
     if permissions or PEP-668 block system site-packages (e.g. `C:\\Program Files\\Python314`
     without Administrator rights), automatically creates `.venv` inside the project directory,
     installs dependencies there once, and launches seamlessly!
"""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, List, Sequence
import venv

PACKAGE_SPEC_MAP = {
    "cv2": "opencv-python>=4.8.0",
    "mediapipe": "mediapipe>=0.10.14",
    "pygame": "pygame-ce>=2.5.0",
    "OpenGL": "PyOpenGL>=3.1.7",
    "numpy": "numpy>=1.26.0",
}

ALL_PROJECT_SPECS = [
    "opencv-python>=4.8.0",
    "mediapipe>=0.10.14",
    "pygame-ce>=2.5.0",
    "PyOpenGL>=3.1.7",
    "numpy>=1.26.0",
]


def _missing_modules(modules: Sequence[str]) -> List[str]:
    return [m for m in modules if importlib.util.find_spec(m) is None]


def _venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _candidate_pythons() -> List[Path]:
    """Collect candidate Python executables across venvs, Conda, PATH & Windows `py` launcher."""
    candidates: List[Path] = []
    seen = {Path(sys.executable).resolve()}

    def _add(p: Path) -> None:
        try:
            if p.is_file():
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    candidates.append(p)
        except OSError:
            pass

    repo_dir = Path(__file__).resolve().parent
    search_roots = [Path.cwd(), repo_dir, *repo_dir.parents[:3], Path.home()]
    venv_names = (".venv", "venv", "env", ".env", "vr_env", "aerodrive_env")

    for root in search_roots:
        for vname in venv_names:
            _add(root / vname / "Scripts" / "python.exe")
            _add(root / vname / "bin" / "python")
    _add(Path("/tmp/aerodrive_venv/bin/python"))

    # Check if `pip` on PATH belongs to a different Python installation than `python`
    for pip_cmd in ("pip", "pip3"):
        pip_path = shutil.which(pip_cmd)
        if pip_path:
            pp = Path(pip_path)
            _add(pp.parent / "python.exe")
            _add(pp.parent / "python")
            _add(pp.parent.parent / "python.exe")

    # Windows Python Launcher (`py -0p`) enumerates every installed Python version on Windows
    if os.name == "nt" and shutil.which("py"):
        try:
            res = subprocess.run(
                ["py", "-0p"],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            )
            for line in res.stdout.splitlines():
                parts = line.strip().split()
                if parts:
                    _add(Path(parts[-1].strip('"')))
        except Exception:
            pass

    # Common Windows & Conda installation locations
    home = Path.home()
    local_app = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    for pattern_base, glob_pat in (
        (local_app / "Programs" / "Python", "Python*/python.exe"),
        (home, "anaconda3/python.exe"),
        (home, "miniconda3/python.exe"),
        (home, "anaconda3/envs/*/python.exe"),
        (home, "miniconda3/envs/*/python.exe"),
    ):
        if pattern_base.is_dir():
            for match in sorted(pattern_base.glob(glob_pat), reverse=True):
                _add(match)

    return candidates


def _interpreter_has_modules(py_exe: Path, modules: Iterable[str]) -> bool:
    check_stmt = (
        "import importlib.util, sys; "
        f"mods = {list(modules)!r}; "
        "sys.exit(0 if all(importlib.util.find_spec(m) is not None for m in mods) else 1)"
    )
    try:
        res = subprocess.run(
            [str(py_exe), "-c", check_stmt],
            capture_output=True,
            timeout=6,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def ensure_runtime_packages(required_modules: Sequence[str] = ("cv2", "mediapipe", "pygame", "numpy")) -> None:
    """Ensure required packages exist in the active runtime, auto-switching or installing if needed."""
    missing = _missing_modules(required_modules)
    if not missing:
        return

    # 1. Try existing virtual environments / other installed Pythons first (instant & 100% offline!)
    if os.environ.get("_AERODRIVE_REEXEC") != "1":
        for candidate in _candidate_pythons():
            if _interpreter_has_modules(candidate, required_modules):
                print(
                    f"[AeroDrive Auto-Env] Found working Python environment:\n"
                    f"  -> {candidate}\n"
                    f"[AeroDrive Auto-Env] Launching {Path(sys.argv[0]).name} automatically..."
                )
                env = os.environ.copy()
                env["_AERODRIVE_REEXEC"] = "1"
                sys.exit(subprocess.call([str(candidate), *sys.argv], env=env))

    # 2. Try installing directly into current `sys.executable` (works cleanly on user-level Windows Python)
    specs = ALL_PROJECT_SPECS
    print("=" * 72)
    print(f"[AeroDrive Setup] Missing packages in {sys.executable}: {', '.join(missing)}")
    print("[AeroDrive Setup] Installing required packages automatically (one-time setup)...")
    print("=" * 72)
    rc = subprocess.call([sys.executable, "-m", "pip", "install", *specs])
    if rc == 0 and not _missing_modules(required_modules):
        print("[AeroDrive Setup] Ready! Launching application...\n")
        return

    # 3. Fallback when system Python is read-only / Administrator-locked / PEP-668 managed:
    #    Create a persistent `.venv` in the repo (or `/tmp/aerodrive_venv`) and re-launch inside it!
    repo_dir = Path(__file__).resolve().parent
    fallback_venv = Path("/tmp/aerodrive_venv") if os.environ.get("_AERODRIVE_SANDBOX_TEST") == "1" else (repo_dir / ".venv")
    venv_py = _venv_python_path(fallback_venv)
    print(f"\n[AeroDrive Setup] Creating dedicated virtual environment at: {fallback_venv}")
    if not venv_py.is_file():
        venv.create(fallback_venv, with_pip=True)
    rc_venv = subprocess.call([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"])
    rc_pkgs = subprocess.call([str(venv_py), "-m", "pip", "install", *specs])
    if rc_venv == 0 and rc_pkgs == 0 and _interpreter_has_modules(venv_py, required_modules):
        print("[AeroDrive Setup] Virtual environment ready! Launching application...\n")
        env = os.environ.copy()
        env["_AERODRIVE_REEXEC"] = "1"
        sys.exit(subprocess.call([str(venv_py), *sys.argv], env=env))

    print("\n[ERROR] Automatic package setup could not finish.")
    print("Run this command in PowerShell to install manually:\n")
    print(f'  & "{sys.executable}" -m pip install opencv-python mediapipe pygame-ce PyOpenGL numpy\n')
    sys.exit(1)


# Run automatically as soon as `import runtime_bootstrap` executes at startup
_REQUIRED_AT_BOOT = (
    ("cv2", "mediapipe", "pygame", "OpenGL", "numpy")
    if "step2" in Path(sys.argv[0]).name
    else ("cv2", "mediapipe", "pygame", "numpy")
)
ensure_runtime_packages(_REQUIRED_AT_BOOT)



