"""One-time setup: register the footsight helper with Chrome so the
extension can start projects without Terminal.

    python -m footsight.setup_chrome            # install
    python -m footsight.setup_chrome --uninstall

Writes ~/.footsight/footsight-host (a launcher that runs this Python with
`-m footsight.host`) and Chrome's native-messaging registration, which only
trusts footsight's own extension (its id is fixed by the key in
extension/manifest.json).
"""
import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

from footsight import projects

HOST_NAME = "com.footsight.host"
EXTENSION_DIR = Path(__file__).resolve().parent.parent / "extension"
CHROME_HOSTS_DIR = Path.home() / "Library" / "Application Support" / "Google" / "Chrome" / "NativeMessagingHosts"


def extension_id(public_key_b64: str) -> str:
    """Chrome's id for an extension with this manifest key: the SHA-256 of
    the DER public key, first 32 hex digits, each mapped 0-f to a-p."""
    digest = hashlib.sha256(base64.b64decode(public_key_b64)).hexdigest()[:32]
    return "".join(chr(ord("a") + int(c, 16)) for c in digest)


def _manifest_extension_id() -> str:
    return extension_id(json.loads((EXTENSION_DIR / "manifest.json").read_text())["key"])


def install(home: Path, hosts_dir: Path, python: str) -> tuple[Path, Path]:
    home.mkdir(parents=True, exist_ok=True)
    launcher = home / "footsight-host"
    # Chrome starts the helper without a shell profile, so the launcher names
    # the exact Python (the project's venv, where footsight is installed).
    launcher.write_text(f'#!/bin/sh\nexec "{python}" -m footsight.host "$@"\n')
    launcher.chmod(0o755)

    hosts_dir.mkdir(parents=True, exist_ok=True)
    registration = hosts_dir / f"{HOST_NAME}.json"
    registration.write_text(json.dumps({
        "name": HOST_NAME,
        "description": "footsight helper: starts and manages footsight projects for the Chrome extension",
        "path": str(launcher),
        "type": "stdio",
        "allowed_origins": [f"chrome-extension://{_manifest_extension_id()}/"],
    }, indent=1))
    return launcher, registration


def uninstall(home: Path, hosts_dir: Path) -> None:
    for path in (hosts_dir / f"{HOST_NAME}.json", home / "footsight-host"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the footsight helper with Chrome.")
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()

    home = projects.state_dir()
    if args.uninstall:
        uninstall(home, CHROME_HOSTS_DIR)
        print("Removed footsight's registration with Chrome.")
        return
    launcher, registration = install(home, CHROME_HOSTS_DIR, sys.executable)
    print(f"Registered the footsight helper with Chrome:\n  {registration}\n  launcher: {launcher}")
    print(f"Extension id: {_manifest_extension_id()}")
    print("Next: open chrome://extensions and click the reload arrow on footsight "
          "(or Load unpacked -> the extension/ folder). Then click the footsight icon.")


if __name__ == "__main__":
    main()
