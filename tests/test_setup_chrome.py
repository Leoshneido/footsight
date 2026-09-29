import json
import os

from footsight import setup_chrome

# The public key in extension/manifest.json and the id Chrome derives from it
# (sha256 of the DER key, first 32 hex digits, 0-f mapped to a-p).
MANIFEST_ID = "kpjoeofameimecgoapeepcgbacpbkfai"


def test_the_extension_id_comes_from_the_manifest_key():
    key = json.loads((setup_chrome.EXTENSION_DIR / "manifest.json").read_text())["key"]
    assert setup_chrome.extension_id(key) == MANIFEST_ID


def test_extension_ids_use_chromes_a_to_p_alphabet():
    extension_id = setup_chrome.extension_id("AAAA")
    assert len(extension_id) == 32 and set(extension_id) <= set("abcdefghijklmnop")


def test_install_writes_a_launcher_and_registers_it_with_chrome(tmp_path):
    home, hosts = tmp_path / "home", tmp_path / "hosts"

    launcher, registration = setup_chrome.install(home, hosts, python="/venv/bin/python")

    assert launcher == home / "footsight-host" and os.access(launcher, os.X_OK)
    script = launcher.read_text()
    assert script.startswith("#!/bin/sh") and 'exec "/venv/bin/python" -m footsight.host' in script
    manifest = json.loads(registration.read_text())
    assert registration == hosts / "com.footsight.host.json"
    assert manifest == {
        "name": "com.footsight.host",
        "description": "footsight helper: starts and manages footsight projects for the Chrome extension",
        "path": str(launcher),
        "type": "stdio",
        "allowed_origins": [f"chrome-extension://{MANIFEST_ID}/"],
    }


def test_uninstall_removes_the_registration_and_launcher(tmp_path):
    home, hosts = tmp_path / "home", tmp_path / "hosts"
    launcher, registration = setup_chrome.install(home, hosts, python="/venv/bin/python")

    setup_chrome.uninstall(home, hosts)

    assert not launcher.exists() and not registration.exists()
