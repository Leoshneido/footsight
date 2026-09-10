# footsight/pitch_calibration.py
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PNLCALIB_VENDOR_DIR = REPO_ROOT / "vendor" / "PnLCalib"
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "pnlcalib_infer.py"
DEFAULT_WEIGHTS_KP = REPO_ROOT / "weights" / "pnlcalib" / "SV_kp"
DEFAULT_WEIGHTS_LINE = REPO_ROOT / "weights" / "pnlcalib" / "SV_lines"


def compute_homography(
    image_path: str,
    weights_kp: str = str(DEFAULT_WEIGHTS_KP),
    weights_line: str = str(DEFAULT_WEIGHTS_LINE),
    device: str = "cpu",
) -> np.ndarray | None:
    """Run PnLCalib in an isolated subprocess and return a 3x3 image->pitch homography.

    Runs out-of-process, never imported: PnLCalib is GPL-2.0 licensed. See
    scripts/pnlcalib_infer.py and MEMORY.md, 2026-09-10.
    """
    result = subprocess.run(
        [
            sys.executable, str(RUNNER_SCRIPT),
            "--image", str(Path(image_path).resolve()),
            "--weights-kp", str(Path(weights_kp).resolve()),
            "--weights-line", str(Path(weights_line).resolve()),
            "--device", device,
        ],
        cwd=PNLCALIB_VENDOR_DIR,
        env={**os.environ, "PYTHONPATH": str(PNLCALIB_VENDOR_DIR)},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pnlcalib_infer.py failed (exit {result.returncode}):\n{result.stderr}"
        )
    data = json.loads(result.stdout)
    if data["homography"] is None:
        return None
    return np.array(data["homography"])
