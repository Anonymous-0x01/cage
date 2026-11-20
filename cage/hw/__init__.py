"""Hardware specs (nominal + calibrated) for the cheap cost model."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

HW_DIR = Path(__file__).resolve().parent
SPEC_FILE = HW_DIR / "h20.yaml"
CALIB_FILE = HW_DIR / "calibration.json"


def device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def load_hw(name: str | None = None) -> dict:
    """Return the spec dict for ``name`` (default: current device), merged with calibration."""
    name = name or device_name()
    devices = yaml.safe_load(SPEC_FILE.read_text())["devices"]
    chosen_key, chosen = "generic", devices["generic"]
    for key, spec in devices.items():
        if any(m.lower() in name.lower() for m in spec.get("match", [])):
            chosen_key, chosen = key, spec
            break
    hw = {k: v for k, v in chosen.items() if k != "match"}
    hw["name"] = chosen_key
    hw["device"] = name
    hw["calibrated"] = False
    if CALIB_FILE.exists():
        calib = json.loads(CALIB_FILE.read_text()).get(name)
        if calib:
            hw.update(calib)
            hw["calibrated"] = True
    return hw


def save_calibration(name: str, values: dict) -> None:
    data = json.loads(CALIB_FILE.read_text()) if CALIB_FILE.exists() else {}
    data[name] = values
    CALIB_FILE.write_text(json.dumps(data, indent=2))
