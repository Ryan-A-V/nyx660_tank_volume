from __future__ import annotations

import argparse
import json

from nyx660_tank_volume.camera.factory import create_camera
from nyx660_tank_volume.config import load_config
from nyx660_tank_volume.core.calibration import CalibrationStore
from nyx660_tank_volume.core.service import TankVolumeService


def main() -> None:
    parser = argparse.ArgumentParser(description="Local CLI for NYX660 tank measurement")
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=["calibrate", "measure"])
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    service = TankVolumeService(cfg, create_camera(cfg.camera), CalibrationStore(cfg.storage.calibration_file))
    service.start()
    try:
        if args.command == "calibrate":
            print(json.dumps(service.calibrate_empty_tank(frames=args.frames), indent=2))
        else:
            print(json.dumps(service.measure().to_dict(), indent=2))
    finally:
        service.stop()


if __name__ == "__main__":
    main()
