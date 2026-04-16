from __future__ import annotations

import argparse

import uvicorn

from nyx660_tank_volume.api.app import build_app
from nyx660_tank_volume.camera.factory import create_camera
from nyx660_tank_volume.config import load_config
from nyx660_tank_volume.core.calibration import CalibrationStore
from nyx660_tank_volume.core.service import TankVolumeService


def main() -> None:
    parser = argparse.ArgumentParser(description="NYX660 tank measurement server")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    camera = create_camera(cfg.camera)
    store = CalibrationStore(cfg.storage.calibration_file)
    service = TankVolumeService(cfg, camera, store)
    service.start()

    app = build_app(cfg, service)
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    main()
