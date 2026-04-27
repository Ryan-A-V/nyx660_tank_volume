"""
CLI — updated for continuous measurement loop.

New commands:
    run       — start continuous measurement (blocks until Ctrl+C)
    history   — print last N measurements from SQLite
    stats     — print loop and store stats
"""

from __future__ import annotations

import argparse
import json
import signal
import time

from nyx660_tank_volume.camera.factory import create_camera
from nyx660_tank_volume.config import load_config
from nyx660_tank_volume.core.calibration import CalibrationStore
from nyx660_tank_volume.core.service import TankVolumeService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local CLI for tank measurement"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "command",
        choices=["calibrate", "measure", "run", "history", "stats", "test"],
    )
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument(
        "--hours",
        type=float,
        default=24,
        help="Hours of history for 'history' command",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max records for 'history' command",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    service = TankVolumeService(
        cfg,
        create_camera(cfg.camera),
        CalibrationStore(cfg.storage.calibration_file),
    )
    service.start()

    try:
        if args.command == "calibrate":
            result = service.calibrate_empty_tank(frames=args.frames)
            print(json.dumps(result, indent=2))

        elif args.command == "measure":
            # Wait for at least one measurement from the loop
            print("Waiting for measurement...", flush=True)
            for _ in range(60):
                result = service.last_measurement
                if result is not None:
                    print(json.dumps(result.to_dict(), indent=2))
                    break
                time.sleep(0.5)
            else:
                print("No measurement available. Is calibration loaded?")

        elif args.command == "run":
            # Run continuously until Ctrl+C
            print("Continuous measurement running. Press Ctrl+C to stop.")
            print(f"Loop stats: {json.dumps(service.get_loop_stats(), indent=2)}")

            stop = False

            def handle_signal(sig, frame):
                nonlocal stop
                stop = True

            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)

            last_print = 0.0
            while not stop:
                time.sleep(1)
                now = time.monotonic()
                if now - last_print >= 5.0:
                    last_print = now
                    latest = service.last_measurement
                    stats = service.get_loop_stats()
                    if latest is not None:
                        print(
                            f"[{latest.timestamp_utc}] "
                            f"vol={latest.estimated_volume_m3:.3f} m³, "
                            f"fill={latest.relative_fill_ratio or 0:.1%}, "
                            f"quality={latest.valid_pixel_ratio:.1%}, "
                            f"fps={stats.get('fps', 0):.1f}"
                        )
                    else:
                        print(
                            f"Loop running: {stats.get('loop_count', 0)} cycles, "
                            f"no measurement yet (calibration loaded: "
                            f"{stats.get('has_calibration', False)})"
                        )

            print("\nStopping...")

        elif args.command == "history":
            records = service.get_measurement_history(
                hours=args.hours, limit=args.limit
            )
            print(json.dumps({"count": len(records), "measurements": records}, indent=2))

        elif args.command == "stats":
            print("Loop stats:")
            print(json.dumps(service.get_loop_stats(), indent=2))
            print("\nStore stats:")
            print(json.dumps(service.get_store_stats(), indent=2))

        elif args.command == "test":
            # Interactive testing mode — trigger measurements one at a time
            service.set_testing_mode(True)
            print("Testing mode active.")
            print("Press Enter to trigger a measurement, 'q' to quit.\n")

            measurement_num = 0
            while True:
                stats = service.get_loop_stats()
                if not stats.get("has_calibration", False):
                    print("No calibration loaded. Run 'calibrate' first.")
                    break

                user_input = input(
                    f"[{measurement_num}] Ready — press Enter to measure "
                    f"(or 'q' to quit): "
                )
                if user_input.strip().lower() == "q":
                    break

                accepted = service.trigger_next_measurement()
                if not accepted:
                    print("  Trigger not accepted — loop may still be processing.")
                    continue

                print("  Measuring...", end="", flush=True)
                result = service.wait_for_measurement_result(timeout=30.0)

                if result is None:
                    print(" timed out.")
                    continue

                measurement_num += 1
                print(" done.")
                print(f"  Volume:    {result.estimated_volume_m3:.4f} m³")
                if result.estimated_volume_liters is not None:
                    print(f"  Litres:    {result.estimated_volume_liters:.1f} L")
                if result.relative_fill_ratio is not None:
                    print(f"  Fill:      {result.relative_fill_ratio:.1%}")
                print(f"  Avg height: {result.average_fill_height_m:.3f} m")
                print(f"  Max height: {result.max_fill_height_m:.3f} m")
                print(f"  Quality:   {result.valid_pixel_ratio:.1%}")
                if result.notes:
                    for note in result.notes:
                        print(f"  Note: {note}")
                print()

    finally:
        service.stop()


if __name__ == "__main__":
    main()
