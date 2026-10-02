"""Bounded GPU/Isaac Lab preflight; no environment installation or robot training."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new preflight output directory")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting", "python": sys.version, "executable": sys.executable}
    app = None
    started = time.monotonic()
    try:
        launcher = AppLauncher(args)
        app = launcher.app
        import torch
        import isaaclab.sim as sim_utils
        report["versions"] = {name: importlib.metadata.version(name) for name in
                              ("torch", "isaacsim", "isaaclab", "rsl-rl-lib")}
        torch.cuda.set_device(args.device)
        tensor = torch.eye(16, device=args.device)
        torch.testing.assert_close(tensor @ tensor, tensor)
        simulation = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device=args.device))
        simulation.reset()
        for _ in range(10):
            simulation.step(render=False)
        torch.cuda.synchronize()
        report.update(status="passed", gpu=torch.cuda.get_device_name(args.device),
                      cuda_available=torch.cuda.is_available(), physics_steps=10,
                      elapsed_seconds=time.monotonic() - started)
        print("TRON1_TRAINING_PREFLIGHT_PASSED", json.dumps(report), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
