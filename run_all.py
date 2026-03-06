"""One-command project runner for training, validation, and optional UI launch."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "admet_multitask.csv"
REQUIRED_ENDPOINT_DATA = ["dili.csv", "bbb.csv", "ames.csv", "herg.csv"]


def _print_banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _run_step(name: str, cmd: Sequence[str]) -> int:
    _print_banner(name)
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode == 0:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} (exit code {result.returncode})")
    return int(result.returncode)


def _ensure_training_data(python_exe: str, dataset_path: Path) -> Tuple[bool, List[str]]:
    """Generate endpoint sample data if missing and verify dataset path."""
    messages: List[str] = []

    missing_endpoint_files = [
        filename for filename in REQUIRED_ENDPOINT_DATA if not (PROJECT_ROOT / "data" / filename).exists()
    ]
    if missing_endpoint_files:
        messages.append(
            "Missing endpoint sample files: "
            + ", ".join(missing_endpoint_files)
            + ". Running generate_sample_data.py."
        )
        rc = _run_step("Generate Sample Data", [python_exe, "generate_sample_data.py"])
        if rc != 0:
            return False, messages + ["Sample data generation failed."]

    if not dataset_path.exists():
        messages.append(
            f"Multitask dataset not found: {dataset_path}. "
            "Provide --data with a valid CSV path."
        )
        return False, messages

    return True, messages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run training + tests + system verification for in_silico_admet."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATASET),
        help="Path to multitask training CSV (default: data/admet_multitask.csv).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training step.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip unit tests.",
    )
    parser.add_argument(
        "--skip-system-check",
        action="store_true",
        help="Skip test_system.py health check.",
    )
    parser.add_argument(
        "--launch-app",
        action="store_true",
        help="Launch Streamlit app after all checks pass.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=3,
        help="Scaffold CV folds for train_multitask.py (default: 3).",
    )
    parser.add_argument(
        "--calibration-folds",
        type=int,
        default=3,
        help="Calibration folds for train_multitask.py (default: 3).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    python_exe = sys.executable
    dataset_path = Path(args.data)
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()

    _print_banner("In-Silico ADMET: Full Runner")
    print("Project root:", PROJECT_ROOT)
    print("Python:", python_exe)
    print("Dataset:", dataset_path)

    data_ok, data_messages = _ensure_training_data(python_exe, dataset_path)
    for line in data_messages:
        print("[INFO]", line)
    if not data_ok:
        return 1

    steps: List[Tuple[str, List[str]]] = []
    if not args.skip_train:
        steps.append(
            (
                "Train Multitask Model",
                [
                    python_exe,
                    "train_multitask.py",
                    "--data",
                    str(dataset_path),
                    "--splits",
                    str(args.splits),
                    "--calibration-folds",
                    str(args.calibration_folds),
                ],
            )
        )

    if not args.skip_tests:
        steps.append(
            (
                "Run Unit Tests",
                [python_exe, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
            )
        )

    if not args.skip_system_check:
        steps.append(("Run System Verification", [python_exe, "test_system.py"]))

    failed_steps: List[str] = []
    for name, cmd in steps:
        rc = _run_step(name, cmd)
        if rc != 0:
            failed_steps.append(name)

    _print_banner("Run Summary")
    if failed_steps:
        print("[FAIL] One or more steps failed:")
        for name in failed_steps:
            print(" -", name)
        return 1

    print("[PASS] All selected steps completed successfully.")

    if args.launch_app:
        return _run_step("Launch Streamlit App", [python_exe, "-m", "streamlit", "run", "app.py"])

    print("Tip: add --launch-app to start the Streamlit UI after checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
