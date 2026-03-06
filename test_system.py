"""System verification and health check."""

import sys
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def test_imports() -> bool:
    """Test all critical imports."""
    print("\n" + "=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)

    modules = {
        "rdkit": "RDKit",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "sklearn": "Scikit-learn",
        "xgboost": "XGBoost",
        "streamlit": "Streamlit",
        "reportlab": "ReportLab",
        "joblib": "Joblib",
    }

    all_ok = True
    for module, display_name in modules.items():
        try:
            __import__(module)
            print(f"{PASS} {display_name:<20} OK")
        except ImportError as exc:
            print(f"{FAIL} {display_name:<20} FAILED: {exc}")
            all_ok = False

    return all_ok


def test_versions() -> bool:
    """Test critical version requirements."""
    print("\n" + "=" * 60)
    print("CHECKING VERSIONS")
    print("=" * 60)

    try:
        import numpy
        import pandas
        import sklearn
        import xgboost
    except ImportError as exc:
        print(f"{FAIL} Cannot check versions; missing dependency: {exc}")
        return False

    checks = [
        ("NumPy", numpy.__version__, (1, 26, 0), ">=1.26.0"),
        ("Pandas", pandas.__version__, (1, 5, 0), ">=1.5.0"),
        ("Scikit-learn", sklearn.__version__, (1, 3, 0), ">=1.3.0"),
        ("XGBoost", xgboost.__version__, (2, 0, 0), ">=2.0.0"),
    ]

    all_ok = True
    for name, version, min_version, requirement in checks:
        parts = version.split(".")
        major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
        minor_raw = parts[1] if len(parts) > 1 else "0"
        patch_raw = parts[2] if len(parts) > 2 else "0"
        minor = int("".join(ch for ch in minor_raw if ch.isdigit()) or "0")
        patch = int("".join(ch for ch in patch_raw if ch.isdigit()) or "0")

        ok = (major, minor, patch) >= min_version

        status = PASS if ok else FAIL
        print(f"{status} {name:<20} {version:<15} (require {requirement})")
        if not ok:
            all_ok = False

    return all_ok


def test_directories() -> bool:
    """Test directory structure."""
    print("\n" + "=" * 60)
    print("CHECKING DIRECTORIES")
    print("=" * 60)

    project_root = Path(__file__).parent
    required_dirs = [
        project_root / "data",
        project_root / "models",
        project_root / "reports",
    ]

    all_ok = True
    for directory in required_dirs:
        if directory.exists():
            print(f"{PASS} {directory.name:<20} exists")
        else:
            print(f"{WARN} {directory.name:<20} missing (creating)")
            directory.mkdir(parents=True, exist_ok=True)

    return all_ok


def test_data() -> bool:
    """Check if training data exists."""
    print("\n" + "=" * 60)
    print("CHECKING TRAINING DATA")
    print("=" * 60)

    project_root = Path(__file__).parent
    data_dir = project_root / "data"

    required_files = ["dili.csv", "bbb.csv", "ames.csv", "herg.csv"]
    all_found = True

    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"{PASS} {filename:<20} found ({size} bytes)")
        else:
            print(f"{FAIL} {filename:<20} NOT FOUND")
            all_found = False

    multitask_csv = data_dir / "admet_multitask.csv"
    if multitask_csv.exists():
        print(f"{PASS} {'admet_multitask.csv':<20} found ({multitask_csv.stat().st_size} bytes)")
    else:
        print(f"{WARN} {'admet_multitask.csv':<20} NOT FOUND")

    if not all_found:
        print("\nMissing sample endpoint training data.")
        print("Run: python generate_sample_data.py")
        return False

    return True


def test_models() -> bool:
    """Check if trained models exist."""
    print("\n" + "=" * 60)
    print("CHECKING TRAINED MODELS")
    print("=" * 60)

    project_root = Path(__file__).parent
    models_dir = project_root / "models"

    endpoint_models = [
        "DILI_model.pkl",
        "BBB_model.pkl",
        "Ames_model.pkl",
        "hERG_model.pkl",
    ]
    bundle_model = models_dir / "multitask_admet_model.pkl"

    found_endpoint_models = 0
    for model_file in endpoint_models:
        model_path = models_dir / model_file
        if model_path.exists():
            size = model_path.stat().st_size / 1024
            print(f"{PASS} {model_file:<26} found ({size:.1f} KB)")
            found_endpoint_models += 1
        else:
            print(f"{WARN} {model_file:<26} NOT FOUND")

    if bundle_model.exists():
        size = bundle_model.stat().st_size / 1024
        print(f"{PASS} {'multitask_admet_model.pkl':<26} found ({size:.1f} KB)")

    has_models = bundle_model.exists() or found_endpoint_models > 0
    if not has_models:
        print("\nNo trained models found.")
        print("Run: python train_multitask.py --data data/admet_multitask.csv")
        return False

    return True


def test_functionality() -> bool:
    """Test core functionality."""
    print("\n" + "=" * 60)
    print("TESTING CORE FUNCTIONALITY")
    print("=" * 60)

    try:
        from featurizer import featurize_smiles, is_valid_smiles

        print(f"{PASS} Featurizer imports OK")

        smiles = "CCO"
        assert is_valid_smiles(smiles), "Valid SMILES rejected"
        print(f"{PASS} SMILES validation OK ({smiles})")

        features = featurize_smiles(smiles)
        assert features.shape == (2055,), f"Wrong shape: {features.shape}"
        print(f"{PASS} Featurization OK (2055 dims)")

        from predictor import ADMETPredictor

        predictor = ADMETPredictor()
        assert len(predictor.models) > 0, "No models loaded"
        print(f"{PASS} Predictor loaded {len(predictor.models)} models")

        result = predictor.predict_single(smiles)
        assert result is not None, "Prediction failed"
        assert result.get("valid", False), f"Prediction marked invalid: {result.get('error')}"
        endpoint_predictions = result.get("predictions", {})
        assert endpoint_predictions, "No endpoint predictions returned"
        print(f"{PASS} Single prediction OK: {endpoint_predictions}")

        from risk_aggregator import get_risk_summary

        summary = get_risk_summary(endpoint_predictions)
        assert "composite_score" in summary, "Missing composite score"
        assert "risk_category" in summary, "Missing risk category"
        print(f"{PASS} Risk aggregation OK: {summary['risk_category']}")

        return True

    except Exception as exc:
        print(f"{FAIL} Functionality test failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def main() -> int:
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print(" ADMET PLATFORM SYSTEM VERIFICATION ".center(60))
    print("=" * 60)

    results = {
        "Imports": test_imports(),
        "Versions": test_versions(),
        "Directories": test_directories(),
        "Training Data": test_data(),
        "Trained Models": test_models(),
        "Functionality": test_functionality(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for value in results.values() if value)
    total = len(results)

    for test_name, result in results.items():
        status = f"{PASS} PASS" if result else f"{FAIL} FAIL"
        print(f"{status:<12} {test_name}")

    print(f"\n{passed}/{total} checks passed")

    if passed == total:
        print("\n" + "=" * 60)
        print("SYSTEM READY")
        print("=" * 60)
        print("\nRun: streamlit run app.py")
        return 0

    print("\n" + "=" * 60)
    print("SYSTEM SETUP INCOMPLETE")
    print("=" * 60)
    if not results["Training Data"]:
        print("\nFix: python generate_sample_data.py")
    if not results["Trained Models"]:
        print("Fix: python train_multitask.py --data data/admet_multitask.csv")
    return 1


if __name__ == "__main__":
    sys.exit(main())
