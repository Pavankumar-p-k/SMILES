"""Minimal local API server that bridges React UI to in_silico_admet predictor.

Usage:
    python tools/admet_api_server.py --silico-root "C:\\...\\in_silico_admet" --host 127.0.0.1 --port 5051
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict


PREDICTOR = None
RISK_SUMMARY = None
ENDPOINT_INFO = None
LOAD_ERROR = None
SILICO_ROOT = ""


def _risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    if prob >= 0.3:
        return "MODERATE"
    return "LOW"


def _risk_color(prob: float) -> str:
    if prob >= 0.7:
        return "#ff4d6d"
    if prob >= 0.3:
        return "#ffcc00"
    return "#00ff88"


def _build_organs(predictions: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    bbb = float(predictions.get("BBB", 0.0))
    dili = float(predictions.get("DILI", 0.0))
    herg = float(predictions.get("hERG", 0.0))
    ames = float(predictions.get("Ames", 0.0))

    def build(endpoint: str, prob: float, hi_msg: str, lo_msg: str) -> Dict[str, Any]:
        pct = prob * 100.0
        return {
            "endpoint": endpoint,
            "probability": prob,
            "score": f"{prob:.3f}",
            "percent": round(pct, 2),
            "percent_text": f"{pct:.1f}%",
            "risk_label": _risk_label(prob),
            "risk_color": _risk_color(prob),
            "message": hi_msg if prob >= 0.7 else lo_msg,
        }

    return {
        "brain": build("BBB", bbb, "BBB penetration toxicity elevated.", "BBB risk currently controlled."),
        "heart": build("hERG", herg, "hERG cardiac risk elevated.", "hERG risk currently controlled."),
        "liver": build("DILI", dili, "DILI hepatotoxicity elevated.", "DILI risk currently controlled."),
        "genetic": build("Ames", ames, "Ames mutagenicity elevated.", "Mutagenicity risk currently controlled."),
    }


def _build_flags(predictions: Dict[str, float]) -> list[str]:
    bbb = float(predictions.get("BBB", 0.0))
    dili = float(predictions.get("DILI", 0.0))
    herg = float(predictions.get("hERG", 0.0))
    ames = float(predictions.get("Ames", 0.0))
    flags: list[str] = []

    if bbb >= 0.7:
        flags.append("Brain risk HIGH: BBB toxicity signal elevated.")
    elif bbb >= 0.3:
        flags.append("Brain risk MODERATE: BBB signal needs follow-up.")
    else:
        flags.append("Brain risk LOW by BBB model output.")

    if dili >= 0.7:
        flags.append("Liver risk HIGH: DILI output elevated.")
    if herg >= 0.7:
        flags.append("Cardiac risk HIGH: hERG output elevated.")
    if ames >= 0.7:
        flags.append("Genetic risk HIGH: Ames output elevated.")

    if len(flags) == 1:
        flags.append("No additional organ in high-risk zone.")
    return flags


class AdmetHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/health":
            ok = PREDICTOR is not None and LOAD_ERROR is None
            self._send_json(
                {
                    "ok": ok,
                    "silico_root": SILICO_ROOT,
                    "load_error": LOAD_ERROR,
                    "endpoints": list(ENDPOINT_INFO.keys()) if ENDPOINT_INFO else [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status=200 if ok else 500,
            )
            return
        self._send_json({"ok": False, "error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/predict":
            self._send_json({"ok": False, "error": "Not found"}, status=404)
            return
        if PREDICTOR is None or LOAD_ERROR:
            self._send_json({"ok": False, "error": f"Predictor unavailable: {LOAD_ERROR}"}, status=500)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            data = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)
            return

        smiles = str(data.get("smiles", "")).strip()
        include_shap = bool(data.get("include_shap", False))
        if not smiles:
            self._send_json({"ok": False, "error": "smiles is required"}, status=400)
            return

        result = PREDICTOR.predict_single(smiles, include_shap=include_shap)
        valid = bool(result.get("valid", False))
        predictions = result.get("predictions", {}) if valid else {}
        summary = RISK_SUMMARY(predictions) if valid else None
        organs = _build_organs(predictions) if valid else {}
        flags = _build_flags(predictions) if valid else []

        payload = {
            "ok": valid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "smiles": smiles,
            "raw": result,
            "predictions": predictions,
            "summary": summary,
            "organs": organs,
            "risk_flags": flags,
            "endpoint_info": ENDPOINT_INFO,
            "dataset_reference": {
                "training_fingerprints": len(getattr(PREDICTOR, "training_fps", []) or []),
                "sources": {
                    "brain": "BBB model output",
                    "heart": "hERG model output",
                    "liver": "DILI model output",
                    "genetic": "Ames model output",
                },
            },
        }
        if not valid:
            payload["error"] = result.get("error", "Prediction failed")
        self._send_json(payload, status=200 if valid else 400)


def _load_predictor(silico_root: str) -> None:
    global PREDICTOR, RISK_SUMMARY, ENDPOINT_INFO, LOAD_ERROR, SILICO_ROOT
    SILICO_ROOT = silico_root
    try:
        os.chdir(silico_root)
        sys.path.insert(0, silico_root)
        from predictor import ADMETPredictor  # type: ignore
        from risk_aggregator import get_risk_summary  # type: ignore
        from config import ENDPOINT_INFO as CFG_ENDPOINT_INFO  # type: ignore

        PREDICTOR = ADMETPredictor()
        RISK_SUMMARY = get_risk_summary
        ENDPOINT_INFO = CFG_ENDPOINT_INFO
        LOAD_ERROR = None
    except Exception as exc:  # noqa: BLE001
        LOAD_ERROR = str(exc)
        PREDICTOR = None
        RISK_SUMMARY = None
        ENDPOINT_INFO = {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silico-root", required=True, help="Path to in_silico_admet project root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5051)
    args = parser.parse_args()

    _load_predictor(args.silico_root)
    server = ThreadingHTTPServer((args.host, args.port), AdmetHandler)
    print(f"ADMET API listening on http://{args.host}:{args.port} (silico_root={args.silico_root})")
    if LOAD_ERROR:
        print(f"Predictor failed to load: {LOAD_ERROR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
