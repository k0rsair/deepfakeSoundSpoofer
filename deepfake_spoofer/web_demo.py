from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request

from deepfake_spoofer.predict import checkpoint_max_seconds, load_checkpoint_and_model, predict_audio_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight web demo for the deepfake detector.")
    parser.add_argument("--checkpoint", default="runs/wav2vec_pyara/best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--title", default="Демо детектора аудиодипфейков")
    return parser.parse_args()


def create_app(args: argparse.Namespace) -> Flask:
    device = torch.device(args.device)
    checkpoint, model = load_checkpoint_and_model(args.checkpoint, device)
    resolved_max_seconds = checkpoint_max_seconds(checkpoint, args.max_seconds)

    base_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
        static_url_path="/static",
    )
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    app.config["CHECKPOINT_PATH"] = str(Path(args.checkpoint))
    app.config["DEMO_TITLE"] = args.title
    app.config["DEVICE_LABEL"] = str(device)
    app.config["MAX_SECONDS"] = resolved_max_seconds
    app.config["MODEL_TYPE"] = checkpoint.get("config", {}).get("model_type", "unknown")
    app.config["BUNDLE_NAME"] = checkpoint.get("config", {}).get("bundle", "unknown")

    @app.get("/")
    def index() -> str:
        return render_template(
            "web_demo.html",
            demo_title=app.config["DEMO_TITLE"],
            checkpoint_path=app.config["CHECKPOINT_PATH"],
            device_label=app.config["DEVICE_LABEL"],
            model_type=app.config["MODEL_TYPE"],
            bundle_name=app.config["BUNDLE_NAME"],
            max_seconds=app.config["MAX_SECONDS"],
        )

    @app.get("/health")
    def health() -> tuple[dict[str, object], int]:
        return {
            "ok": True,
            "checkpoint": app.config["CHECKPOINT_PATH"],
            "device": app.config["DEVICE_LABEL"],
            "model_type": app.config["MODEL_TYPE"],
            "bundle": app.config["BUNDLE_NAME"],
        }, 200

    @app.post("/api/predict")
    def api_predict():
        uploaded = request.files.get("audio")
        if uploaded is None or uploaded.filename == "":
            return jsonify({"error": "Сначала загрузите аудиофайл."}), 400

        suffix = Path(uploaded.filename).suffix or ".wav"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                uploaded.save(handle)
                temp_path = Path(handle.name)

            result = predict_audio_path(
                checkpoint,
                model,
                device,
                temp_path,
                max_seconds=args.max_seconds,
            )
        except Exception as exc:
            return jsonify({"error": f"Не удалось выполнить инференс: {exc}"}), 400
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

        return jsonify(
            {
                "filename": uploaded.filename,
                "label": result.label,
                "p_bonafide": result.p_bonafide,
                "p_spoof": result.p_spoof,
                "confidence": result.confidence,
                "sample_rate": result.sample_rate,
                "original_seconds": result.original_seconds,
                "processed_seconds": result.processed_seconds,
                "cropped": result.cropped,
                "checkpoint": app.config["CHECKPOINT_PATH"],
                "model_type": app.config["MODEL_TYPE"],
                "bundle": app.config["BUNDLE_NAME"],
            }
        )

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
