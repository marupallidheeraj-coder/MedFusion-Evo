from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

APP_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = APP_ROOT / "artifacts"

app = FastAPI(title="MedFusion-Evo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded ML stack. This keeps `/api/health` lightweight and avoids loading
# torch/shap until an analysis request is actually made.
_services = None


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _get_services():
    global _services
    if _services is not None:
        return _services

    missing = [
        ARTIFACT_DIR / "metadata.json",
        ARTIFACT_DIR / "clinical_scaler.joblib",
        ARTIFACT_DIR / "clinical_model.joblib",
        ARTIFACT_DIR / "image_pca.joblib",
        ARTIFACT_DIR / "fusion_model.joblib",
        ARTIFACT_DIR / "clinical_background.npy",
    ]
    missing_files = [str(p.relative_to(APP_ROOT)) for p in missing if not p.exists()]
    if missing_files:
        raise RuntimeError(
            "Model artifacts are missing. Run scripts/train_multimodal.py first. "
            f"Missing: {', '.join(missing_files)}"
        )

    from joblib import load
    from ml.image_model import ImageModelService

    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    clinical_scaler = load(ARTIFACT_DIR / "clinical_scaler.joblib")
    clinical_model = load(ARTIFACT_DIR / "clinical_model.joblib")
    image_pca = load(ARTIFACT_DIR / "image_pca.joblib")
    fusion_model = load(ARTIFACT_DIR / "fusion_model.joblib")
    clinical_background = np.load(ARTIFACT_DIR / "clinical_background.npy")
    image_service = ImageModelService()

    _services = {
        "metadata": metadata,
        "clinical_scaler": clinical_scaler,
        "clinical_model": clinical_model,
        "image_pca": image_pca,
        "fusion_model": fusion_model,
        "clinical_background": clinical_background,
        "image_service": image_service,
    }
    return _services


def _parse_float(name: str, raw: str | None, minimum: float, maximum: float) -> float:
    try:
        value = float(raw) if raw is not None else float("nan")
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"{name} must be numeric.")
    if not np.isfinite(value):
        raise HTTPException(status_code=400, detail=f"{name} is required.")
    if value < minimum or value > maximum:
        raise HTTPException(status_code=400, detail=f"{name} must be between {minimum} and {maximum}.")
    return value


def _sex_code(sex: str) -> float:
    normalized = sex.strip().lower()
    if normalized in {"male", "m", "1"}:
        return 1.0
    if normalized in {"female", "f", "0"}:
        return 0.0
    raise HTTPException(status_code=400, detail="Sex must be Male or Female.")


def _risk_label(score: float) -> str:
    # UI label only; the score is not a calibrated clinical probability.
    if score >= 0.70:
        return "Elevated"
    if score >= 0.40:
        return "Intermediate"
    return "Lower"


def _data_uri_png(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _clinical_shap(model, background: np.ndarray, x: np.ndarray, feature_names: list[str]):
    import shap

    explainer = shap.LinearExplainer(model, background)
    sv = explainer.shap_values(x)

    # SHAP API changed shape conventions across versions. Normalize to one row.
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[0, :, -1]
    elif sv.ndim == 2:
        sv = sv[0]
    else:
        sv = sv.reshape(-1)

    contributions = [
        {"feature": f, "shap": float(v)}
        for f, v in zip(feature_names, sv)
    ]
    contributions.sort(key=lambda item: abs(item["shap"]), reverse=True)
    return contributions


@app.get("/health")
def health():
    return {
        "status": "ok",
        "artifacts_ready": (ARTIFACT_DIR / "metadata.json").exists(),
    }


@app.post("/analyze")
async def analyze(
    image: Annotated[UploadFile, File(...)],
    age: Annotated[str, Form(...)],
    sex: Annotated[str, Form(...)],
    temperature: Annotated[str, Form(...)],
    spo2: Annotated[str, Form(...)],
    wbc: Annotated[str, Form(...)],
    neutrophils: Annotated[str, Form(...)],
    lymphocytes: Annotated[str, Form(...)],
):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Upload a JPEG, PNG, or WebP chest X-ray.")

    payload = await image.read()
    if len(payload) > 4_000_000:
        raise HTTPException(status_code=413, detail="Image is too large. Keep the upload below 4 MB.")

    try:
        pil_image = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    svc = _get_services()
    metadata = svc["metadata"]

    # Feature order is fixed by training metadata. This is critical: inference
    # must use the exact same order and preprocessing used during training.
    clinical_values = {
        "age": _parse_float("Age", age, 0, 120),
        "sex": _sex_code(sex),
        "temperature": _parse_float("Temperature", temperature, 25, 45),
        "spo2": _parse_float("SpO2", spo2, 50, 100),
        "wbc": _parse_float("WBC", wbc, 0, 100),
        "neutrophils": _parse_float("Neutrophils", neutrophils, 0, 100),
        "lymphocytes": _parse_float("Lymphocytes", lymphocytes, 0, 100),
    }

    all_features = metadata["clinical_features"]
    selected_features = metadata["selected_clinical_features"]
    raw_x = np.array([[clinical_values[f] for f in selected_features]], dtype=np.float32)
    scaled_x = svc["clinical_scaler"].transform(raw_x)

    clinical_prob = float(svc["clinical_model"].predict_proba(scaled_x)[0, 1])
    shap_rows = _clinical_shap(
        svc["clinical_model"],
        svc["clinical_background"],
        scaled_x,
        selected_features,
    )

    image_result = svc["image_service"].predict(pil_image)
    image_embedding = image_result["embedding"]
    image_reduced = svc["image_pca"].transform(image_embedding.reshape(1, -1)).astype(np.float32)

    fusion_input = np.concatenate([image_reduced, scaled_x], axis=1)
    fusion_prob = float(svc["fusion_model"].predict_proba(fusion_input)[0, 1])

    heatmap = image_result["gradcam_png"]
    result = {
        "project": "MedFusion-Evo",
        "disease": metadata.get("target_name", "Pneumonia"),
        "screening": {
            "score": round(fusion_prob * 100, 2),
            "label": _risk_label(fusion_prob),
            "note": "Model score for research/demo screening; not a calibrated clinical probability or diagnosis.",
        },
        "xray": {
            "model": metadata.get("image_model_repo", "nismal1u/chestX-rays-DenseNet121"),
            "pneumonia_score": round(image_result["probability"] * 100, 2),
            "uncertainty": round(image_result["uncertainty"] * 100, 2),
            "assessment": "Higher model score for pneumonia pattern" if image_result["probability"] >= 0.5 else "Lower model score for pneumonia pattern",
            "gradcam": _data_uri_png(heatmap),
        },
        "clinical": {
            "input": clinical_values,
            "ga_selected_features": selected_features,
            "ga_selected_count": len(selected_features),
            "shap": shap_rows,
        },
        "fusion": {
            "image_representation_dimensions": int(image_reduced.shape[1]),
            "clinical_representation_dimensions": int(scaled_x.shape[1]),
            "prediction_network": metadata.get("fusion_architecture", "MLP"),
        },
    }

    return JSONResponse(content=json.loads(json.dumps(result, default=_jsonable)))
