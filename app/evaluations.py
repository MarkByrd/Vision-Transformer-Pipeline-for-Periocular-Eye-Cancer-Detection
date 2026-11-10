"""
Storage of evaluation records (spec FR-09 / FR-12).

Each saved evaluation becomes one folder under the data directory:

    <data dir>/
        evaluations.csv              one row per record (index for review / analysis)
        EV-20261110-142233-7F3A/
            record.json              all form fields, AI result, references to the images
            original.<ext>           the uploaded file, byte-for-byte
            annotated.png            original + the user's lines (only if they drew any)
            gmar.jpg                 GMAR attention overlay shown to the user

The folder is written under a temporary name and renamed when complete, so a crash
never leaves a half-written record behind. Access control (FR-12) relies on the
operating system: on macOS/Linux the data directory is created readable by the
current user only; on Windows, place it in a folder restricted to authorised staff.
"""

import csv
import json
import os
import secrets
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

PERFORMANCE = ("Good", "Poor", "Unknown")
MEDICAL_ASSESSMENT = ("Benign", "Malignant", "Don't know")

CSV_COLUMNS = [
    "record_id", "saved_at",
    "clinical_user", "evaluation_datetime", "performance", "comments",
    "consultant", "medical_assessment", "medical_comments",
    "model", "prediction", "confidence", "probability_malignant", "iris_method",
    "original_filename", "original_image", "annotated_image", "gmar_image",
]

_lock = threading.Lock()


def _required_text(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("is required")
    return v


class ClinicalUser(BaseModel):
    name: str = Field(max_length=200)
    evaluation_datetime: str = Field(max_length=40)
    performance: Literal[PERFORMANCE]
    comments: str = Field(default="", max_length=500)

    _check_name = field_validator("name")(_required_text)


class Consultant(BaseModel):
    name: str = Field(default="", max_length=200)
    medical_assessment: Literal[MEDICAL_ASSESSMENT] | None = None
    medical_comments: str = Field(default="", max_length=5000)


class AIResult(BaseModel):
    model: str
    prediction: Literal["Benign", "Malignant"]
    confidence: float
    probability_malignant: float
    iris_method: str


class EvaluationIn(BaseModel):
    """The JSON part of a save request (images are sent as separate files)."""
    clinical_user: ClinicalUser
    consultant: Consultant
    ai_result: AIResult
    original_filename: str = Field(max_length=500)
    # Lines drawn on the image: each line is a list of [x, y] points, normalised to 0-1.
    annotation_lines: list[list[list[float]]] = Field(default_factory=list)


def ensure_data_dir(data_dir: Path) -> None:
    """Create the data directory if needed (owner-only on macOS/Linux).

    Permissions are only set at creation, so access rules an administrator
    puts on an existing folder are left alone.
    """
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        if os.name == "posix":
            os.chmod(data_dir, 0o700)


def _new_record_id(data_dir: Path) -> str:
    while True:
        rid = f"EV-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2).upper()}"
        if not (data_dir / rid).exists():
            return rid


def save_evaluation(
    data_dir: Path,
    record: EvaluationIn,
    checkpoint_name: str,
    original: tuple[bytes, str],
    gmar_jpeg: bytes,
    annotated_png: bytes | None,
) -> dict:
    """Write one evaluation record. `original` is (file bytes, file extension)."""
    original_bytes, original_ext = original
    with _lock:
        ensure_data_dir(data_dir)
        record_id = _new_record_id(data_dir)
        saved_at = datetime.now().astimezone().isoformat(timespec="seconds")

        images = {
            "original": f"original{original_ext}",
            "annotated": "annotated.png" if annotated_png else None,
            "gmar": "gmar.jpg",
        }
        doc = {
            "record_id": record_id,
            "saved_at": saved_at,
            "clinical_user": record.clinical_user.model_dump(),
            "consultant": record.consultant.model_dump(),
            "ai_result": {**record.ai_result.model_dump(), "checkpoint": checkpoint_name},
            "original_filename": record.original_filename,
            "images": images,
            "annotation_lines": record.annotation_lines,
        }

        tmp = data_dir / f".{record_id}.partial"
        try:
            tmp.mkdir()
            (tmp / images["original"]).write_bytes(original_bytes)
            (tmp / images["gmar"]).write_bytes(gmar_jpeg)
            if annotated_png:
                (tmp / images["annotated"]).write_bytes(annotated_png)
            (tmp / "record.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.rename(data_dir / record_id)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

        _append_csv(data_dir / "evaluations.csv", doc)
    return {"record_id": record_id, "saved_at": saved_at}


def _append_csv(path: Path, doc: dict) -> None:
    cu, co, ai, im = doc["clinical_user"], doc["consultant"], doc["ai_result"], doc["images"]
    folder = doc["record_id"]
    row = {
        "record_id": doc["record_id"],
        "saved_at": doc["saved_at"],
        "clinical_user": cu["name"],
        "evaluation_datetime": cu["evaluation_datetime"],
        "performance": cu["performance"],
        "comments": cu["comments"],
        "consultant": co["name"],
        "medical_assessment": co["medical_assessment"] or "",
        "medical_comments": co["medical_comments"],
        "model": ai["model"],
        "prediction": ai["prediction"],
        "confidence": ai["confidence"],
        "probability_malignant": ai["probability_malignant"],
        "iris_method": ai["iris_method"],
        "original_filename": doc["original_filename"],
        "original_image": f"{folder}/{im['original']}",
        "annotated_image": f"{folder}/{im['annotated']}" if im["annotated"] else "",
        "gmar_image": f"{folder}/{im['gmar']}",
    }
    new_file = not path.exists()
    # utf-8-sig so Excel opens names with accents correctly.
    with open(path, "a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
