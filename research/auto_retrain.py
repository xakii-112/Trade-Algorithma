"""Yeni canlı sonuçlarla günlük aday eğitimi; aynı veri tekrar sınanmaz.

Mevcut live_retrain.assess (zaman/sembol + purged CV) kullanılır. Her tamamlanan
koşu veri/kod kimliğiyle saklanır; DURABLE adayın SINANAN alt-modeli ayrı dizine
yazılır. Canlı model/terfi politikası değişmez. Geçmiş sınavların tekrarları
bağımsız kanıt değildir; canlı model terfisi ayrıca forward kanıt ister.

    python -m research.auto_retrain
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from nihai.atomic_io import atomic_write_text
from nihai.config import DATA_DIR
from nihai.console import ensure_utf8_stdout
from nihai.signals.meta_features import FEATURE_NAMES
from research import build_live_dataset, live_retrain, purged_cv, train_meta

RUNS = DATA_DIR / "retraining"
KINDS = ("logistic", "gbm")
MIN_NEW_ROWS = live_retrain.MIN_ROWS


def run() -> dict:
    """İlk uygun veri veya ≥300 yeni satır + yeni veri günü geldiğinde eğit.

    ponytail: tüm adaylar için tek OS dosya kilidi; ayrı kohortlar eşzamanlı
    eğitilecekse kohort başına kilit gerekir. Süreç ölünce OS kilidi bırakır.
    """
    RUNS.mkdir(parents=True, exist_ok=True)
    with (RUNS / ".lock").open("a+b") as handle:
        try:
            if sys.platform == "win32":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return {"status": "BUSY", "reason": "Başka aday eğitim döngüsü kilidi tutuyor."}
        return _run_locked()


def _run_locked() -> dict:
    started = time.perf_counter()
    latest = RUNS / "latest.json"
    previous = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else {}
    df = build_live_dataset.build()
    if df.empty:
        return {"status": "INSUFFICIENT", "rows": 0, "reason": "Çözülmüş özellik/etiket çifti yok."}
    cohorts = df.attrs.get("cohorts", [])
    dates = train_meta._chrono_key(df["entry_time"])
    if (len(cohorts) != 1 or not all(cohorts[0]) or dates.isna().any()
            or not np.isfinite(df[[*FEATURE_NAMES, "pnl", "label"]].to_numpy(float)).all()
            or not df["label"].isin([0, 1]).all()):
        raise ValueError("Eğitim için tek bar/ufuk kohortu, geçerli tarihler ve sonlu sayılar gerekli.")
    if int(cohorts[0][1]) < 1:
        raise ValueError("Eğitim ufku pozitif olmalı.")
    df = train_meta.chrono_sort(df)
    recipe = {"models": list(KINDS), "folds": 5, "cohort": list(cohorts[0])}
    code_hash = hashlib.sha256(b"".join(Path(str(m.__file__)).read_bytes() for m in
        (build_live_dataset, live_retrain, purged_cv, train_meta)) + Path(__file__).read_bytes()).hexdigest()
    csv_text = df.to_csv(index=False, lineterminator="\n")
    data_hash = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    run_id = hashlib.sha256((data_hash + code_hash + json.dumps(recipe, sort_keys=True)).encode()).hexdigest()
    path = RUNS / run_id
    if (path / "report.json").exists():
        cached = (path / "report.json").read_text(encoding="utf-8")
        if json.loads(cached)["finished_at"] > previous.get("finished_at", ""):
            atomic_write_text(latest, cached)  # önceki koşuda yalnız latest yazımı kesilmiş olabilir
        return {"status": "UNCHANGED", "run_id": run_id, "rows": len(df)}
    last_day = dates.max().date().isoformat()
    new_rows = len(df)
    if (previous.get("verdict") in ("DURABLE", "NO_EDGE")
            and previous.get("code_sha256") == code_hash and previous.get("recipe") == recipe):
        old_id = previous["run_id"]
        if len(old_id) != 64 or any(c not in "0123456789abcdef" for c in old_id):
            raise ValueError("Geçersiz önceki eğitim kimliği.")
        with (RUNS / old_id / "dataset.csv").open(encoding="utf-8", newline="") as source:
            known = {(r["symbol"], r["entry_time"]) for r in csv.DictReader(source)}
        new_rows = sum((s, t) not in known for s, t in zip(df.symbol, df.entry_time, strict=True))
        if new_rows < MIN_NEW_ROWS or last_day <= previous["data_last_day"]:
            return {"status": "WAITING_DATA", "rows": len(df),
                    "new_rows": new_rows, "last_run_id": old_id}
    # Tur-88: örtüşen canlı veride her gece yeni sınav = çoklu test (Tur-85 dersi); sayı taşınır.
    sinav_no = 1 + sum(1 for _ in RUNS.glob("*/report.json"))
    path.mkdir(parents=True, exist_ok=True)
    dataset = path / "dataset.csv"
    atomic_write_text(dataset, csv_text, newline="")  # Windows'ta da hash'lenen LF baytlarını koru
    # Mevcut sklearn bağımlılığı; BLAS'ın canlı tarayıcıyla tüm çekirdekler için yarışmasını önler.
    with threadpool_limits(limits=1):
        result = live_retrain.assess(df, kinds=list(KINDS), folds=5, verbose=False)
    serial = {**result, "results": {k: {**v, "splits": {
        s: {a: b for a, b in detail.items() if a != "model"} if detail else None
        for s, detail in v["splits"].items()}} for k, v in result["results"].items()}}
    serial.update(status="COMPLETED", run_id=run_id, rows=len(df), new_rows=new_rows,
                  symbols=int(df.symbol.nunique()),
                  data_last_day=last_day, dataset_sha256=data_hash, code_sha256=code_hash,
                  recipe=recipe, candidate_path=None, promoted=False, sinav_no=sinav_no,
                  finished_at=datetime.now(UTC).isoformat(timespec="seconds"))
    if result["verdict"] == "DURABLE":
        kind = result["durable_kind"]
        tested = result["results"][kind]["splits"]["time"]
        provenance = train_meta._provenance(dataset_path=str(dataset), n_rows=len(df), train_frac=0.6,
            min_gate_keep=50, n_configs_tried=4 * len(KINDS) * 2, calibrate=None,
            ece_test=tested.get("ece_test"))
        provenance.update(code_sha256=code_hash, scope="candidate_only", cohort=recipe["cohort"],
                          sinav_no=sinav_no)
        candidate = train_meta.save_model(tested["model"], trained_on=str(dataset),
            oos_auc=result["results"][kind]["cv"]["mean_auc"], gate_threshold=result["threshold"],
            provenance=provenance, output_dir=path)
        serial["candidate_path"] = str(candidate)
    serial["seconds"] = round(time.perf_counter() - started, 2)
    # PF=inf gibi tanımlı sınır değerleri HTTP/JSON tüketicisinde geçersiz sayı olmasın.
    text = json.dumps(serial, ensure_ascii=False, default=str)
    serial = json.loads(text, parse_constant=lambda _: None)
    text = json.dumps(serial, ensure_ascii=False, allow_nan=False, indent=2)
    atomic_write_text(path / "report.json", text)  # tamamlanma işareti modelden SONRA
    atomic_write_text(latest, text)
    return serial


if __name__ == "__main__":
    ensure_utf8_stdout()
    print(json.dumps(run(), ensure_ascii=False, indent=2))
