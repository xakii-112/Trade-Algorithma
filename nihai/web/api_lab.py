"""Mevcut simülasyon/araştırma araçlarının operatör kontrollü web yüzeyi."""
from __future__ import annotations

import csv
import io
import json
import re
import runpy
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nihai.config import CONFIG, DATA_DIR, HTF_INTERVAL, INITIAL_EQUITY, LOG_DIR
from nihai.web import lab_jobs as jobs


def operator_request(request: Request) -> None:
    if request.method == "POST":
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            raise HTTPException(415, "application/json gerekli.")
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "İşlemi aynı panel üzerinden başlatın.")


router = APIRouter(dependencies=[Depends(operator_request)])


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def normalize_symbols(value: list[str]) -> list[str]:
    out = list(dict.fromkeys(s.strip().upper() for s in value))
    if not out or any(not re.fullmatch(r"[A-Z^][A-Z0-9.^-]{0,14}", s) for s in out):
        raise ValueError("Geçerli hisse sembolleri girin.")
    return out


def dataset_path(name: str) -> Path:
    if not isinstance(name, str) or not name or len(name) > 240 or "\x00" in name:
        raise ValueError("Geçerli CSV adı gerekli.")
    path = (DATA_DIR / name).resolve()
    if not path.is_relative_to(DATA_DIR.resolve()) or path.suffix.lower() != ".csv" or not path.is_file():
        raise ValueError("Veri dizinindeki mevcut bir CSV seçin.")
    if path.stat().st_size > 250_000_000:
        raise ValueError("Panel eğitim CSV sınırı 250 MB; daha küçük bir dilim seçin.")
    return path


class BacktestInput(Input):
    symbols: list[str] = Field(min_length=1, max_length=20)
    strategy: str = "event_breakout"
    interval: Literal["1m", "5m", "15m", "1h", "1d"] = "1d"
    period: Literal["1mo", "3mo", "6mo", "1y", "2y", "5y"] = "1y"
    from_date: date | None = None
    to_date: date | None = None
    equity: float = Field(default=INITIAL_EQUITY, gt=0, le=1e10)
    mc_runs: int = Field(default=1000, ge=0, le=5000)
    impact_coef: float | None = Field(default=None, ge=0, le=5)
    mode: Literal["single", "portfolio", "options", "walk_forward"] = "single"
    is_bars: int = Field(default=180, ge=61, le=10000)
    oos_bars: int = Field(default=60, ge=1, le=2000)
    wf_metric: Literal["expectancy", "sharpe", "return"] = "expectancy"
    stop_atr: float = Field(default=CONFIG.strategy.trend_stop_atr, gt=0, le=20)
    target_atr: float = Field(default=CONFIG.strategy.trend_target_atr, gt=0, le=100)

    _symbols = field_validator("symbols")(normalize_symbols)

    @field_validator("strategy")
    @classmethod
    def check_strategy(cls, value):
        from nihai.signals.strategy import BASE_STRATEGY_MODES
        if value not in (*BASE_STRATEGY_MODES, "ensemble", "confluence"):
            raise ValueError("Tarihsel teste uygun strateji seçin.")
        return value

    @model_validator(mode="after")
    def dates(self):
        if self.mode == "options" and self.interval != "1d":
            raise ValueError("Gerçek Dolt opsiyon testi günlük zincir kullanır; bar aralığı 1d olmalı.")
        if self.to_date and not self.from_date:
            raise ValueError("Bitiş için başlangıç tarihi de gerekli.")
        if self.from_date and self.from_date >= (self.to_date or date.today()):
            raise ValueError("Başlangıç bitişten önce olmalı.")
        return self


@router.post("/api/backtest/run", status_code=202)
def backtest_run(req: BacktestInput):
    return jobs.start("backtest", req.model_dump(mode="json"))


@router.get("/api/backtest/status/{job_id}")
@router.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return jobs.status(job_id)


@router.get("/api/backtest/report/{job_id}")
@router.get("/api/jobs/{job_id}/report")
def job_report(job_id: str):
    return jobs.report(job_id)


@router.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    return jobs.cancel(job_id)


@router.get("/api/jobs")
def job_list():
    paths = sorted(jobs.JOBS_DIR.glob("*/request.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"jobs": [jobs.status(p.parent.name) for p in paths[:30]]}


@router.get("/api/research/edge-lab/summary")
def edge_summary():
    from research.edge_lab import ledger
    entries = ledger.read_entries()
    latest = {r["candidate"]: r for r in entries if r.get("candidate")}
    return {"entries": list(latest.values()), "runs": len(entries), "families": len(latest),
            "next_family_count": len(latest) + 1, "note": ledger.summary()}


@router.get("/api/reports/tax-8949")
def tax_report(year: int | None = Query(default=None, ge=1900, le=2200)):
    from nihai.tax_report import build
    result = build(DATA_DIR / "trades.csv")
    if year is not None:
        for key in ("rows", "section_1256"):
            result[key] = [r for r in result[key] if r["date_sold"].startswith(f"{year}-")]
    rows, other = result["rows"], result["section_1256"]
    result.update(n=len(rows), n_1256=len(other), short_term_total=round(sum(r["gain_loss"] for r in rows if r["part"].startswith("I (")), 2),
                  long_term_total=round(sum(r["gain_loss"] for r in rows if r["part"].startswith("II (")), 2),
                  total=round(sum(r["gain_loss"] for r in rows), 2),
                  total_1256=round(sum(r["gain_loss"] for r in other), 2),
                  n_wash_candidates=sum(bool(r["wash_sale_candidate"]) for r in rows + other),
                  virtual=True)
    return jobs.clean(result)


@router.get("/api/reports/tax-8949/download")
def tax_download(year: int | None = Query(default=None, ge=1900, le=2200),
                 section: Literal["8949", "6781"] = "8949"):
    from nihai.tax_report import FORM_8949_COLS
    result = tax_report(year)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FORM_8949_COLS)
    writer.writeheader()
    writer.writerows(result["section_1256" if section == "6781" else "rows"])
    return Response("\ufeff" + output.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="virtual_form_{section}_{year or "all"}.csv"'})


@router.get("/api/system/audit-chain")
def audit_chain():
    from nihai.audit_chain import CHAIN_PATH, seal
    result = seal(CHAIN_PATH)
    result["measured"] = result["n"] > 0 or not result["ok"]
    return result


@router.get("/api/system/clock-drift")
def clock_drift():
    from nihai.audit_chain import clock_drift as measure
    return measure()


@router.get("/api/screener/last-report")
def screener_report():
    import gzip
    paths = sorted([*LOG_DIR.glob("auto_screen_*.csv"), *LOG_DIR.glob("auto_screen_*.csv.gz")],
                   key=lambda p: p.name, reverse=True)
    if not paths:
        return {"rows": [], "reason": "Henüz sıralama raporu yok."}
    path = paths[0]
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {"rows": rows, "file": path.name, "modified_at": path.stat().st_mtime}


class ScreenerInput(Input):
    universe: Literal["sp500", "nasdaq100", "both"] = "both"
    top_n: int = Field(default=20, ge=1, le=500)


@router.post("/api/screener/run-now", status_code=202)
def screener_run(req: ScreenerInput):
    return jobs.start("screener", req.model_dump())


class MaintenanceInput(Input):
    apply: bool = False
    days: int = Field(default=30, ge=3, le=3650)
    compress_days: int = Field(default=7, ge=2, le=3650)
    confirmed: bool = False
    plan_token: str | None = None

    @model_validator(mode="after")
    def policy(self):
        if self.compress_days >= self.days:
            raise ValueError("Sıkıştırma günü saklama gününden küçük olmalı.")
        if self.apply and (not self.confirmed or not self.plan_token):
            raise ValueError("Bakım uygulaması için önizleme ve onay gerekli.")
        return self


@router.post("/api/system/log-maintenance")
def log_maintenance(req: MaintenanceInput):
    if req.apply:
        return jobs.start("maintenance", req.model_dump())
    return maintenance_plan(req)


def maintenance_plan(req: MaintenanceInput):
    import hashlib

    from nihai.log_retention import adaylar, sikistirma_adaylari
    def describe(items):
        return [{"file": a.yol.name, "date": a.tarih.isoformat(), "bytes": a.boyut} for a in items]
    plan = {"apply": False, "delete": describe(adaylar(LOG_DIR, req.days)),
            "compress": describe(sikistirma_adaylari(LOG_DIR, req.compress_days, req.days))}
    plan["plan_token"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


class DsrInput(Input):
    dataset: str
    columns: list[str] = Field(min_length=1, max_length=16)
    n_trials: int = Field(default=1, ge=1, le=1000000)
    blocks: Literal[4, 6, 8, 10, 12, 14, 16] = 8

    @field_validator("columns")
    @classmethod
    def check_columns(cls, value):
        if any(not c.strip() or len(c) > 100 for c in value) or len(set(value)) != len(value):
            raise ValueError("Birbirinden farklı, boş olmayan sütun adları gerekli.")
        return value

    @field_validator("dataset")
    @classmethod
    def check_dataset(cls, value):
        dataset_path(value)
        return value


@router.post("/api/research/run-dsr", status_code=202)
def run_dsr(req: DsrInput):
    return jobs.start("dsr", req.model_dump())


class EdgeInput(Input):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[\w .-]+$")
    dataset: str
    feature_cols: list[str] = Field(min_length=1, max_length=100)
    label_col: str = Field(default="label", min_length=1, max_length=100)
    ts_col: str = Field(default="ts", min_length=1, max_length=100)
    symbol_col: str = Field(default="symbol", min_length=1, max_length=100)
    pnl_col: str | None = Field(default=None, min_length=1, max_length=100)
    n_folds: int = Field(default=5, ge=2, le=10)
    thr: float = Field(default=0.55, gt=0, lt=1)
    expected_start: date
    expected_end: date
    warmup_days: int = Field(default=0, ge=0, le=1000)

    @model_validator(mode="after")
    def check(self):
        dataset_path(self.dataset)
        DsrInput.check_columns(self.feature_cols)
        if set(self.feature_cols) & {self.label_col, self.pnl_col, self.ts_col, self.symbol_col}:
            raise ValueError("Etiket, PnL, zaman ve sembol alanları özellik olamaz.")
        if self.expected_start >= self.expected_end:
            raise ValueError("Beyan edilen başlangıç bitişten önce olmalı.")
        return self


@router.post("/api/research/edge-lab/run", status_code=202)
def edge_run(req: EdgeInput):
    return jobs.start("edge_lab", req.model_dump(mode="json"))


def _cli(module: str, args: list[str]) -> None:
    saved = sys.argv
    try:
        sys.argv = [module, *args]
        try:
            runpy.run_module(module, run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise RuntimeError(f"Araç çıkış kodu {exc.code}") from exc
    finally:
        sys.argv = saved


def execute(kind: str, params: dict, path: Path):
    if kind == "edge_lab":
        from research.edge_lab.harness import run_candidate
        edge = EdgeInput(**params)
        options = edge.model_dump(mode="json", exclude={"dataset"})
        return run_candidate(dataset_csv=dataset_path(edge.dataset), **options)
    if kind == "backtest":
        from nihai.execution import backtest as bt
        req = BacktestInput(**params)
        # Yalnız iş sürecinin CONFIG'i değişir; canlı tarayıcı etkilenmez.
        object.__setattr__(CONFIG.strategy, "trend_stop_atr", req.stop_atr)
        object.__setattr__(CONFIG.strategy, "trend_target_atr", req.target_atr)
        if req.impact_coef is not None:
            object.__setattr__(CONFIG.backtest, "impact_coef", req.impact_coef)
        common: dict = dict(interval=req.interval, period=None if req.from_date else req.period,
                      start=req.from_date.isoformat() if req.from_date else None,
                      end=req.to_date.isoformat() if req.to_date else None,
                      initial_equity=req.equity, verbose=True)
        results = []
        if req.mode == "walk_forward":
            for sym in req.symbols:
                print(f"Walk-forward: {sym}", flush=True)
                wf_result = bt.walk_forward(sym, **common, is_bars=req.is_bars, oos_bars=req.oos_bars,
                                         metric=req.wf_metric)
                bt.print_walk_forward(wf_result)
                results.append(asdict(wf_result))
        else:
            common.update(strategy_mode=req.strategy, mc_iters=0)
            runners: dict[str, Callable[..., bt.BacktestResult]] = {
                "single": bt.backtest_symbol, "options": bt.backtest_option_chain, "portfolio": bt.backtest_portfolio}
            runner = runners[req.mode]
            targets = [req.symbols] if req.mode == "portfolio" else req.symbols
            for target in targets:
                print(f"Backtest: {target}", flush=True)
                result = runner(target, **common)
                if req.mc_runs and result.trade_pnls:
                    result.monte_carlo = bt.monte_carlo_paths(result.trade_pnls, req.equity,
                                                             req.mc_runs, include_bands=True)
                results.append(asdict(result))
        if not results or not any(r.get("bars_total", len(r.get("folds", []))) for r in results):
            raise ValueError("Seçilen aralıkta yeterli bar / walk-forward penceresi bulunamadı.")
        return {"mode": req.mode, "results": results, "parameters": params}
    if kind == "screener":
        from nihai.execution import auto_screener as screen
        screening = ScreenerInput(**params)
        screen.AUTO_SCREEN_UNIVERSE = screening.universe
        screen.AUTO_SCREEN_TOP_N = screening.top_n
        screen.FOCUSED_TOP_N = screening.top_n
        screen.AUTO_SCREEN_WRITE_CSV = True
        started = time.time()
        selected = screen.scan_and_update()
        report = screener_report()
        fresh = report.get("modified_at", 0) >= started
        return {"symbols": selected, "report": report, "ranking_created": fresh,
                "note": "Yeni sıralama kaydedildi." if fresh else "Bu koşuda yeni sıralama raporu üretilemedi; günlük ve varsa eski rapor gösteriliyor."}
    if kind == "maintenance":
        from nihai.log_retention import compress, purge
        maintenance = MaintenanceInput(**params)
        if maintenance.apply and maintenance_plan(maintenance)["plan_token"] != maintenance.plan_token:
            raise ValueError("Bakım adayları önizlemeden sonra değişti; yeniden önizleyin.")
        return {"compress": compress(LOG_DIR, maintenance.compress_days, maintenance.days, apply=maintenance.apply),
                "purge": purge(LOG_DIR, maintenance.days, apply=maintenance.apply)}
    if kind == "dsr":
        import numpy as np
        import pandas as pd

        from research.deflated import dsr, sharpe_estimator_var, sharpe_moments
        from research.pbo import cscv_pbo
        dsr_input = DsrInput(**params)
        matrix = pd.read_csv(dataset_path(dsr_input.dataset), usecols=dsr_input.columns).apply(pd.to_numeric, errors="raise")
        if len(matrix) < max(20, dsr_input.blocks * 2) or not np.isfinite(matrix.to_numpy()).all():
            raise ValueError("En az 20 dönem ve eksiksiz, sonlu getiri değerleri gerekli.")
        output = []
        n_trials = max(dsr_input.n_trials, len(dsr_input.columns))
        for col in dsr_input.columns:
            m = sharpe_moments(matrix[col].to_numpy())
            variance = sharpe_estimator_var(m["sr"], m["n"], m["skew"], m["kurt"])
            output.append(dict(column=col, **m, dsr=dsr(m["sr"], m["n"], n_trials,
                                                       variance, m["skew"], m["kurt"])))
        return {"columns": output, "n_trials": n_trials,
                "pbo": cscv_pbo(matrix.to_numpy(), s=dsr_input.blocks) if len(dsr_input.columns) >= 5 else None,
                "note": "Mevcut PBO motoru en az beş hizalı strateji sütunu ister; daha azında yalnız DSR ölçülür. DSR tek başına terfi değildir."}
    return execute_tool(kind, params, path)


def choice(label, values, default=None):
    return {"label": label, "type": "choice", "choices": values,
            "default": values[0] if default is None else default}


def number(label, default, minimum, maximum):
    return {"label": label, "type": "number", "default": default, "min": minimum, "max": maximum}


_DATASET = {"label": "Eğitim CSV", "type": "dataset", "default": ""}
_FOLDS = number("Doğrulama bölmesi", 5, 2, 10)
_MODELS = choice("Model", ["logistic", "gbm", "xgb", "both", "all"], "both")
_PERIODS = choice("Test dönemleri", ["1y", "2y", "1y,2y", "2y,5y"], "1y,2y")
_INTERVAL = choice("Bar aralığı", ["1h", "1d"], "1d")
_SYMBOLS = {"label": "Semboller (virgülle)", "type": "symbols", "default": "AAPL,MSFT,SPY"}
_DATE = {"label": "Başlangıç", "type": "date", "default": "2022-01-01"}
_BOOL = {"type": "bool", "default": False}

# Modül/adres/çıktı yolu istemciden alınmaz. UI ve doğrulama AYNI kayıtları okur.
TOOLS: dict[str, dict] = {
    "purged_cv": {"label": "Purged CV ve özellik önemi", "group": "research", "module": "research.purged_cv",
                  "fields": {"csv": _DATASET, "model": _MODELS, "folds": _FOLDS}, "json": True},
    "ablation": {"label": "Strateji ablasyonu", "group": "research", "module": "research.ablation", "fields": {}},
    "confluence": {"label": "Mutabakat karşılaştırması", "group": "research", "module": "research.confluence_test",
                   "fields": {"periods": _PERIODS, "interval": _INTERVAL, "min-votes": number("Asgari oy", 2, 2, 7)}},
    "agreement": {"label": "Model ve momentum uzlaşması", "group": "research", "module": "research.agreement",
                  "fields": {"csv": _DATASET}},
    "pit": {"label": "Point-in-time doğrulama", "group": "research", "module": "research.run_pit_validation",
            "fields": {"only": choice("Adımlar", ["2,3,4", "3,4", "4"]), "as-of": _DATE,
                       "limit": number("Sembol sınırı (0: tümü)", 0, 0, 1000)}},
    "pead": {"label": "PEAD model doğrulaması", "group": "research", "module": "research.train_pead_ml",
             "fields": {"csv": _DATASET, "model": _MODELS, "folds": _FOLDS}},
    "exit_grid": {"label": "Stop / hedef ızgarası", "group": "research", "module": "research.exit_grid",
                  "fields": {"periods": _PERIODS, "interval": _INTERVAL,
                             "mode": choice("Strateji", ["donchian", "event_breakout", "macd", "squeeze"])}},
    "wr_grid": {"label": "İsabet ızgarası", "group": "research", "module": "research.wr_grid",
                "fields": {"periods": _PERIODS, "interval": _INTERVAL}},
    "horizons": {"label": "Tahmin ufku / gecikme taraması", "group": "research", "module": "research.horizon_sweep",
                 "fields": {"max-symbols": number("Sembol sınırı (0: tümü)", 0, 0, 1000)}, "json": True},
    "capacity": {"label": "Veri miktarı / model kapasitesi", "group": "research", "module": "research.capacity_data_sweep",
                 "fields": {"dataset": _DATASET, "folds": _FOLDS, "max-epochs": number("Azami epoch", 40, 1, 500)}, "json": True},
    "feed_ab": {"label": "IEX / SIP veri karşılaştırması", "group": "research", "module": "research.feed_ab_test",
                "fields": {"symbols": _SYMBOLS, "interval": _INTERVAL, "bars": number("Bar", 400, 60, 5000)}},
    "pairs": {"label": "İkili işlem / eşbütünleşme", "group": "research", "module": "research.pairs_test",
              "fields": {"limit": number("Sembol sınırı (0: tümü)", 50, 0, 1000),
                         "grid": {**_BOOL, "label": "DSR/PBO ızgarası"},
                         "exec-lag": number("İcra gecikmesi (bar)", 1, 0, 10)}},
    "xsect": {"label": "Kesitsel momentum", "group": "research", "module": "research.xsect",
              "fields": {"universe": choice("Evren", ["pit_local", "basket", "nasdaq100", "sp500"]),
                         "periods": _PERIODS, "top": number("İlk N", 10, 1, 500)}},
    "smallcap": {"label": "Küçük şirket momentumu (Sharadar)", "group": "research", "module": "research.smallcap_momentum",
                 "fields": {"top": number("İlk N", 20, 1, 500), "max-names": number("Sembol sınırı", 400, 10, 1000)}},
    "train_meta": {"label": "Meta model eğitimi (iki bölme)", "group": "ml", "module": "research.train_meta",
                   "fields": {"csv": _DATASET, "model": _MODELS}},
    "train_neural": {"label": "Nöral model eğitimi", "group": "ml", "module": "research.train_neural",
                     "fields": {"dataset": _DATASET, "tier": choice("Mimari", ["deep_mlp", "quant_transformer", "quant_transformer_xl"]),
                                "folds": _FOLDS, "max-epochs": number("Azami epoch", 40, 1, 500),
                                "device": choice("Hesaplama", ["cpu", "auto", "cuda"])}, "json": True},
    "option_archive": {"label": "OPRA günlük opsiyon arşivi", "group": "data", "module": "nihai.data.option_archive",
                       "fields": {"symbols": _SYMBOLS, "expiries": number("Vade sayısı", 1, 1, 12)}},
    "float": {"label": "Float / devir hızı önbelleği", "group": "data", "module": "nihai.data.market_structure",
              "fields": {"warm": _SYMBOLS}},
    "edgar": {"label": "SEC EDGAR bilanço güncelleme", "group": "data", "module": "nihai.data.edgar_facts",
              "fields": {"symbols": {**_SYMBOLS, "positional": True}}, "args": ["--refresh"]},
    "factors": {"label": "Fama–French faktörleri ve momentum", "group": "data", "module": "nihai.data.factors",
                "fields": {}, "args": ["--momentum"]},
    "macro": {"label": "FRED makro göstergeleri", "group": "data", "module": "nihai.data.macro_regime",
              "fields": {"since": _DATE}},
    "macro_etf": {"label": "Makro ETF vekillerini güncelle", "group": "data", "module": "nihai.data.macro_regime",
        "fields": {"since": _DATE}, "args": ["--etf-only"],
        "note": "Alpaca: GLD/USO, CPER/GLD, HYG/LQD ve BIL geçmiş toplam getirisi. Temettü düzeltmesi gerekli."},
    "fda_calendar": {"label": "FDA sponsor onay geçmişi", "group": "data", "fields": {
        "sponsor": {"label": "FDA sponsor adı (borsa sembolü değil)", "type": "text", "default": "MODERNA", "max_length": 120},
        "since": _DATE, "until": {**_DATE, "label": "Bitiş", "default": date.today().isoformat()},
        "force": {**_BOOL, "label": "Önbelleği yenile"}},
        "note": "Geçmiş onay kayıtlarını gösterir; gelecekteki PDUFA tarihleri bu API'de yoktur."},
    "pdufa_calendar": {"label": "PDUFA takvimi (SEC 8-K/6-K bültenleri)", "group": "data", "fields": {
        "since": {**_DATE, "default": (date.today() - timedelta(days=400)).isoformat()},
        "until": {**_DATE, "label": "Bitiş", "default": date.today().isoformat()},
        "force": {**_BOOL, "label": "Önbelleği yenile"}},
        "note": "Tarih ŞİRKET BEYANIDIR, FDA takvimi değildir; uzatma/erken onay ancak sonraki dosyalamada görünür."},
    "wiki_trends": {"label": "Wikipedia günlük ilgi ölçümü", "group": "data", "fields": {
        "article": {"label": "Wikipedia sayfa adı", "type": "text", "default": "Apple_Inc.", "max_length": 200},
        "since": {**_DATE, "default": "2025-01-01"},
        "until": {**_DATE, "label": "Bitiş", "default": date.today().isoformat()},
        "force": {**_BOOL, "label": "Önbelleği yenile"}},
        "note": "İngilizce Wikipedia'da insan görüntülenmeleri; yalnız tamamlanmış UTC günleri."},
    "dolt": {"label": "DoltHub toplu veri", "group": "data", "fields": {
        "table": choice("Tablo", ["volatility_history", "option_chain"]),
        "force": {**_BOOL, "label": "Taze olsa da yeniden indir"}},
        "note": "Opsiyon zinciri yaklaşık 10 GB; OneDrive dışındaki dolt_cache dizinine yazılır."},
}


class ToolInput(Input):
    tool: str
    parameters: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def check(self):
        if self.tool not in TOOLS:
            raise ValueError("Bilinmeyen araç.")
        fields = TOOLS[self.tool]["fields"]
        if set(self.parameters) - set(fields):
            raise ValueError("Aracın tanımında olmayan parametre.")
        values = {}
        for key, spec in fields.items():
            v = self.parameters.get(key, spec["default"])
            typ = spec["type"]
            if typ == "choice" and v not in spec["choices"]:
                raise ValueError(f"{key}: geçersiz seçim.")
            if typ == "number" and (isinstance(v, bool) or not isinstance(v, int) or not spec["min"] <= v <= spec["max"]):
                raise ValueError(f"{key}: {spec['min']}–{spec['max']} arasında tamsayı gerekli.")
            if typ == "bool" and not isinstance(v, bool):
                raise ValueError(f"{key}: true/false gerekli.")
            if typ == "text":
                if (not isinstance(v, str) or not v.strip() or len(v) > spec["max_length"]
                        or any(ord(c) < 32 for c in v)):
                    raise ValueError(f"{key}: geçerli metin gerekli (azami {spec['max_length']} karakter).")
                v = v.strip()
            if typ == "dataset":
                dataset_path(v)
            if typ == "date":
                if not isinstance(v, str):
                    raise ValueError(f"{key}: ISO tarih gerekli.")
                v = date.fromisoformat(v).isoformat()
            if typ == "symbols":
                if not isinstance(v, str) or len(v) > 400:
                    raise ValueError("En fazla 20 sembol girin.")
                syms = normalize_symbols(v.split(","))
                if len(syms) > 20:
                    raise ValueError("En fazla 20 sembol girin.")
                v = ",".join(syms)
            values[key] = v
        self.parameters = values
        if self.tool in ("fda_calendar", "wiki_trends", "pdufa_calendar") and values["since"] > values["until"]:
            raise ValueError("Başlangıç bitişten sonra olamaz.")
        return self


@router.get("/api/lab/catalog")
def lab_catalog():
    from nihai.signals.strategy import BASE_STRATEGY_MODES
    files = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.name)
    return {"tools": TOOLS, "datasets": [{"name": p.name, "bytes": p.stat().st_size} for p in files],
            "strategies": [*BASE_STRATEGY_MODES, "ensemble", "confluence"]}


@router.post("/api/research/run", status_code=202)
@router.post("/api/data/refresh", status_code=202)
def tool_run(req: ToolInput):
    return jobs.start("tool", req.model_dump())


@router.get("/api/research/feature-importance")
def feature_importance():
    for item in job_list()["jobs"]:
        if item["parameters"].get("tool") == "purged_cv" and item["status"] == "completed":
            return {"job_id": item["job_id"], **jobs.report(item["job_id"])}
    return {"status": "unmeasured", "reason": "Henüz tamamlanmış Purged CV işi yok."}


class RetrainInput(Input):
    model_type: Literal["logistic", "gbm", "both"] = "both"
    min_samples: int = Field(default=300, ge=300, le=1000000)
    folds: int = Field(default=5, ge=2, le=10)
    promote_flag: bool = False
    confirmed: bool = False

    @model_validator(mode="after")
    def promotion(self):
        if self.promote_flag and not self.confirmed:
            raise ValueError("Kanıt kapılarını geçen modelin kaydedilmesi için onay gerekli.")
        return self


@router.post("/api/ml/retrain", status_code=202)
def ml_retrain(req: RetrainInput):
    return jobs.start("retrain", req.model_dump())


@router.get("/api/ml/training-status")
def training_status():
    return {"jobs": [j for j in job_list()["jobs"] if j["kind"] == "retrain" or
                     j["parameters"].get("tool") in ("train_meta", "train_neural")]}


class ShadowInput(Input):
    interval: Literal["1m", "5m", "15m", "1h", "1d"] = Field(default=HTF_INTERVAL, validate_default=True)


class ActivateInput(Input):
    review_job: str = Field(pattern=r"^[0-9a-f]{32}$")
    confirmed: bool = False


@router.post("/api/ml/promote-shadow", status_code=202)
def shadow_review(req: ShadowInput):
    return jobs.start("shadow_review", req.model_dump())


@router.get("/api/ml/promotion-status")
def promotion_status():
    from nihai.signals.model_promotion import status
    return {**status(), "default_interval": HTF_INTERVAL}


@router.post("/api/ml/activate-shadow", status_code=202)
def activate_shadow(req: ActivateInput):
    if not req.confirmed:
        raise HTTPException(422, "Canlı terfi için onay gerekli.")
    info = jobs.status(req.review_job)
    if info["kind"] != "shadow_review" or info["status"] != "completed":
        raise HTTPException(409, "Tamamlanmış terfi incelemesi seçin.")
    return jobs.start("shadow_activate", req.model_dump())


def execute_tool(kind: str, params: dict, path: Path):
    out_path = path / "tool_result.json"
    if kind == "retrain":
        retrain = RetrainInput(**params)
        args = ["--model", retrain.model_type, "--min-rows", str(retrain.min_samples), "--folds", str(retrain.folds),
                "--json", str(out_path)]
        if retrain.promote_flag:
            args.append("--promote")
        _cli("research.live_retrain", args)
    elif kind == "shadow_review":
        from nihai.signals.model_promotion import review
        return review(ShadowInput(**params).interval)
    elif kind == "shadow_activate":
        from nihai.signals.model_promotion import activate
        activation = ActivateInput(**params)
        if not activation.confirmed:
            raise ValueError("Canlı terfi için onay gerekli.")
        if jobs.status(activation.review_job)["kind"] != "shadow_review":
            raise ValueError("Terfi incelemesi gerekli.")
        return activate(jobs.report(activation.review_job)["result"])
    elif kind == "tool":
        req = ToolInput(**params)
        spec = TOOLS[req.tool]
        if req.tool == "ablation":
            from research.ablation import main
            return {"ablations": main()}
        if req.tool in ("fda_calendar", "wiki_trends", "pdufa_calendar"):
            from nihai.data import fda_calendar, wiki_trends
            readers: dict[str, Callable[..., dict | None]] = {
                "fda_calendar": fda_calendar.calendar, "wiki_trends": wiki_trends.report,
                "pdufa_calendar": fda_calendar.pdufa_calendar}
            reader = readers[req.tool]
            public_report = reader(**req.parameters, fetch=True)
            if public_report is None:
                raise ValueError("Dış veri ölçülemedi; girdi ve sağlayıcı hatasını iş günlüğünden inceleyin.")
            return public_report
        if req.tool == "dolt":
            from nihai.data import dolt_options
            if req.parameters["table"] == "option_chain":
                dolt_options.DOLT_CACHE_DIR = Path.home() / "dolt_cache"
            started = time.time()
            result = dolt_options.bulk_csv(**req.parameters)
            if result is None:
                raise ValueError("Dolt verisi alınamadı; logdaki sağlayıcı hatasını inceleyin.")
            modified = result.stat().st_mtime
            age_days = (time.time() - modified) / 86400
            if age_days >= 7 or (req.parameters["force"] and modified < started):
                raise ValueError("Dolt yenilenemedi; eski önbellek korunuyor. Sağlayıcı hatasını logdan inceleyin.")
            return {"file": result.name, "bytes": result.stat().st_size, "cache_directory": str(result.parent),
                    "cache_age_days": round(age_days, 3), "downloaded": modified >= started}
        args = list(spec.get("args", []))
        for key, value in req.parameters.items():
            field = spec["fields"][key]
            if field["type"] == "bool":
                if value:
                    args.append("--" + key)
            elif field.get("positional"):
                args.extend(value.split(","))
            else:
                args.extend(["--" + key, str(dataset_path(value)) if field["type"] == "dataset" else str(value)])
        if spec.get("json"):
            args += ["--json", str(out_path)]
        if req.tool == "train_neural":
            # Yeni artefakt iş dizininde incelenir; registry/canlı model seçimi değişmez.
            args += ["--save", str(path / "neural_candidate"), "--register", "--registry-dir", str(path / "models"), "--progress"]
        _cli(spec["module"], args)
        if not out_path.exists():
            return {"tool": req.tool, "note": "Araç süreci tamamlandı. Veri önkoşulları ve ölçülebilen sonuçlar iş günlüğündedir.",
                    "artifacts": [p.name for p in path.glob("neural_candidate.*")]}
    else:
        from nihai.web.api_operations import execute_operation
        return execute_operation(kind, params, path)
    if not out_path.exists():
        raise ValueError("Araç ölçüm raporu üretmedi; veri önkoşullarını iş günlüğünden kontrol edin.")
    return json.loads(out_path.read_text(encoding="utf-8"))
