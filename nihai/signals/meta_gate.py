"""
meta_gate.py — Canlı meta-labeling kapısı (ikincil model çıkarımı).

Eğitilmiş modeli (train_meta.py çıktısı) yükler ve bir sinyalin kazanma olasılığını
tahmin eder. generate_signal bunu 'düşük-olasılıkları ele' kapısı olarak kullanır.

Zarif degrade: model dosyası yoksa veya META_GATE kapalıysa kapı PASİF — predict None
döner, hiçbir sinyal elenmez (mevcut davranış birebir korunur). Bu, doğrulanmamış bir
kaldıracın sisteme sessizce sızmasını önler (EDGE_FINDINGS opt-in disiplini).

Gölge (shadow) modu (META_GATE=shadow): kapı SKORLAR + metadata/reason'a yazar ama
sinyali ASLA bastırmaz — canlı kanıt döngüsü (prediction_tracker pwin logu → gate_sim/
pwin_calibration) kapı fiilen kapalıyken de yaşar. Canlı gate_sim GATE_HURTS derken
enforce açılmamalı; kip ayrımı strategy.py'dedir, bu modül kip-agnostiktir.

Öncelik: data/meta_model.pkl (GBM) → data/meta_model.json (lojistik) → yok (pasif).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import threading
from typing import Any

import numpy as np
from loguru import logger

from nihai.config import DATA_DIR
from nihai.signals.meta_features import FEATURE_NAMES, extract_features, features_to_vector

# ---------------------------------------------------------------------------
# BULGU-44: SESSİZ degrade ≠ ZARİF degrade.
#
# `probability`/`explain` hatayı yutup None döndürür, `strategy.generate_signal` de
# kapı çağrısını geniş bir `except Exception: pass` ile sarar. Gerekçe doğrudur (kapı
# hatası sinyali BLOKE etmemeli) ama bedeli ölçülmemişti: kapı KONFİGÜRE AÇIK ve model
# YÜKLÜ iken her çağrıda susarsa, sistem dışarıdan "kapı kapalı"dan AYIRT EDİLEMEZ.
# Tur-43'te tam bu oldu: `probability` imzası `shared=` ile genişleyince test ikizleri
# TypeError verdi ve kapı sessizce pasifleşti — testler yakaladı, ÜRETİM yakalamazdı.
# Zararı somut: gölge kipin varlık nedeni canlı kanıt döngüsüdür (prediction_tracker
# p_win logu → gate_sim). Kapı susarsa o log BOŞ birikir ve ACI DERS #5'in "gerçek
# darboğaz veri" tespiti bir tur daha ertelenir; kimse nedenini bilmez.
#
# Defter MODÜL düzeyindedir: `reload_gate()` örneği atar, hata geçmişi atılmamalı;
# ayrıca örneğe hiç ulaşamayan hatalar (import/kurulum) da buradan geçebilsin.
_LEDGER: dict = {"calls": 0, "failures": 0, "last": None, "where": None}
_LOG_EVERY = 100          # ilk hata + her 100'de bir logla (backtest bar başına çağırır)


def note_call() -> None:
    """Kapıya AKTİF hâlde bir soru soruldu (model yokken sayılmaz — o hata değildir)."""
    _LEDGER["calls"] += 1


def note_error(exc: BaseException, nerede: str) -> None:
    """Yutulan kapı hatasını SAY + ilkini (ve her 100.'yü) UYARI olarak logla."""
    _LEDGER["failures"] += 1
    _LEDGER["last"] = f"{type(exc).__name__}: {exc}"[:200]
    _LEDGER["where"] = nerede
    n = _LEDGER["failures"]
    if n == 1 or n % _LOG_EVERY == 0:
        logger.warning(f"[meta_gate] {nerede}: {_LEDGER['last']} — kapı SESSİZCE None "
                       f"döndürüyor (toplam {n} hata / {_LEDGER['calls']} soru). "
                       "P(win) üretilmiyor; gölge kanıt döngüsü BOŞ birikiyor.")


def gate_health() -> dict:
    """Kapının GERÇEK durumu. `active` yalnız MODEL YÜKLENDİ demektir — ÜRETİYOR demek
    DEĞİLDİR; iki kavramı ayırmak bu fonksiyonun tek varlık nedenidir (Tur-31 dersi:
    ölçüm yokluğu ölçüm gibi sunulamaz).

    durum: MODEL_YOK · HENUZ_SORULMADI · SAGLIKLI · KISMI_HATA · **SUSUYOR**
    """
    g = get_gate()
    c, f = int(_LEDGER["calls"]), int(_LEDGER["failures"])
    if not g.active:
        durum = "MODEL_YOK"
    elif c == 0:
        durum = "HENUZ_SORULMADI"
    elif f == 0:
        durum = "SAGLIKLI"
    elif f >= c:
        durum = "SUSUYOR"          # sorulan HER soruda hata → kapı fiilen ölü
    else:
        durum = "KISMI_HATA"
    return {"durum": durum, "active": g.active, "kind": g.kind,
            "calls": c, "failures": f,
            "failure_rate": round(f / c, 4) if c else 0.0,
            "last_error": _LEDGER["last"], "where": _LEDGER["where"]}


def reset_health() -> None:
    """Defteri sıfırla (test/operatör). Üretimde çağrılmaz."""
    _LEDGER.update(calls=0, failures=0, last=None, where=None)


class _MetaGate:
    def __init__(self):
        self.kind: str | None = None
        self.threshold: float = 0.5
        self._features: list[str] = FEATURE_NAMES
        self._log = None      # lojistik: (w, mu, sd)
        self._clf = None      # gbm: sklearn estimator
        self.model_id: str | None = None
        self._load()

    def _load(self) -> None:
        pkl = DATA_DIR / "meta_model.pkl"
        js = DATA_DIR / "meta_model.json"
        try:
            if pkl.exists():
                raw = pkl.read_bytes()
                d = pickle.loads(raw)
                self._clf = d["clf"]
                self._features = d.get("features", FEATURE_NAMES)
                self.threshold = float(d.get("gate_threshold", 0.5))
                self.kind = "gbm"
                self.model_id = hashlib.sha256(raw).hexdigest()
                return
            if js.exists():
                raw = js.read_bytes()
                d = json.loads(raw)
                self._log = (np.array(d["w"]), np.array(d["mu"]), np.array(d["sd"]))
                self._features = d.get("features", FEATURE_NAMES)
                self.threshold = float(d.get("gate_threshold", 0.5))
                self.kind = "logistic"
                self.model_id = hashlib.sha256(raw).hexdigest()
        except Exception:
            self.kind = None  # bozuk model → pasif

    @property
    def active(self) -> bool:
        return self.kind is not None

    def _prob_from_vector(self, x) -> float:
        """HAZIR özellik vektöründen P(win) — model matematiğinin TEK kaynağı (Tur-43).

        `probability` ve `explain` bu fonksiyonu paylaşır. Eskiden `explain` önce kendi
        `extract_features`'ını koşuyor, sonra `self.probability(df_slice, signal)`
        çağırarak AYNI çıkarımı İKİNCİ KEZ yaptırıyordu — panonun her sembolde ödediği
        iki kat maliyet (ölçüldü: 4000 barda 15,99 ms ↔ 7,63 ms). Matematik değişmedi."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        if self.kind == "gbm":
            return float(self._clf.predict_proba(x)[:, 1][0])
        w, mu, sd = self._log
        xs = (x - mu) / sd
        xb = np.hstack([np.ones((1, 1)), xs])
        z = float((xb @ w).ravel()[0])  # (1,20)@(20,)→(1,); numpy 2.x float() 0-d ister
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def probability(self, df_slice, signal, *, shared=None) -> float | None:
        """Sinyalin kazanma olasılığı [0,1]; model yoksa None.

        shared: opsiyonel `strategy._SharedIndicators`/`_SharedSlice` — çağıran
            göstergeleri zaten hesapladıysa yeniden hesaplanmaz (Tur-43/BULGU-43-1).
            Görünüm dilimle hizalı değilse `extract_features` onu yok sayar."""
        if not self.active:
            return None
        note_call()
        try:
            feats = extract_features(df_slice, signal, shared=shared)
            # Vektör, modelle kaydedilmiş eğitim-anı sırasında kurulur (train/serve hizası).
            return self._prob_from_vector(features_to_vector(feats, self._features))
        except Exception as exc:
            note_error(exc, "probability")   # BULGU-44: yut ama SESSİZ yutma
            return None

    def explain(self, df_slice, signal, *, top_k: int = 8, shared=None) -> dict | None:
        """P(win) için SHAP-tarzı özellik-katkısı kartı (Explainable AI, D1).

        Lojistik model için katkılar TAM (yaklaşık değil): standardize özellik × ağırlık =
        log-odds katkısı. Lineer modelde bu, tam olarak SHAP değerine eşittir (baz = sabit
        terim). GBM için: shap kuruluysa TreeExplainer; değilse global feature_importances_
        (örnek-bazlı DEĞİL, bayrakla işaretlenir).

        Dönüş: {kind, prob, base_prob, contributions:[{feature,value,effect}], note} | None.
        DÜRÜSTLÜK: model OOS-AUC'si zayıf (~0.5-0.6, EDGE_FINDINGS §16) → bu kart 'modelin
        ne düşündüğünü' açıklar, GELECEĞİ değil; kenar iddiası taşımaz."""
        if not self.active:
            return None
        try:
            feats = extract_features(df_slice, signal, shared=shared)
            names = list(self._features)
            # Değerler ve etiketler (names) AYNI sırada olmalı → self._features geç.
            x = np.array(features_to_vector(feats, self._features), dtype=float)
            try:
                prob = self._prob_from_vector(x)   # Tur-43: ikinci extract_features YOK
            except Exception:
                prob = None      # eski sözleşme: model hatası kartı düşürmez, prob=None
            if self.kind == "logistic":
                w, mu, sd = self._log
                xs = (x - mu) / sd
                effects = w[1:] * xs                  # her özelliğin log-odds katkısı
                base = float(w[0])
                base_prob = 1.0 / (1.0 + np.exp(-np.clip(base, -30, 30)))
                contribs: list[dict[str, Any]] = [{"feature": names[i], "value": round(float(x[i]), 4),
                             "effect": round(float(effects[i]), 4)}
                            for i in range(len(names))]
                contribs.sort(key=lambda c: abs(c["effect"]), reverse=True)
                return {"kind": "logistic", "prob": prob, "base_prob": round(base_prob, 4),
                        "contributions": contribs[:top_k], "note": "exact"}
            # GBM
            xr = x.reshape(1, -1)
            try:
                import shap
                expl = shap.TreeExplainer(self._clf)
                sv = expl.shap_values(xr)
                vals = sv[1][0] if isinstance(sv, list) else sv[0]
                contribs = [{"feature": names[i], "value": round(float(x[i]), 4),
                             "effect": round(float(vals[i]), 4)} for i in range(len(names))]
                contribs.sort(key=lambda c: abs(c["effect"]), reverse=True)
                return {"kind": "gbm", "prob": prob, "base_prob": None,
                        "contributions": contribs[:top_k], "note": "shap"}
            except Exception:
                imp = getattr(self._clf, "feature_importances_", None)
                if imp is None:
                    return {"kind": "gbm", "prob": prob, "base_prob": None,
                            "contributions": [], "note": "no_shap"}
                contribs = [{"feature": names[i], "value": round(float(x[i]), 4),
                             "effect": round(float(imp[i]), 4)} for i in range(len(names))]
                contribs.sort(key=lambda c: c["effect"], reverse=True)
                return {"kind": "gbm", "prob": prob, "base_prob": None,
                        "contributions": contribs[:top_k], "note": "global_importance"}
        except Exception as exc:
            note_error(exc, "explain")       # BULGU-44
            return None


_GATE: _MetaGate | None = None
_GATE_STAMP = None
_GATE_LOCK = threading.RLock()


def get_gate() -> _MetaGate:
    global _GATE, _GATE_STAMP
    with _GATE_LOCK:
        stamp: list[tuple[str, int | None, int | None]] = []
        for name in ("meta_model.pkl", "meta_model.json"):
            path = DATA_DIR / name
            try:
                st = path.stat()
                stamp.append((str(path), st.st_mtime_ns, st.st_size))
            except FileNotFoundError:
                stamp.append((str(path), None, None))
        if _GATE is None or stamp != _GATE_STAMP:
            _GATE = _MetaGate()
            _GATE_STAMP = stamp
        return _GATE


def reload_gate() -> None:
    """Model yeniden eğitildiğinde canlı süreçte tazele."""
    global _GATE
    _GATE = None
