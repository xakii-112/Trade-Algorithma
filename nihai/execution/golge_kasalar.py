"""
golge_kasalar.py — Gölgede/danışman olarak çalışan 5 karar mekanizmasına AYRI SANAL KASA (Tur-102).

Soru: bu mekanizmalar karar YETKİSİ alsaydı canlı veride ne yapardı? Bugün hiçbiri canlı kararı
değiştirmez (META_GATE=shadow, AI_LIVE_MODE=advise, SHADOW_BRIDGE=advise, nöral gölge yalnız log,
CONFIDENCE/PROB_SIZING kapalı). Bu modül her birine `GOLGE_KASA_BAKIYE` (1.000.000 $) büyüklüğünde
AYRI bir sanal kasa verir ve mekanizmayı o kasada ENFORCE kipinde çalıştırır. Canlı portföye,
canlı sinyale, emir metnine ve üretim defterlerine (trades.csv / trade_log / decision_log) DOKUNMAZ.

ADAY AKIŞI — neden canlı stratejinin (quant_trend) adayları DEĞİL: beş mekanizma da 1 dk TA
adaylarında eğitildi/ölçüldü ve quant adayı için KAPSAM DIŞI ilan edildi (BULGU-97-4 meta kapı,
BULGU-99-1 AI köprüsü). Ayrıca quant_trend 1m barda hep `QUANT_GUNLUK_BAR_GEREKLI` döndüğü için
tahmin servisi 2026-09-27'den beri hiç aday üretmiyor (BULGU-102-1). Kasalar `GOLGE_KASA_STRATEJI`
(varsayılan event_breakout) ile tahmin servisinin 1m barlarında üretilen AYNI aday akışını görür;
mekanizmaların kendi karar dalları hiç karışmaz (meta enforce'un elediği aday bile geri alınır).
Altıncı kasa `kontrol` mekanizmasızdır — beş kasanın farkı ona göre okunur, çünkü piyasanın kendisi
bütün kasaları birlikte oynatır. TEK DEĞİŞKEN mekanizmadır.

KASA KURALLARI — her biri mekanizmanın KENDİ enforce sözleşmesi, eşik UYDURULMAZ:
  kontrol        — her aday, tam risk.
  meta_kapi      — meta p_win < eşik → giriş yok (`strategy.py` enforce dalı; eşik META_GATE_THRESHOLD).
  ai_karar       — iç AI verdikti adayın TERS yönündeyse VETO; ABSTAIN ve aynı yön GEÇER, boyut posterior
                   rampasıyla (`enforce_boyut` TEK kaynak). Tur-105'ten beri canlı enforce'tan (`ai_advisor.advise`:
                   ABSTAIN = veto) AYRIDIR: o sözleşmede kasa 5.531 adayın hiçbirine girmedi (BULGU-105-1).
  golge_kopru    — köprünün terfi defteri BOŞ olduğundan (`shadow_bridge.PROMOTED`) enforce'u kontrolün
                   kopyası olurdu. Kasa, köprünün KENDİ terfi hattının bekleyen iki adayını uygular — Tur-88'de
                   (BULGU-88-5) DONDURULMUŞ eşiklerle, bugün seçilmeden: `conf_bar` ≥ 0,60 ve `value_area_pos`
                   ≥ 1,0 (`KOPRU_KURALLARI`); ayrıca gözlemevinin veri KARANTİNASI veto eder. Evren farkı:
                   kurallar yön tahmini satırlarında ölçüldü, burada işlem adayına uygulanır.
  noral          — eşik sabit DEĞİL: modelin KENDİ son 200 ölçümünün medyanı (nedensel; getiriye bakmaz). 282M
                   modelin p'si 0,484-0,487'e sıkışık (Tur-93) → sabit 0,50 her şeyi, taban oran (0,481) hiçbir
                   şeyi elemezdi; medyan, modelin SIRALAMASINI sınar. 30 ölçümden önce ve ölçüm yoksa GİRMEZ.
  olasilik_boyut — boyut = watchlist güven çarpanı × prob_to_size_mult(meta p_win); `main.py`'daki
                   CONFIDENCE_SIZING + PROB_SIZING dallarının İKİSİ de açıkmış gibi. Veto etmez.
  Ölçüm yoksa (model pasif/hata) aday GEÇER — canlı enforce dallarının fail-soft sözleşmesi (nöral kasa
  hariç: canlı enforce'u yok, model görüş bildirmediyse girmez); ölçümsüz aday sayısı `kararlar.csv`de görünür.

MOTOR — backtest ile AYNI sözleşme: karar bar T kapanışında, giriş T'den sonraki ilk barın AÇILIŞINDA;
her yeni bar önce açık pozisyonları günceller (stop/hedef/izleyen stop/gap), sonra bekleyen girişi
doldurur. Boyut/tavan/fren `RiskManager`, çıkış/kayma/komisyon `PositionManager(record_logs=False)`.
Adet = min(risk kuralı, özsermaye × `GOLGE_KASA_POZ_PAY` (0,10) / fiyat) × kasa çarpanı (`kasa_adedi`, BULGU-102-7:
1m stopu dar olduğundan risk kuralı hep alım gücü tavanına çarpar; çarpan tavandan SONRA). Kasa başına sembol başına en fazla
BİR pozisyon (bekleyen emir dahil). Yalnız KAPANMIŞ bar işlenir
(başlangıcı + 1 dk ≤ şimdi); bekleyen emir 5 dk içinde dolmazsa (seans sonu) iptal edilir.

DOSYALAR (`GOLGE_KASA_DIR`, varsayılan data/golge_kasalar/):
  <kasa>.json   — StateStore biçiminde portföy (özsermaye, açık pozisyonlar); yeniden başlatmada sürer.
  islemler.csv  — kapanan işlemler.   kararlar.csv — aday başına mekanizma ölçüleri + altı kasanın kararı.
  motor.json    — sembol başına son işlenen bar (yeniden başlatmada kaçan barlar df'ten işlenir).

.env:
    GOLGE_KASA=on|off                 # varsayılan on (test sürecinde conftest kapatır)
    GOLGE_KASA_STRATEJI=event_breakout
    GOLGE_KASA_BAKIYE=1000000
Rapor (salt okuma, koşan süreçle çakışmaz): python -m nihai.execution.golge_kasalar
Ayrı koşucu (canlı ana sürece dokunmadan): python -m nihai.execution.golge_kasalar --kos
    Kasa dizinine TEK süreç yazar (`motor.lock`); ana süreç ve koşucu birlikte açıksa ilk alan yazar.
"""
from __future__ import annotations

import csv
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from nihai.atomic_io import atomic_write_text
from nihai.config import (
    CONFIG,
    DATA_DIR,
    PROB_SIZING_FLOOR,
    PROB_SIZING_P_HI,
    PROB_SIZING_P_LO,
    get_sector,
)
from nihai.data.market_calendar import can_enter_new_position
from nihai.execution.state import StateStore
from nihai.risk.position_manager import PositionManager
from nihai.risk.prob_sizing import prob_to_size_mult
from nihai.risk.risk_manager import RiskManager
from nihai.signals.strategy import Signal, SignalAction, generate_signal, reanchor_to_live

KASALAR = ("kontrol", "meta_kapi", "ai_karar", "golge_kopru", "noral", "olasilik_boyut")
ACIKLAMA = {
    "kontrol": "Mekanizmasız taban — her aday, tam risk",
    "meta_kapi": "Meta kapı (ML P(win)) — eşik altı aday girmez",
    "ai_karar": "İç AI karar motoru (Bayes) — ters yön veto, ABSTAIN geçer, posteriorla boyut",
    "golge_kopru": "Gölge köprüsü — karantina + Tur-88'de dondurulmuş 2 terfi adayı (conf_bar, value_area_pos)",
    "noral": "Nöral gölge p_win — modelin kendi son 200 ölçümünün medyanı altı girmez",
    "olasilik_boyut": "Olasılık + güven boyutlama — veto yok, boyut küçülür",
}
# Tur-88 ön-kaydı (BULGU-88-5, 2026-09-22): köprü terfi hattının bekleyen adayları, eşik ve yön ("tut: yüksek")
# DONDURULMUŞTUR. Değiştirmek yeni bir ön-kayıttır; `shadow_bridge.RULES`e bir kural terfi ederse bu kasa güncellenir.
KOPRU_KURALLARI = {"conf_bar": 0.60, "value_area_pos": 1.0}
NORAL_PENCERE, NORAL_ISINMA = 200, 30
_KAPALI = frozenset({"0", "off", "false", "no", "kapali", "kapalı"})
_BAYAT = timedelta(minutes=5)
_ISLEM_KOLON = ("kasa", "sembol", "taraf", "adet", "giris_zamani", "giris", "cikis_zamani", "cikis",
                "tetik", "kz", "ozsermaye")
_KARAR_KOLON = ("kayit", "bar", "sembol", "taraf", "fiyat", "stop", "hedef", "meta_p", "noral_p", "noral_esik",
                "ai_action", "ai_post", "ai_veto", "karantina", "conf_bar", "value_area_pos", "guven") + KASALAR


def etkin() -> bool:
    """`GOLGE_KASA` çağrı anında okunur; boş/tanınmayan değer → açık (ölçüm yolu, karar-dışı)."""
    return (os.getenv("GOLGE_KASA") or "on").strip().lower() not in _KAPALI


def _envf(ad: str, varsayilan: float) -> float:
    try:
        return float(os.getenv(ad) or varsayilan)
    except (TypeError, ValueError):
        return varsayilan


def strateji() -> str:
    return (os.getenv("GOLGE_KASA_STRATEJI") or "event_breakout").strip().lower()


def dizin() -> Path:
    return Path(os.getenv("GOLGE_KASA_DIR") or DATA_DIR / "golge_kasalar")


# --------------------------------------------------------------------------- karar çekirdeği (SAF)
@dataclass(frozen=True)
class Olcum:
    """Bir aday için mekanizma çıktıları — aday başına BİR kez ölçülür, altı kasa aynı sayıyı görür."""

    meta_p: float | None = None
    noral_p: float | None = None
    ai_action: str | None = None
    ai_post: float | None = None
    ai_veto: int | None = None
    karantina: bool | None = None      # None = gözlemevi ölçemedi
    conf_bar: float | None = None      # tahmin servisinin bar kapanışı güveni (deftere yazılanla aynı)
    value_area_pos: float | None = None
    guven: float = 1.0                 # watchlist risk_mult (CONFIDENCE_SIZING)


def karar(kasa: str, o: Olcum, *, meta_esik: float, noral_esik: float | None) -> tuple[bool, float, str]:
    """SAF: kasanın bu aday için hükmü → (gir, boyut çarpanı ∈ (0,1], gerekçe)."""
    if kasa == "kontrol":
        return True, 1.0, "taban"
    if kasa == "meta_kapi":
        if o.meta_p is None:
            return True, 1.0, "meta_p yok (geçer)"
        if o.meta_p < meta_esik:
            return False, 0.0, f"meta_p {o.meta_p:.3f} < {meta_esik:.2f}"
        return True, 1.0, f"meta_p {o.meta_p:.3f}"
    if kasa == "ai_karar":
        if o.ai_veto is None:
            return True, 1.0, "AI verdikti yok (geçer)"
        # Tur-105: yalnız TERS yön veto eder; ABSTAIN geçer ve boyutu posterior rampası verir. `ai_veto` canlı
        # enforce'un cevabıdır (ABSTAIN dahil) ve deftere öyle yazılır — kasanın hükmü ondan ayrıdır.
        if o.ai_veto and o.ai_action != "ABSTAIN":
            return False, 0.0, f"AI {o.ai_action} (ters yön, veto)"
        from nihai.execution.ai_advisor import enforce_boyut
        return True, enforce_boyut(float(o.ai_post or 0.0)), f"AI {o.ai_action} post={o.ai_post}"
    if kasa == "golge_kopru":
        if o.karantina:
            return False, 0.0, "veri karantinası"
        for ad, esik in KOPRU_KURALLARI.items():
            v = getattr(o, ad)
            if v is not None and v < esik:
                return False, 0.0, f"{ad} {v:.2f} < {esik:.2f}"
        eksik = [ad for ad in KOPRU_KURALLARI if getattr(o, ad) is None]
        return True, 1.0, f"ölçüm yok (geçer): {','.join(eksik)}" if eksik else "kurallar geçti"
    if kasa == "noral":
        if o.noral_p is None:
            return False, 0.0, "nöral p yok"
        if noral_esik is None:
            return False, 0.0, f"ısınma (<{NORAL_ISINMA} ölçüm)"
        if o.noral_p < noral_esik:
            return False, 0.0, f"nöral p {o.noral_p:.3f} < {noral_esik:.2f}"
        return True, 1.0, f"nöral p {o.noral_p:.3f}"
    if kasa == "olasilik_boyut":
        m = o.guven * prob_to_size_mult(o.meta_p, p_lo=PROB_SIZING_P_LO, p_hi=PROB_SIZING_P_HI,
                                        floor=PROB_SIZING_FLOOR)
        return True, m, f"boyut×{m:.2f}"
    raise ValueError(f"bilinmeyen kasa: {kasa}")


def poz_payi() -> float:
    """Kasa başına tek pozisyonun nominal tavanı (özsermaye payı); bozuk/aralık dışı değer → 0,10."""
    v = _envf("GOLGE_KASA_POZ_PAY", 0.10)
    return v if 0.0 < v <= 1.0 else 0.10


def kasa_adedi(risk_adet: int, fiyat: float, ozsermaye: float, carpan: float, pay: float) -> int:
    """SAF: kasa pozisyon adedi = min(risk kuralının adedi, özsermaye × pay / fiyat) × çarpan.

    BULGU-102-7: 1m TA stopu fiyatın ~%0,1-0,3'ü → %1,5 risk kuralı özsermayenin 10-50 katı nominal ister ve adet alım
    gücü tavanında kesilir. Canlıdaki sıra (çarpan riske, sonra tavan) böyle bir tavanda çarpanı ETKİSİZ kılar ve kasanın
    ilk adayı bütün alım gücünü alır (canlı ölçüm: IBM 9.104 hisse ≈ 2,0 M$, sonraki adaylar 1-4 hisse; olasılık kasası
    ×0,25'le AYNI adet). Pozisyon payı sermayeyi adaylara yayar, çarpan tavandan SONRA uygulanır → boyut mekanizması
    kasada gerçekten boyut değiştirir."""
    if fiyat <= 0 or ozsermaye <= 0:
        return 0
    return int(min(risk_adet, int(ozsermaye * pay // fiyat)) * carpan)


def meta_esik() -> float:
    return float(CONFIG.strategy.meta_gate_threshold or 0.45)


# --------------------------------------------------------------------------- aday + ölçüm
def aday(symbol: str, df: pd.DataFrame, shared=None) -> Signal | None:
    """Kasaların ortak aday sinyali (LONG/SHORT) — yoksa None.

    `live_interval` VERİLMEZ: terfi politikası (model_promotion) akışı süzmesin. META_GATE=enforce
    açıksa aday WAIT'e çevrilmiş gelir; elenen yön `meta_gated`tan geri alınır — meta kapının
    hükmünü yalnız `meta_kapi` kasası uygular."""
    sig = generate_signal(symbol, df, shared=shared, strategy_mode=strateji())
    taraf = (sig.action.value if sig.action in (SignalAction.LONG, SignalAction.SHORT)
             else (sig.metadata or {}).get("meta_gated"))
    if taraf not in ("LONG", "SHORT"):
        return None
    return sig if sig.action.value == taraf else replace(sig, action=SignalAction(taraf))


def olc(symbol: str, df: pd.DataFrame, sig: Signal, shared=None, *, conf_bar: float | None = None) -> Olcum:
    """Beş mekanizmanın bu aday için çıktısı. Her yüzey kendi try'ında: biri patlarsa o ölçü None
    olur, kalanlar ölçülür (fail-soft) ve `durum()["son_hata"]`e yazılır (sessiz ölüm yok)."""
    taraf = sig.action.value
    md = sig.metadata or {}
    meta_p = md.get("meta_p_win")
    if meta_p is None:
        try:
            from nihai.signals.meta_gate import get_gate
            meta_p = get_gate().probability(df, sig, shared=shared)
        except Exception as exc:  # noqa: BLE001
            _hata("meta", exc)
    noral_p = None
    try:
        from nihai.execution.neural_shadow import get_shadow
        from nihai.signals.meta_features import extract_features
        feats: dict[str, Any] = dict(extract_features(df, sig, shared=shared))
        feats["is_long"] = 1.0 if taraf == "LONG" else 0.0
        r = get_shadow().pwin(feats, taraf)
        noral_p = None if r is None else float(r[0])
    except Exception as exc:  # noqa: BLE001
        _hata("noral", exc)
    ai: dict | None = None
    try:
        from nihai.execution.ai_advisor import shadow_verdict
        ai = shadow_verdict(symbol, df, sig, strategy=strateji(), side=taraf)
    except Exception as exc:  # noqa: BLE001
        _hata("ai", exc)
    karantina = vap = None
    try:
        from nihai.ai.observatory import bar_shadow  # köprünün ölçüm yüzeyi (`shadow_bridge._observatory`)
        obs = bar_shadow(df, symbol) or {}
        if obs:
            karantina = bool(obs.get("quarantine", False))
            vap = None if obs.get("value_area_pos") is None else float(obs["value_area_pos"])
    except Exception as exc:  # noqa: BLE001
        _hata("kopru", exc)
    guven = 1.0
    try:
        from nihai.execution import watchlist
        guven = watchlist.risk_mult(symbol)
    except Exception as exc:  # noqa: BLE001
        _hata("guven", exc)
    return Olcum(meta_p=None if meta_p is None else float(meta_p), noral_p=noral_p,
                 ai_action=(ai or {}).get("action"), ai_post=(ai or {}).get("post"),
                 ai_veto=None if ai is None else int(ai["veto"]), karantina=karantina,
                 conf_bar=conf_bar, value_area_pos=vap, guven=guven)


_son_hata: str | None = None


def _hata(nerede: str, exc: Exception) -> None:
    global _son_hata
    ilk = _son_hata is None
    _son_hata = f"{nerede}: {type(exc).__name__}: {exc}"[:200]
    (logger.warning if ilk else logger.debug)(f"[golge_kasalar] {_son_hata}")


# --------------------------------------------------------------------------- zaman
def _utc(ts) -> pd.Timestamp:
    """Bar zamanı → naive UTC (sistem sözleşmesi: naive = UTC)."""
    t = pd.Timestamp(ts)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def _utc_index(idx) -> pd.DatetimeIndex:
    i = pd.DatetimeIndex(idx)
    return i.tz_convert("UTC").tz_localize(None) if i.tz is not None else i


def _et_naive(ts) -> datetime:
    return _utc(ts).tz_localize("UTC").tz_convert("America/New_York").tz_localize(None).to_pydatetime()


def kapali_barlar(df: pd.DataFrame, simdi: datetime | None = None) -> pd.DataFrame:
    """Yalnız KAPANMIŞ 1m barlar (başlangıç + 1 dk ≤ şimdi, UTC). Oluşan bar kasaya girmez."""
    if df is None or df.empty:
        return df
    simdi_utc = pd.Timestamp(simdi) if simdi is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    return df[_utc_index(df.index) + pd.Timedelta(minutes=1) <= simdi_utc]


# --------------------------------------------------------------------------- kasa + motor
@dataclass
class _Emir:
    sig: Signal
    carpan: float
    karar_bar: pd.Timestamp


class _Kasa:
    def __init__(self, ad: str, kok: Path, bakiye: float):
        self.ad = ad
        self.bakiye = bakiye
        self.store = StateStore(kok / f"{ad}.json")
        self.portfoy = self.store.load_or_init(bakiye)
        self.risk = RiskManager(self.portfoy)
        self.pm = PositionManager(self.portfoy, record_logs=False)
        self.pm.rehydrate_trackers()
        self.bekleyen: dict[str, _Emir] = {}

    def dolu(self, symbol: str) -> bool:
        return symbol in self.bekleyen or any(p.symbol == symbol for p in self.portfoy.open_positions)


class Motor:
    """Altı kasa + ortak bar işleyici. Tahmin servisinin 8 iş parçacığı aynı anda çağırır: model
    çağrıları KİLİT DIŞINDA, portföy değişiklikleri tek kilit içinde."""

    def __init__(self, kok: Path | None = None, bakiye: float | None = None):
        self.kok = kok or dizin()
        self.kok.mkdir(parents=True, exist_ok=True)
        b = bakiye if bakiye is not None else _envf("GOLGE_KASA_BAKIYE", 1_000_000.0)
        self.kasalar = {ad: _Kasa(ad, self.kok, b) for ad in KASALAR}
        self._kilit = threading.Lock()
        self._motor_yolu = self.kok / "motor.json"
        try:
            self._son_bar: dict[str, str] = json.loads(self._motor_yolu.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._son_bar = {}
        self._motor_yazildi = 0.0
        self._noral: deque[float] = deque(_noral_gecmisi(self.kok / "kararlar.csv"), maxlen=NORAL_PENCERE)
        self.sayac = {"bar": 0, "aday": 0, "acilan": 0, "kapanan": 0, "iptal": 0, "seans_kapisi": 0,
                      "rr_kapisi": 0}

    # ------------------------------------------------------------------
    def bar_isle(self, symbol: str, df: pd.DataFrame, shared=None, *,
                 simdi: datetime | None = None, conf_bar: float | None = None) -> None:
        kapali = kapali_barlar(df, simdi)
        if kapali is None or kapali.empty:
            return
        son = _utc(kapali.index[-1])
        with self._kilit:
            onceki = self._son_bar.get(symbol)
        if onceki is not None and pd.Timestamp(onceki) >= son:
            return
        # Aday + ölçüm kilit dışında. `shared`/`conf_bar` df'in son barına aittir: yalnız df'in TAMAMI kapalıysa.
        tam = len(kapali) == len(df)
        paylasim = shared if tam else None
        sig = aday(symbol, kapali, paylasim)
        olcum = (olc(symbol, kapali, sig, paylasim, conf_bar=conf_bar if tam else None)
                 if sig is not None else None)
        with self._kilit:
            onceki = self._son_bar.get(symbol)
            if onceki is not None and pd.Timestamp(onceki) >= son:
                return   # başka iş parçacığı aynı barı işledi
            yeni = kapali[_utc_index(kapali.index) > pd.Timestamp(onceki)] if onceki is not None else kapali.iloc[-1:]
            degisen: set[str] = set()
            for ts, bar in yeni.iterrows():
                degisen |= self._bar(symbol, _utc(ts), bar)
            if sig is not None and olcum is not None:
                degisen |= self._aday_kaydet(symbol, son, sig, olcum)
            self._son_bar[symbol] = son.isoformat()
            if degisen or time.monotonic() - self._motor_yazildi >= 60:
                # TUTARLI anlık görüntü: altı kasa + son bar AYNI anda. Yalnız `motor.json` ilerleseydi kaydedilmemiş
                # güncellemeler (izleyen stop, işaret) yeniden başlatmada atlanırdı; böyle son kayıttan sonraki barlar
                # aynen yeniden işlenir.
                for k in self.kasalar.values():
                    k.store.save()
                atomic_write_text(self._motor_yolu, json.dumps(self._son_bar, indent=0))
                self._motor_yazildi = time.monotonic()

    def _bar(self, symbol: str, ts: pd.Timestamp, bar) -> set[str]:
        """Tek kapalı bar: önce açık pozisyonlar (çıkış), sonra bekleyen giriş (açılışta)."""
        self.sayac["bar"] += 1
        o, h, lo, c = (float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"]))
        simdi = _et_naive(ts)
        degisen: set[str] = set()
        for k in self.kasalar.values():
            k.portfoy.roll_session_if_new_day(simdi.date())
            for ce in k.pm.update_all_with_bars({symbol: (o, h, lo, c)}, now=simdi):
                self.sayac["kapanan"] += 1
                degisen.add(k.ad)
                _csv_ekle(self.kok / "islemler.csv", _ISLEM_KOLON, {
                    "kasa": k.ad, "sembol": symbol, "taraf": ce.pos.side, "adet": ce.pos.quantity,
                    "giris_zamani": pd.Timestamp(ce.pos.entry_time).isoformat(timespec="minutes"), "giris": round(ce.pos.entry_price, 4),
                    "cikis_zamani": simdi.isoformat(timespec="minutes"), "cikis": round(ce.exit_price, 4),
                    "tetik": ce.exit_trigger, "kz": round(ce.realized_pnl, 2),
                    "ozsermaye": round(k.portfoy.equity, 2)})
            emir = k.bekleyen.get(symbol)
            if emir is None or ts <= emir.karar_bar:
                continue
            del k.bekleyen[symbol]
            if ts - emir.karar_bar > _BAYAT:
                self.sayac["iptal"] += 1
                continue
            # Tur-106 (BULGU-106-8): canlı yol yeni girişi `can_enter_new_position` ile kapıdan geçirir (açılışın
            # ilk 5 dk'sı, 15:50 ET MOC kesimi); kasa sormuyordu → 1 Eki'ye kadar 75 giriş ilk 5 dk'da, 136 giriş
            # 15:50'den sonra açıldı. Motor 16:00'da durduğu için son dakikalarda açılan pozisyon SESSION_FLAT'e
            # ulaşamadan gece taşınıyordu ("çıkış canlı motorla aynı" iddiası bozuluyordu).
            if not can_enter_new_position(simdi)[0]:
                self.sayac["seans_kapisi"] += 1
                continue
            taraf = emir.sig.action.value
            # Tur-106 (BULGU-106-10): canlı yolun İKİ giriş kapısı (`main.scan_symbol`): sinyal işlem görülebilir
            # fiyata çapalanır (Tur-39 STALE_SIGNAL), aynı `min_rr` kaymalı DOLUM fiyatında yeniden sınanır (Tur-89
            # FILL_SLIPPAGE). Kasa ikisini de atlıyordu: 1m sinyalin medyan riski %0,108 iken açılış kovalaması +
            # %0,10 taban kayma RR'ı 2,0'dan 0,44'e indiriyordu; kontrol kasasında girişlerin %88'i bu kapıdan
            # geçemezdi ve −268,6 k$ zararın −244,4 k$'ı onlardandı (kasa canlının asla girmeyeceği dolumu ölçüyordu).
            min_rr = CONFIG.strategy.min_rr_ratio
            sig_o, _ = reanchor_to_live(emir.sig, o, min_rr=min_rr)
            if sig_o is None or reanchor_to_live(sig_o, k.pm.slip.apply_entry(o, taraf, "stock"), min_rr=min_rr)[1]:
                self.sayac["rr_kapisi"] += 1
                continue
            # Çarpan size_stock'a VERİLMEZ (BULGU-102-7): frenler/tavanlar orada, pozisyon payı + çarpan `kasa_adedi`nde.
            rc = k.risk.size_stock(entry=o, stop=emir.sig.stop, sector=get_sector(symbol), symbol=symbol, side=taraf)
            if not rc.approved:
                continue
            adet = kasa_adedi(int(rc.sizing["shares"]), o, k.portfoy.equity, emir.carpan, poz_payi())
            if adet <= 0:
                continue
            boyut = {**rc.sizing, "shares": adet, "risk_dollars": adet * abs(o - emir.sig.stop),
                     "notional": adet * o, "confidence_mult": emir.carpan}
            etiket = (emir.sig.metadata or {}).get("strategy", "EventBreakout")
            k.pm.open_from_signal(replace(emir.sig, price=o), boyut, now=simdi, sector=get_sector(symbol),
                                  strategy_name=f"{etiket} V1", signal_price=emir.sig.price)
            self.sayac["acilan"] += 1
            degisen.add(k.ad)
        return degisen

    def _aday_kaydet(self, symbol: str, son: pd.Timestamp, sig: Signal, o: Olcum) -> set[str]:
        self.sayac["aday"] += 1
        me = meta_esik()
        ne = float(pd.Series(self._noral).median()) if len(self._noral) >= NORAL_ISINMA else None
        satir: dict[str, Any] = {
            "kayit": datetime.now().isoformat(timespec="seconds"), "bar": son.isoformat(), "sembol": symbol,
            "taraf": sig.action.value, "fiyat": round(float(sig.price), 4), "stop": round(float(sig.stop), 4),
            "hedef": round(float(sig.target), 4), "meta_p": o.meta_p, "noral_p": o.noral_p,
            "noral_esik": None if ne is None else round(ne, 6),
            "ai_action": o.ai_action, "ai_post": o.ai_post, "ai_veto": o.ai_veto, "karantina": o.karantina,
            "conf_bar": None if o.conf_bar is None else round(o.conf_bar, 4), "value_area_pos": o.value_area_pos,
            "guven": o.guven}
        degisen: set[str] = set()
        for ad, k in self.kasalar.items():
            gir, carpan, neden = karar(ad, o, meta_esik=me, noral_esik=ne)
            if gir and k.dolu(symbol):
                satir[ad] = "dolu"
            elif gir:
                k.bekleyen[symbol] = _Emir(sig, carpan, son)
                satir[ad] = f"gir×{carpan:.2f}"
            else:
                satir[ad] = f"veto: {neden}"
        _csv_ekle(self.kok / "kararlar.csv", _KARAR_KOLON, satir)
        if o.noral_p is not None:
            self._noral.append(o.noral_p)   # eşik yalnız ÖNCEKİ ölçümlerden (bu aday kendi eşiğine girmez)
        return degisen

    # ------------------------------------------------------------------
    def durum(self) -> dict:
        """Bellekteki kasalar (canlı işaretli). Salt okuma — kilit altında kopya."""
        with self._kilit:
            return _ozet({ad: k.portfoy for ad, k in self.kasalar.items()},
                         {ad: len(k.bekleyen) for ad, k in self.kasalar.items()}, self.kok,
                         self.kasalar["kontrol"].bakiye, dict(self.sayac))


def _noral_gecmisi(yol: Path) -> list[float]:
    """Yeniden başlatmada nöral eşiğin penceresi DEFTERDEN kurulur (ayrı durum dosyası yok)."""
    try:
        with yol.open(encoding="utf-8", newline="") as f:
            degerler = [float(r["noral_p"]) for r in csv.DictReader(f) if r.get("noral_p")]
    except (OSError, ValueError, KeyError):
        return []
    return degerler[-NORAL_PENCERE:]


def _csv_ekle(yol: Path, kolonlar: tuple[str, ...], satir: dict) -> None:
    yeni = not yol.exists()
    with yol.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(kolonlar))
        if yeni:
            w.writeheader()
        w.writerow({k: ("" if satir.get(k) is None else satir.get(k)) for k in kolonlar})


# --------------------------------------------------------------------------- rapor
def _islem_istatistik(kok: Path) -> dict[str, dict]:
    out: dict[str, dict] = {ad: {"n": 0, "kazanan": 0, "brut_kar": 0.0, "brut_zarar": 0.0} for ad in KASALAR}
    try:
        with (kok / "islemler.csv").open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                s = out.get(r.get("kasa") or "")
                if s is None:
                    continue
                kz = float(r["kz"])
                s["n"] += 1
                s["kazanan"] += kz > 0
                s["brut_kar" if kz > 0 else "brut_zarar"] += abs(kz)
    except (OSError, ValueError, KeyError):
        pass
    return out


def _ozet(portfoyler: dict, bekleyen: dict, kok: Path, bakiye: float, sayac: dict | None) -> dict:
    ist = _islem_istatistik(kok)
    kasalar: dict[str, dict] = {}
    for ad in KASALAR:
        p = portfoyler[ad]
        acik = [{"sembol": x.symbol, "taraf": x.side, "adet": x.quantity, "giris": round(x.entry_price, 4),
                 "son": round(x.last_mark, 4),
                 "kz": round(x.unrealized_pnl, 2) + 0.0} for x in p.open_positions]   # +0.0: −0,0 → 0,0 (pano "$-0.00")
        mtm = p.equity + sum(a["kz"] for a in acik)
        s = ist[ad]
        kasalar[ad] = {
            "aciklama": ACIKLAMA[ad], "bakiye": bakiye, "ozsermaye": round(p.equity, 2), "mtm": round(mtm, 2),
            "getiri_pct": round((mtm / bakiye - 1) * 100, 4), "acik": acik, "bekleyen": bekleyen.get(ad, 0),
            "islem": s["n"], "isabet": round(s["kazanan"] / s["n"], 4) if s["n"] else None,
            "kar_faktoru": (round(s["brut_kar"] / s["brut_zarar"], 3) if s["brut_zarar"] > 0
                            else None)}
    taban = kasalar["kontrol"]["getiri_pct"]
    for v in kasalar.values():
        v["fark_kontrol_pp"] = round(v["getiri_pct"] - taban, 4)
    return {"etkin": etkin(), "strateji": strateji(), "dizin": str(kok), "kasalar": kasalar,
            "sayac": sayac, "son_hata": _son_hata}


def durum_diskten(kok: Path | None = None) -> dict:
    """Diskteki kasalar — koşan süreçle çakışmaz (yazmaz). Açık pozisyonlar son KAYDEDİLEN işaretle."""
    kok = kok or dizin()
    b = _envf("GOLGE_KASA_BAKIYE", 1_000_000.0)
    return _ozet({ad: StateStore(kok / f"{ad}.json").load_or_init(b) for ad in KASALAR}, {}, kok, b, None)


_MOTOR: Motor | None = None
_MOTOR_KILIT = threading.Lock()
_YAZAR: Any = None            # süreç ömrü boyunca açık tutulan tek-yazar kilidi (kapanınca işletim sistemi bırakır)
_SON_DENEME = 0.0
_DENEME_ARALIGI = 300.0


def _yazar_kilidi(kok: Path) -> bool:
    """Kasa dizinine TEK süreç yazar (Tur-80 çift kayıtçı dersi): canlı ana süreç ve ayrı koşucu (`--kos`) aynı anda
    açıksa ikincisi kasalara dokunmaz. Engelsiz dener; kilidi alan süreç kapanınca işletim sistemi bırakır."""
    global _YAZAR
    from nihai.atomic_io import tek_yazar_kilidi

    f = tek_yazar_kilidi(kok / "motor.lock")
    if f is None:
        return False
    _YAZAR = f
    return True


def motor() -> Motor | None:
    """Süreç başına tek motor; kasa dizinini başka bir süreç tutuyorsa None (5 dk'da bir yeniden denenir)."""
    global _MOTOR, _SON_DENEME
    if _MOTOR is None:
        with _MOTOR_KILIT:
            if _MOTOR is None:
                if time.monotonic() - _SON_DENEME < _DENEME_ARALIGI and _SON_DENEME:
                    return None
                _SON_DENEME = time.monotonic()
                if not _yazar_kilidi(dizin()):
                    logger.info("[golge_kasalar] kasa dizinini başka bir süreç yazıyor — bu süreç kasalara dokunmaz")
                    return None
                _MOTOR = Motor()
    return _MOTOR


def isle(symbol: str, df: pd.DataFrame, shared=None, *, conf_bar: float | None = None) -> None:
    """Tahmin servisinin tek çağrısı. FAIL-SOFT: kasa hatası tahmin servisini ASLA düşürmez."""
    if not etkin():
        return
    try:
        m = motor()
        if m is not None:
            m.bar_isle(symbol, df, shared, conf_bar=conf_bar)
    except Exception as exc:  # noqa: BLE001
        _hata(f"bar_isle {symbol}", exc)


def durum() -> dict:
    """API/pano: aynı süreçte motor varsa bellekten (canlı işaret), yoksa diskten (başka süreç yazıyorsa ≤60 sn geride)."""
    return _MOTOR.durum() if _MOTOR is not None else durum_diskten()


def kos() -> None:
    """AYRI koşucu: canlı ana sürece dokunmadan kasaları canlı veriyle koşturur (`python -m … --kos`).

    Tahmin servisini bu süreçte `PREDICT_TRACK=false` ile başlatır → tahmin defterine ÇİFT kayıt yazılmaz (canlı süreç
    kendi defterini yazmaya devam eder); kasa çağrısı izleme koşulunun DIŞINDADIR. Tek-yazar kilidi: kasaları o an
    başka süreç yazıyorsa hiç başlamaz."""
    os.environ["PREDICT_TRACK"] = "false"
    os.environ["PREDICT_REFRESH_SEC"] = os.getenv("GOLGE_KASA_YENILEME_SN") or "15"
    os.environ["GOLGE_KASA"] = "on"
    m = motor()
    if m is None:
        print("Kasa dizinini başka bir süreç yazıyor — koşucu başlamadı.")
        return
    from nihai.execution.live_predict_service import get_service
    get_service().start()
    logger.info(f"[golge_kasalar] ayrı koşucu başladı: dizin {m.kok}, strateji {strateji()}")
    while True:
        time.sleep(300)
        d = m.durum()
        logger.info("[golge_kasalar] " + " · ".join(
            f"{ad} {k['getiri_pct']:+.3f}% ({k['islem']} işlem, {len(k['acik'])} açık)" for ad, k in d["kasalar"].items())
            + f" · sayaç {d['sayac']}")


def main() -> None:
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()
    import argparse
    ap = argparse.ArgumentParser(description="Gölge kasalar: rapor (varsayılan) ya da ayrı canlı koşucu.")
    ap.add_argument("--kos", action="store_true", help="kasaları canlı ana sürece dokunmadan AYRI süreçte koştur")
    if ap.parse_args().kos:
        # `python -m` bu dosyayı `__main__` olarak yükler; tahmin servisi ise `nihai.execution.golge_kasalar`ı İKİNCİ kez
        # içe aktarır → iki modül kopyası, iki `_MOTOR`, aynı süreçte kilit çatışması (ikinci kopya kasalara hiç
        # dokunmaz). Koşucu kancanın gördüğü KANONİK modülde çalışmalı.
        from nihai.execution import golge_kasalar as kanonik
        kanonik.kos()
        return
    d = durum_diskten()
    print(f"Gölge kasalar — strateji {d['strateji']}, dizin {d['dizin']}")
    print(f"{'kasa':15} {'MTM $':>14} {'getiri %':>9} {'fark pp':>8} {'işlem':>6} {'isabet':>7} {'KF':>6} açık")
    for ad, k in d["kasalar"].items():
        isb = "—" if k["isabet"] is None else f"{k['isabet']:.1%}"
        kf = "—" if k["kar_faktoru"] is None else f"{k['kar_faktoru']:.2f}"
        print(f"{ad:15} {k['mtm']:>14,.2f} {k['getiri_pct']:>9.3f} {k['fark_kontrol_pp']:>8.3f} "
              f"{k['islem']:>6} {isb:>7} {kf:>6} {len(k['acik'])}")


if __name__ == "__main__":
    main()
