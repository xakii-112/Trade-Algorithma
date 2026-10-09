"""
state.py — Kalıcı durum (cooldown, sanal pozisyonlar, kill-switch, gün-içi sayaç).

PDF Referansları:
- Madde 19 : Karar/işlem kalıcı dosyalara
- Madde 20 : Strateji kill-switch (rolling expectancy)
- Madde 41 : Başlangıç mutabakatı (sanal portföyde bile state ↔ portfolio tutarlılık)

JSON formatında diske yazılır. main.py başlangıçta okur, doğrular, hazırsa devam eder.
"""
from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import asdict, fields, is_dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from loguru import logger

from nihai.atomic_io import atomic_write_text
from nihai.config import (
    CONFIG,
    ENABLE_KILL_SWITCH,
    HTF_INTERVAL,
    INTERVAL_MINUTES,
    KILL_SWITCH_MIN_TRADES,
    KILL_SWITCH_RESET_EXPECTANCY,
    STATE_FILE,
)
from nihai.risk.risk_manager import PortfolioState, VirtualPosition

# Tur-106 (BULGU-106-2): anahtar DOSYA başına. Tur-102'den beri aynı süreç 6 gölge kasa durumunu
# sırayla yüklüyor; tek süreç-genel anahtar her yüklemede değiştiği için Tur-89'un "yalnız
# değişince" kuralı hiç tutmuyordu (1 Ekim: system.log satırlarının %31'i bu satırdı).
_LAST_LOAD_LOG: dict[str, tuple[float, int]] = {}


def _serialize(obj):
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, deque):
        return list(obj)
    return obj


def _deserialize_position(d: dict) -> VirtualPosition:
    """JSON'dan yüklenen pozisyon dict'ini VirtualPosition'a çevirir.

    BUG düzeltmesi: `VirtualPosition(**pos)` tip dönüşümü yapmaz → entry_time bir
    STRING, greeks bir DICT olarak kalıyordu. Restart sonrası taşınan pozisyon
    kapanınca post-mortem `exit_time - entry_time` TypeError, Greek toplamada
    `greeks.delta` AttributeError veriyordu. Burada doğru tiplere parse edilir.
    """
    from nihai.risk.options_analyzer import GreekSet
    d = dict(d)
    et = d.get("entry_time")
    if isinstance(et, str):
        try:
            d["entry_time"] = datetime.fromisoformat(et)   # tam zaman (eski date-only da midnight olur)
        except ValueError:
            d["entry_time"] = date.fromisoformat(et)
    g = d.get("greeks")
    if isinstance(g, dict):
        try:
            d["greeks"] = GreekSet(**g)
        except TypeError:
            d["greeks"] = None
    # Bilinmeyen/fazla anahtarları ele (ileri uyumluluk)
    valid = {f.name for f in fields(VirtualPosition)}
    d = {k: v for k, v in d.items() if k in valid}
    return VirtualPosition(**d)


class StateStore:
    """Sanal portföy + rolling expectancy + cooldown'lar."""

    def __init__(self, path: Path = STATE_FILE):
        self.path = path
        self.portfolio: PortfolioState | None = None
        self.recent_trades_pnl: deque[float] = deque(maxlen=50)  # Madde 20
        self.kill_switch_active: bool = False
        self.symbol_cooldowns: dict[str, datetime] = {}          # Madde 7
        # Madde 53 — wash-sale defteri: son kapanan zararlı işlemler {symbol, date(iso), loss}
        self.loss_history: list[dict] = []
        # Sinyal akışı kalıcılığı (restart'ta kaybolmasın)
        self.recent_signals: list[str] = []
        # Madde 23 — T+1: takas edilmemiş nakit dilimleri {settle_date(iso), amount}
        self.unsettled_lots: list[dict] = []
        # Tur-106 (BULGU-106-34) — quant taramasının GEÇİCİ izleri (state.json'a YAZILMAZ; süreç ömürlü):
        # sembol başına son değerlendirme anı (ET) ve o ET günü yazılmış muhafız satırları. `main.scan_symbol`
        # okur; bu öznitelikleri taşımayan depo (test ikizi) seyreltilmez.
        self.quant_son_tarama: dict[str, datetime] = {}
        self.quant_gun_satiri: dict = {"gun": None, "s": set()}

    # -----------------------------------------------------------------
    def load_or_init(self, initial_equity: float) -> PortfolioState:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                p = raw.get("portfolio", {})
                self.portfolio = PortfolioState(
                    equity=p.get("equity", initial_equity),
                    ath_equity=p.get("ath_equity", initial_equity),
                    cash_balance=p.get("cash_balance", initial_equity),
                    day_trades_count=p.get("day_trades_count", 0),
                    day_trades_window_start=date.fromisoformat(p.get(
                        "day_trades_window_start", date.today().isoformat())),
                    daily_pnl=p.get("daily_pnl", 0.0),
                    consecutive_stops=p.get("consecutive_stops", 0),
                    unsettled_cash=p.get("unsettled_cash", 0.0),
                    # Alan yoksa (eski state) geçmiş tarih → ilk recon'da günlük sayaçlar sıfırlanır
                    last_session_date=date.fromisoformat(p["last_session_date"])
                    if p.get("last_session_date") else date(1970, 1, 1),
                    open_positions=[_deserialize_position(pos) for pos in p.get("open_positions", [])],
                    day_trade_dates=([date.fromisoformat(d) for d in p["day_trade_dates"]]
                                     if "day_trade_dates" in p else self._day_trades_from_ledger()),
                )
                self.recent_trades_pnl = deque(raw.get("recent_trades_pnl", []), maxlen=50)
                self.kill_switch_active = raw.get("kill_switch_active", False) if ENABLE_KILL_SWITCH else False
                self.loss_history = raw.get("loss_history", [])
                self.recent_signals = raw.get("recent_signals", [])
                self.unsettled_lots = raw.get("unsettled_lots", [])
                self.symbol_cooldowns = {
                    k: datetime.fromisoformat(v)
                    for k, v in raw.get("symbol_cooldowns", {}).items()
                }
                # Tur-89: pano her yoklamada durumu diskten okur → satır yalnız DEĞİŞİNCE yazılır
                # (eskiden system.log'un 7.857 satırı buydu; konsol akışı okunamıyordu).
                key = (round(self.portfolio.equity, 2), len(self.portfolio.open_positions))
                if _LAST_LOAD_LOG.get(str(self.path)) != key:
                    _LAST_LOAD_LOG[str(self.path)] = key
                    logger.info(f"[state] yüklendi: equity={self.portfolio.equity:.2f}, "
                                f"{len(self.portfolio.open_positions)} açık pozisyon")
            except Exception as e:
                logger.error(f"[state] bozuk durum dosyası, sıfırdan başlatılıyor: {e}")
                self._init_fresh(initial_equity)
        else:
            self._init_fresh(initial_equity)
        assert self.portfolio is not None   # load/_init_fresh her dalda set eder (mypy garantisi)
        return self.portfolio

    def _day_trades_from_ledger(self) -> list[date]:
        """Tur-90 göçü: `day_trade_dates` taşımayan eski durum için day-trade günlerini defterden
        kur (giriş ve çıkış AYNI ET günü; son 14 gün yeter — pencere 5 seanstır). Defter yoksa [] →
        `pdt_refresh` eski sayacı pencere başına koyar (davranış eskisinden sıkı değil)."""
        ledger = self.path.parent / "trades.csv"
        esik = (datetime.now() - timedelta(days=14)).date().isoformat()
        out: list[date] = []
        try:
            with ledger.open(encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    g, c = (r.get("entry_time") or "")[:10], (r.get("exit_time") or "")[:10]
                    if g and g == c and c >= esik:
                        out.append(date.fromisoformat(c))
        except (OSError, ValueError):
            return []
        return out

    def _init_fresh(self, initial_equity: float) -> None:
        self.portfolio = PortfolioState(
            equity=initial_equity, ath_equity=initial_equity, cash_balance=initial_equity,
        )

    # -----------------------------------------------------------------
    def save(self) -> None:
        if not self.portfolio:
            return
        payload = {
            "portfolio": _serialize(self.portfolio),
            "recent_trades_pnl": list(self.recent_trades_pnl),
            "kill_switch_active": self.kill_switch_active,
            "loss_history": self.loss_history,
            "recent_signals": self.recent_signals[-50:],
            "unsettled_lots": self.unsettled_lots,
            "symbol_cooldowns": {k: v.isoformat() for k, v in self.symbol_cooldowns.items()},
            "saved_at": datetime.now().isoformat(),
        }
        # Atomik + OneDrive-kilidine dayanıklı yazma (Tur-12, nihai/atomic_io): önce .tmp'ye
        # yaz, sonra yeniden-denemeli os.replace. (a) yazma sırasında çökme/SIGKILL state.json'u
        # YARIM bırakmasın → load_or_init bozuk JSON'da SIFIRDAN başlar (açık pozisyon+equity
        # kaybı); (b) OneDrive senkron kilidi ham os.replace'i PermissionError'la düşürüp ~60sn'lik
        # canlı döngüyü çökertmesin (2026-07-11 predictions_log olayının state.json karşılığı).
        atomic_write_text(self.path, json.dumps(payload, indent=2, default=str))

    # -----------------------------------------------------------------
    # Madde 20 — rolling expectancy & kill-switch
    def record_trade_pnl(self, pnl: float) -> None:
        self.recent_trades_pnl.append(pnl)
        if not (ENABLE_KILL_SWITCH and len(self.recent_trades_pnl) >= KILL_SWITCH_MIN_TRADES):
            return
        exp = sum(self.recent_trades_pnl) / len(self.recent_trades_pnl)
        if not self.kill_switch_active and exp < 0:
            self.kill_switch_active = True
            logger.warning(f"[state] KILL-SWITCH AKTİF — rolling expectancy {exp:.2f} < 0")
        elif self.kill_switch_active and exp >= KILL_SWITCH_RESET_EXPECTANCY:
            # Otomatik reset: rolling expectancy toparladı → kilit kalkar (AI_trade deseni)
            self.kill_switch_active = False
            logger.warning(f"[state] KILL-SWITCH RESET — rolling expectancy {exp:.2f} "
                           f"≥ {KILL_SWITCH_RESET_EXPECTANCY:.2f}, yeni girişler tekrar açık")

    def cooldown_active(self, symbol: str, now: datetime) -> bool:
        ts = self.symbol_cooldowns.get(symbol)
        if not ts:
            return False
        # Madde 7 — X bar cooldown. "bar" uzunluğu HTF interval'inden tek kaynak olarak gelir
        # (önceden sabit 60 sn=1 dk varsayılıyordu; 1h stratejisinde etkisizdi).
        bar_min = INTERVAL_MINUTES.get(HTF_INTERVAL, 60)
        cooldown_sec = CONFIG.strategy.reentry_cooldown_bars * bar_min * 60
        return (now - ts).total_seconds() < cooldown_sec

    def set_cooldown(self, symbol: str, now: datetime) -> None:
        self.symbol_cooldowns[symbol] = now

    # -----------------------------------------------------------------
    # Madde 53 — Wash-sale izleme
    def register_close_for_wash_sale(self, symbol: str, pnl: float, when: datetime,
                                     asset_type: str = "") -> None:
        """Kapanan işlem zararlıysa wash-sale defterine ekle; eski kayıtları buda.

        `asset_type` (Tur-58, additive): kayıp HİSSEDEN mi OPSİYONDAN mı geldi. Defterin
        anahtarı zaten DAYANAK sembolüdür (opsiyon pozisyonları da `pos.symbol` ile
        yazılır) → çapraz eşleşme fiilen ZATEN oluyordu; eksik olan, gerekçenin bunu
        SÖYLEYEBİLMESİYDİ. Eski state.json kayıtlarında alan yoktur → `.get()` ile okunur."""
        if pnl < 0:
            rec: dict = {
                "symbol": symbol.upper(),
                "date": when.date().isoformat(),
                "loss": round(pnl, 2),
            }
            if asset_type:
                rec["asset_type"] = asset_type
            self.loss_history.append(rec)
        # 30 günden eski kayıtları temizle
        cutoff = (when.date() - timedelta(days=CONFIG.regulation.wash_sale_days)).isoformat()
        self.loss_history = [r for r in self.loss_history if r["date"] >= cutoff]

    def wash_sale_check(self, symbol: str, when: datetime,
                        asset_type: str = "") -> tuple[bool, str]:
        """Yeni giriş wash-sale tetikler mi? (aynı DAYANAKTA son 30 günde zararlı kapanış).

        **Section 1256 muafiyeti (Tur-58, Madde 29 + Madde 53).** Nakit uzlaşılı endeks
        opsiyonları (SPX/NDX/VIX/RUT/XSP) 26 U.S.C. §1256 kapsamında yıl sonunda PİYASAYA
        GÖRE DEĞERLENİR; zarar realize edilmiş sayılır ve wash-sale kuralı bu sözleşmelere
        UYGULANMAZ. Eski kod onları da işaretliyordu = YANLIŞ vergi bilgisi. `tax_1256`
        alanı `settlement_style` içinde yıllardır vardı ama yalnız metadata olarak
        taşınıyordu — okunmayan bir ölçüm (Tur-38).

        Çapraz eşleşme (§E "Substantially Identical"): hisse zararından sonra AYNI
        dayanakta opsiyon girişi (ya da tersi) ayrı bir etiketle raporlanır. Derin ITM
        call'ın hisseye "substantially identical" sayılması IRS yorumudur; sistem BLOKE
        ETMEZ, yalnız muhasebe için işaretler."""
        sym = symbol.upper()
        from nihai.risk.options_analyzer import settlement_style
        if settlement_style(sym).get("tax_1256"):
            return False, "SECTION_1256_WASH_SALE_MUAF"
        cutoff = (when.date() - timedelta(days=CONFIG.regulation.wash_sale_days)).isoformat()
        hits = [r for r in self.loss_history if r["symbol"] == sym and r["date"] >= cutoff]
        if not hits:
            return False, "CLEAR"
        total = sum(r["loss"] for r in hits)
        etiket = "WASH_SALE"
        if asset_type:
            kinds = {str(r.get("asset_type") or "") for r in hits}
            if kinds - {asset_type, ""}:
                etiket = "WASH_SALE_CAPRAZ"      # hisse ↔ opsiyon: substantially identical adayı
        return True, f"{etiket}_{sym}_son30g_zarar_{total:+.2f}_({len(hits)}_islem)"

    # -----------------------------------------------------------------
    # Madde 23 — T+1 takas izleme (nakit hesap good-faith farkındalığı)
    def cash_timeline(self, today: date) -> dict:
        """Kayıtlı takas tarihleri ve mevcut wash-sale kuralıyla aynı izleme penceresi."""
        settlements = []
        for lot in sorted(self.unsettled_lots, key=lambda x: x["settle_date"]):
            settle_date = date.fromisoformat(lot["settle_date"])
            settlements.append({**lot, "days_left": max(0, (settle_date - today).days),
                                "matured": settle_date <= today})
        quarantine = []
        for symbol in sorted({r["symbol"] for r in self.loss_history}):
            hit, reason = self.wash_sale_check(symbol, datetime.combine(today, datetime.min.time()))
            if not hit:
                continue
            cutoff = today - timedelta(days=CONFIG.regulation.wash_sale_days)
            losses = [r for r in self.loss_history if r["symbol"] == symbol and date.fromisoformat(r["date"]) >= cutoff]
            last = max(date.fromisoformat(r["date"]) for r in losses)
            # wash_sale_check kesim gününü DAHİL eder; serbest gün +1 olmalı.
            release = last + timedelta(days=CONFIG.regulation.wash_sale_days + 1)
            quarantine.append({"symbol": symbol, "last_loss_date": last.isoformat(),
                "release_date": release.isoformat(), "days_left": (release - today).days,
                "loss": round(sum(r["loss"] for r in losses), 2), "trades": len(losses), "reason": reason})
        return {"as_of": today.isoformat(), "timezone": "America/New_York", "settlements": settlements,
                "quarantine": quarantine, "wash_sale_days": CONFIG.regulation.wash_sale_days,
                "settlement_days": CONFIG.regulation.settlement_days,
                "note": "Sanal portföyün kayıtlı çözülme tarihleri. Wash-sale izlemesi uyarıdır; otomatik giriş engeli değildir."}

    def add_unsettled_proceeds(self, amount: float, sell_date: date, side: str = "LONG") -> None:
        """Satış geliri T+settlement_days'e kadar 'takas edilmemiş' sayılır.

        Tur-89 (BULGU-89-3): yalnız LONG kapanışı bir SATIŞTIR. SHORT kapanışı geri ALIMDIR ve
        hasılat üretmez; eskiden onun tutarı da ekleniyordu ("Bloke (T+1)" nakdin 120 katı)."""
        if amount <= 0 or side != "LONG":
            return
        settle = (sell_date + timedelta(days=CONFIG.regulation.settlement_days)).isoformat()
        self.unsettled_lots.append({"settle_date": settle, "amount": round(amount, 2)})

    def settle_matured(self, today: date) -> None:
        """Takası dolmuş dilimleri düşer; portfolio.unsettled_cash'i günceller."""
        today_iso = today.isoformat()
        self.unsettled_lots = [lot for lot in self.unsettled_lots if lot["settle_date"] > today_iso]
        if self.portfolio is not None:
            self.portfolio.unsettled_cash = round(sum(lot["amount"] for lot in self.unsettled_lots), 2)

    def good_faith_warning(self, needed_cash: float) -> tuple[bool, str]:
        """Madde 23 — takas edilmemiş parayla alım yapılıyorsa good-faith uyarısı (bloke etmez)."""
        if self.portfolio is None:
            return False, "CLEAR"
        settled = self.portfolio.cash_balance - self.portfolio.unsettled_cash
        if needed_cash > settled and self.portfolio.unsettled_cash > 0:
            return True, (f"GOOD_FAITH_RISK_takasedilmemis_{self.portfolio.unsettled_cash:.0f}"
                          f"_settled_{settled:.0f}")
        return False, "CLEAR"
