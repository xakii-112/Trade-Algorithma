"""
risk_manager.py — Pozisyon boyutu, portföy tavanları, PDT, Greek bütçesi.

PDF Referansları:
- Madde 2  : Sabit risk modeli (uzun opsiyon → primin tamamı)
- Madde 9  : ATH-bazlı adaptif risk (drawdown'da %50 ↓)
- Madde 10 : Günlük max kayıp + PDT
- Madde 12 : Korelasyon / kümelenme
- Madde 13 : Buying power + marjin kontrolü (sanal portföyde de doğrula)
- Madde 14 : Portföy maruziyet tavanları (açık poz, brüt risk, net yönsel)
- Madde 23 : T+1 takas (nakit hesap için izleme)
- Madde 27 : Net Greek limitleri
- Madde 32 : Opsiyon komisyon/ücret yapısı (per-contract)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from nihai.config import CONFIG
from nihai.risk.options_analyzer import GreekSet, PortfolioGreeks

# Madde 32 — Tipik per-contract maliyetler (örnek; broker'ına göre ayarla).
OPTION_FEES = {
    "commission_per_contract": 0.65,
    "occ_clearing": 0.02,
    "orf_per_contract": 0.05,        # exchange ücretleri ~
    "sec_taf_estimate": 0.03,        # satışta SEC + TAF (yaklaşık)
}


def pdt_sessions(today: date, n: int) -> list[date]:
    """`today` dahil ona kadarki son `n` NYSE seansı (bugün seans değilse öncekiler). Tur-90."""
    from nihai.data.market_calendar import NYSE
    n = max(1, int(n))
    oturum = NYSE.sessions_in_range(today - timedelta(days=3 * n + 10), today)
    return [d.date() for d in oturum[-n:]]


@dataclass
class VirtualPosition:
    """Sanal portföyde tutulan pozisyon (canlı emir yok, sadece kayıt)."""
    symbol: str
    asset_type: str               # "stock" | "option"
    side: str                     # "LONG" | "SHORT"
    quantity: int                 # hisse: adet, opsiyon: kontrat
    entry_price: float            # hisse: $, opsiyon: prim
    entry_time: date
    stop: float = 0.0
    target: float = 0.0
    # Opsiyona özgü
    option_type: str | None = None   # "c" | "p"
    strike: float | None = None
    expiry: str | None = None
    greeks: GreekSet | None = None
    sector: str = "UNKNOWN"
    last_mark: float = 0.0           # canlı/son işaretlenen fiyat (pano PnL'i için)
    # BULGU-59-3 — hacim-katılımına duyarlı piyasa etkisinin NEDENSEL girdileri
    # (girişte ölçülür, nedensel bar verisi geldikçe yenilenir). None = ölçüm yok
    # → etki terimi uygulanmaz (SlippageModel taban kaymasına düşer).
    adv_shares: float | None = None
    bar_sigma: float | None = None
    spread_pct: float | None = None  # tam makas / referans fiyat; None = ölçülmedi
    entry_signal_price: float | None = None
    entry_decision_price: float | None = None
    # BULGU-59-6 — giriş anında ölçülen benchmark betası. None = ÖLÇÜLMEDİ (uydurma 1.0
    # YOK): `PortfolioState.beta_exposure` ölçülmemişi ayrı sayar ve kapsamı raporlar.
    beta: float | None = None
    # BULGU-59-4 — pozisyonu AÇAN katman ("odak"/"genis"/""). DUAL_TIER_ENABLED açıkken
    # iki bağımsız strateji katmanı TEK deftere (data/trades.csv) yazıyordu; katman
    # kolonu olmadan katman-başı brüt maruziyet/kaldıraç ayrıştırılamıyordu (Tur-50'nin
    # "brüt maruziyet ÖLÇÜLEMİYOR" açık kalemi). Kapanışta değil AÇILIŞTA belirlenir:
    # çıkış farklı bir thread'den (ana döngü / pano / watchdog) gelebilir.
    tier: str = ""
    # Tur-89 (BULGU-89-5): pozisyonu açan stratejinin adı. Tracker bellekte yaşar ve yeniden
    # başlatmada yeniden kurulur; ad pozisyonda taşınmadığı için taşınan pozisyonlar deftere
    # varsayılan "EventBreakout V1" ile düşüyordu (aynı strateji iki adla, planlanan RR 0).
    strategy: str = ""
    # Tur-97 (BULGU-97-3): quant pozisyonunun İZLEYEN stop mesafesi ($/adet), girişte sabitlenir.
    # Tracker (başlangıç stopu, tepe) bellekte yaşar ve yeniden başlatmada pozisyonun o anki
    # stopundan kurulur; stop lehe kaydıysa mesafe |giriş − stop|'a büzülür ve çok günlük pozisyon
    # erken kesilirdi. 0 = ayarlanmadı (TA pozisyonları; eski state.json kayıtları).
    izleyen_mesafe: float = 0.0
    # Tur-106 (BULGU-106-35): tracker'ın en iyi / en kötü fiyatı ve bar sayısı da pozisyonda taşınır. Tracker bellekte
    # yaşar; proje her gece kapatılıp açıldığı için çok günlük pozisyonun MAE/MFE'si ve bar sayısı yalnız SON oturumu
    # ölçüyordu (defter otopsisi ve backtest kıyası yanlı). 0 = ölçülmedi (eski state.json kayıtları).
    en_iyi_fiyat: float = 0.0
    en_kotu_fiyat: float = 0.0
    bar_sayisi: int = 0

    @property
    def unrealized_pnl(self) -> float:
        """Son mark'a göre gerçekleşmemiş PnL (komisyon hariç)."""
        if self.last_mark <= 0:
            return 0.0
        mult = 100 if self.asset_type == "option" else 1
        sign = +1 if self.side == "LONG" else -1
        return (self.last_mark - self.entry_price) * abs(self.quantity) * mult * sign

    @property
    def risk_dollars(self) -> float:
        """Madde 2 — Uzun opsiyon: primin tamamı. Hisse: |entry-stop|*adet.

        BUG düzeltmesi: SHORT'ta stop > entry olduğundan eski `max(entry-stop, 0)`
        negatifi 0'a kırpıp short pozisyonları SIFIR risk gösteriyordu → brüt-risk
        tavanı (gross_risk = Σ risk_dollars) shortları hiç saymıyor, aşırı kaldıraca
        izin veriyordu. abs() ile her iki yön de doğru hesaplanır.
        """
        if self.asset_type == "option":
            return self.entry_price * 100 * abs(self.quantity)
        return abs(self.entry_price - self.stop) * abs(self.quantity)

    @property
    def notional(self) -> float:
        """Pozisyonun nominal büyüklüğü ($). Opsiyon: prim x 100 x kontrat."""
        mult = 100 if self.asset_type == "option" else 1
        return self.entry_price * abs(self.quantity) * mult

    @property
    def directional_sign(self) -> int:
        """Madde 14 — Yönsel işaret: +1 boğa, -1 ayı.

        Long put ve short hisse 'ayı'dır; long call ve long hisse 'boğa'dır.
        """
        if self.asset_type == "option":
            bullish = (self.option_type == "c")
            base = +1 if bullish else -1
            return base if self.side == "LONG" else -base
        return +1 if self.side == "LONG" else -1

    @property
    def signed_notional(self) -> float:
        """Madde 14 — Yönsel nominal (net uzun-kısa maruziyet için)."""
        return self.notional * self.directional_sign


@dataclass
class PortfolioState:
    equity: float
    ath_equity: float
    cash_balance: float
    day_trades_count: int = 0
    day_trades_window_start: date = field(default_factory=date.today)
    daily_pnl: float = 0.0
    consecutive_stops: int = 0
    open_positions: list[VirtualPosition] = field(default_factory=list)
    # Madde 10 — günlük sayaçların (daily_pnl, consecutive_stops) en son sıfırlandığı
    # seans günü. Yeni gün başlayınca bu sayaçlar sıfırlanır (yoksa daily_pnl ömür-boyu
    # birikir + consecutive_stops cooldown'u kalıcı kilitlenir).
    last_session_date: date = field(default_factory=date.today)

    # Madde 23 — T+1: günü tamamlanmamış satışlardan kalan parayı izle (nakit hesap)
    unsettled_cash: float = 0.0
    # Tur-90 (BULGU-90-4): her day-trade'in ET işlem günü. PDT penceresi artık bundan YÜRÜYEN
    # (son N NYSE seansı) kurulur; `day_trades_count` o pencerenin sayımıdır (pano/uyarı okur).
    day_trade_dates: list[date] = field(default_factory=list)

    @property
    def open_pnl(self) -> float:
        """Açık pozisyonların K/Z toplamı. NLV = `equity` (gerçekleşmiş) + bu; kurallar (PDT eşiği, boyut) `equity` ile
        çalışır. Tur-106 (BULGU-106-7/25): pano, Telegram `/status` ve terminal panosu aynı tanımı okur."""
        return sum(p.unrealized_pnl for p in self.open_positions)

    def register_day_trade(self, et_gun: date) -> None:
        self.day_trade_dates.append(et_gun)
        self.day_trades_count += 1

    def pdt_refresh(self, today: date, lookback: int) -> int:
        """Pencereyi `today`e (ET) yürüt: son `lookback` seansı dışındaki günler düşer.

        Eskiden sayaç TAKVİM günüyle blok blok sıfırlanıyordu (`today − start > 5 gün`): 21 Eyl'deki
        97 işlem 27 Eyl (Pazar) topluca silinecekti; bloğun son günündeki 3 işlem + ertesi
        günün 3'ü 5 iş gününde 6 day-trade eder. FINRA penceresi yürüyen 5 İŞ günüdür."""
        pencere = pdt_sessions(today, lookback)
        bas = pencere[0] if pencere else today
        # Tarihsiz sayım (eski durum dosyası / elle kurulan sayaç) pencere başında sayılır.
        eksik = self.day_trades_count - len(self.day_trade_dates)
        if eksik > 0:
            self.day_trade_dates += [self.day_trades_window_start] * eksik
        self.day_trade_dates = [d for d in self.day_trade_dates if d >= bas]
        self.day_trades_window_start = bas
        self.day_trades_count = len(self.day_trade_dates)
        return self.day_trades_count

    def pdt_unlock_date(self, today: date, lookback: int, limit: int) -> date | None:
        """Yeni day-trade yapılmazsa sayımın `limit` altına indiği ilk seans (kilitli değilse None)."""
        from nihai.data.market_calendar import NYSE

        def sayim(gun: date) -> int:
            bas = pdt_sessions(gun, lookback)[0]
            return sum(1 for d in self.day_trade_dates if d >= bas)

        if sayim(today) < limit:
            return None
        for s in NYSE.sessions_in_range(today + timedelta(days=1), today + timedelta(days=5 * lookback + 10)):
            if sayim(s.date()) < limit:
                return s.date()
        return None

    def beta_exposure(self) -> dict:
        """Madde 14'ün BETA-AĞIRLIKLI hâli (BULGU-59-6) — dış raporun D4 maddesi.

        Yürürlükteki `max_net_directional_pct` NOMİNAL toplar: 1,0$ SPY ile 1,0$ yüksek-beta
        bir isim aynı sayılır. Piyasa stresinde ikisi aynı şey DEĞİLDİR. Bu fonksiyon
        maruziyeti beta ile ağırlıklandırır — **ama uydurmaz**: betası ÖLÇÜLMEMİŞ pozisyon
        (`beta is None`) ayrı sayılır ve toplama GİRMEZ. Dönen sözlük:
          net_beta_dollars   : Σ signed_notional × beta (yalnız ölçülmüş pozisyonlar)
          gross_beta_dollars : Σ |signed_notional × beta|
          net_notional       : Σ signed_notional (TÜM pozisyonlar — mevcut tavanın tabanı)
          measured / unmeasured / coverage : ölçüm koşulu (Tur-34: koşul taşınmalıdır)
        `coverage < 1` iken net_beta_dollars portföyün TAMAMINI temsil ETMEZ."""
        net_beta = gross_beta = 0.0
        measured = unmeasured = 0
        for p in self.open_positions:
            if p.beta is None:
                unmeasured += 1
                continue
            measured += 1
            contrib = p.signed_notional * float(p.beta)
            net_beta += contrib
            gross_beta += abs(contrib)
        total = measured + unmeasured
        return {
            "net_beta_dollars": net_beta,
            "gross_beta_dollars": gross_beta,
            "net_notional": sum(p.signed_notional for p in self.open_positions),
            "measured": measured,
            "unmeasured": unmeasured,
            "coverage": (measured / total) if total else 1.0,
        }

    def roll_session_if_new_day(self, today: date) -> bool:
        """Yeni seans günü başladıysa günlük sayaçları sıfırla. True → sıfırlandı.

        daily_pnl ve consecutive_stops GÜNLÜK sayaçlardır; günlük-max-kayıp freni ve
        ardışık-stop cooldown'u her yeni günde temiz başlamalı. (Strateji-düzeyi koruma
        kill-switch'tir; o rolling-50 işlem üzerinden günler-arası kalıcıdır.)"""
        if self.last_session_date < today:
            self.daily_pnl = 0.0
            self.consecutive_stops = 0
            self.last_session_date = today
            return True
        return False


# ---------------------------------------------------------------------------
# Risk kararları
# ---------------------------------------------------------------------------
@dataclass
class RiskCheck:
    approved: bool
    reason: str
    sizing: dict = field(default_factory=dict)


class RiskManager:
    def __init__(self, state: PortfolioState):
        self.state = state
        # A4 — piyasa rejimi risk çarpanı (1.0 = etkisiz) + etiket (karar-desteği için
        # panoda/uyarıda gösterilir). Canlı döngü compute_regime ile günceller; backtest
        # ise regime_multiplier_series'i bar-bazlı uygular (etiket orada kullanılmaz).
        self.regime_multiplier: float = 1.0
        self.regime_label: str = "—"
        # B1 — FRED makro risk-off çarpanı (1.0 = etkisiz). enable_macro_gate açık + FRED
        # key varken refresh_macro() günceller; aksi halde nötr kalır (sistem değişmez).
        self.macro_multiplier: float = 1.0
        self.macro_label: str = "—"

    # Madde 9 — Adaptif risk: ATH'den %X düştüyse riski yarıya in.
    # A4 — rejim çarpanı (CONFIG.regime açıksa) ÇARPIMSAL istiflenir.
    def _current_risk_pct(self, base_pct: float) -> float:
        pct = base_pct
        if self.state.ath_equity > 0:
            dd = (self.state.ath_equity - self.state.equity) / self.state.ath_equity
            if dd >= CONFIG.risk.drawdown_throttle_threshold:
                pct *= CONFIG.risk.drawdown_risk_multiplier
        if CONFIG.regime.enable_regime_risk:
            pct *= self.regime_multiplier
        # B1 — makro risk-off freni (varsayılan kapalı; çarpan 1.0 ise zaten etkisiz)
        if CONFIG.macro.enable_macro_gate:
            pct *= self.macro_multiplier
        return pct

    # B1 — makro çarpanı FRED snapshot'ından tazele. Canlı döngüde periyodik çağrılır
    # (refresh_minutes içinde önbellekli). Kapı kapalı/keysiz → nötr (1.0). Ağ hatası
    # asla sizing'i bozmaz (get_macro_snapshot sessizce nötre düşer).
    def refresh_macro(self) -> None:
        if not CONFIG.macro.enable_macro_gate:
            self.macro_multiplier, self.macro_label = 1.0, "—"
            return
        try:
            from nihai.data.macro_calendar import get_macro_snapshot
            snap = get_macro_snapshot()
            self.macro_multiplier = snap.multiplier
            self.macro_label = snap.label
        except Exception:
            self.macro_multiplier, self.macro_label = 1.0, "—"

    # Madde 10 — Günlük freni
    def daily_loss_breach(self) -> bool:
        pct = CONFIG.risk.daily_max_loss_pct
        if pct <= 0:
            return False  # 0/negatif -> gunluk kayip freni devre disi (DAILY_MAX_LOSS_PCT)
        return self.state.daily_pnl <= -self.state.equity * pct

    def consecutive_stops_cooldown(self) -> bool:
        n = CONFIG.risk.consecutive_stops_cooldown
        if n <= 0:
            return False  # 0/negatif → soğutma devre dışı
        return self.state.consecutive_stops >= n

    # Madde 10, 23 — PDT
    def pdt_blocks_day_trade(self) -> bool:
        """NLV < $25k iken 5 iş günlük day-trade bütçesi dolduysa True (Tur-62'de BAĞLANDI).

        **Neden GİRİŞİ kapatır, ÇIKIŞI değil.** FINRA kuralı kapanış bacağını sayar; yani
        harfiyen okunursa fren çıkışa konurdu. Konmadı ve konmamalı: bu sistemin her hisse
        girişi bir STOP ile açılır (`size_stock(entry, stop, ...)`). Bütçe bittikten sonra
        açılan pozisyonun stop'u aynı gün tetiklenirse bot ya kuralı çiğner ya da stop'u
        yutar — ikincisi sermaye kaybıdır. İki kuralı da AYNI ANDA sağlanabilir bırakan tek
        hamle pozisyonu hiç açmamaktır. Bu yüzden fren, kardeşleriyle (günlük kayıp,
        ardışık stop) aynı yerde ve aynı yönde durur: yalnız maruziyeti AZALTIR.

        Ölçüldü (Tur-62, `data/trades.csv` 544 kapanış, 2026-07-06→08-31): kapanışların
        **%82,5'i (449) aynı-gün** round-trip, 13 günde günlük 3 sınırı aşılmış (azami 89).
        Yani bu strateji küçük hesapta pratikte KOŞAMAZ. Canlı hesapta fren yine de ATIL:
        `nlv_after` aralığı 294.296$–645.264$, eşiğin 11,8 katı — bağlanması BİT-DENK.
        Atıl olması sessiz kalmasını gerektirmez; `.env.example` `INITIAL_EQUITY=25000`
        ile geliyor ve tek bir kayıp eşiğin altına düşürür."""
        if not CONFIG.regulation.pdt_enabled:
            return False  # PDT ön-kontrolü devre dışı (config.regulation.pdt_enabled)
        if self.state.equity >= CONFIG.regulation.pdt_threshold_equity:
            return False
        # NLV < $25k → 5 iş günü içinde max 3 day-trade (sınır: BULGU-62-4'te 10'dan 3'e).
        # Tur-90 (BULGU-90-4): Tur-62'nin ponytail notu "hesap eşiğin altına inerse iş-günü
        # takvimine bağla" diyordu; hesap 22 Eyl'de 19,5 k$'a indi. Pencere artık yürüyen son N
        # NYSE seansıdır ve gün ET'dir (eskiden `date.today()` = makinenin TR tarihi, 7 sa ileri).
        from nihai.data.market_calendar import now_et_naive
        n = self.state.pdt_refresh(now_et_naive().date(), CONFIG.regulation.pdt_lookback_days)
        return n >= CONFIG.regulation.pdt_max_day_trades_under_threshold

    # Madde 9, 10, 23 — YENİ POZİSYON frenleri TEK YERDE (Tur-62).
    # `size_stock` ve `size_long_option` bu üçünü ayrı ayrı kopyalıyordu; PDT ise
    # HİÇBİRİNDE yoktu. Tur-28'in dersi ("bir kural iki kapıdan yalnız birine konursa
    # yalnız birinde geçerlidir") burada YAPIYA çevrildi: bundan sonra eklenen bir fren
    # iki yola da düşer, çünkü düşmemesi için ayrı bir hamle gerekir.
    def _entry_blocked(self) -> str | None:
        """Yeni pozisyon açmayı engelleyen fren varsa SEBEBİ, yoksa None."""
        if self.daily_loss_breach():
            return "DAILY_MAX_LOSS_HIT"
        if self.consecutive_stops_cooldown():
            return "CONSECUTIVE_STOPS_COOLDOWN"
        if self.pdt_blocks_day_trade():
            return "PDT_DAY_TRADE_LIMIT"
        return None

    # Madde 14 (beta tarafı) — BULGU-59-6
    def beta_hedge_suggestion(self, bench_price: float | None = None) -> dict | None:
        """Portföy beta maruziyeti eşiği aşıyorsa ÖNERİLEN endeks hedge bacağı; yoksa None.

        Bu sistem EMİR GÖNDERMEZ: dönen şey bir emir değil, operatöre gösterilecek bir
        ÖLÇÜM + öneridir (Madde 47'nin `/dur` freniyle aynı duruş). Eşik için İKİNCİ bir
        sabit tanımlanmaz — `max_net_directional_pct` okunur; iki ayrı yönsel-maruziyet
        eşiği tanımlamak Tur-50'nin "çelişen iki tanım" kusurunu yeniden üretirdi.

        `coverage < 1` (bazı pozisyonların betası ölçülmemiş) durumunda öneri yine
        üretilir ama `coverage` ALANIYLA BİRLİKTE — okuyan eksik ölçümü görmeden sayıyı
        portföyün tamamı sanamaz."""
        eq = self.state.equity
        if eq <= 0:
            return None
        exp = self.state.beta_exposure()
        if exp["measured"] == 0:
            return None
        limit = eq * CONFIG.portfolio.max_net_directional_pct
        net = exp["net_beta_dollars"]
        if abs(net) <= limit:
            return None
        excess = abs(net) - limit
        out = {
            "benchmark": CONFIG.regime.benchmark,
            "side": "SHORT" if net > 0 else "LONG",   # ters bacak
            "notional": excess,
            "net_beta_dollars": net,
            "limit_dollars": limit,
            "coverage": exp["coverage"],
            "unmeasured": exp["unmeasured"],
            "shares": None,
        }
        if bench_price and bench_price > 0:
            out["shares"] = int(excess // float(bench_price))
        return out

    # Madde 14, 12 — Portföy tavanları
    def portfolio_caps_breach(
        self, new_risk: float, new_sector: str,
        *, new_signed_notional: float = 0.0, new_symbol: str | None = None,
        new_sign: int = 0,
    ) -> tuple[bool, str]:
        caps = CONFIG.portfolio
        # (a) Açık pozisyon sayısı tavanı
        if len(self.state.open_positions) >= caps.max_open_positions:
            return True, f"MAX_OPEN_POSITIONS_{caps.max_open_positions}"
        # (b) Brüt risk tavanı
        gross_risk = sum(p.risk_dollars for p in self.state.open_positions) + new_risk
        if gross_risk > self.state.equity * caps.max_gross_risk_pct:
            return True, f"GROSS_RISK_CAP_{caps.max_gross_risk_pct*100:.0f}%"
        # (c) Sektör/korelasyon tavanı (Madde 12)
        sector_count = sum(1 for p in self.state.open_positions if p.sector == new_sector)
        if sector_count >= caps.max_per_sector_positions:
            return True, f"SECTOR_CAP_{caps.max_per_sector_positions}_{new_sector}"
        # (d) Net yönsel maruziyet tavanı (Madde 14) — daha önce config'de tanımlı ama uygulanmıyordu
        net_signed = sum(p.signed_notional for p in self.state.open_positions) + new_signed_notional
        if abs(net_signed) > self.state.equity * caps.max_net_directional_pct:
            return True, (f"NET_DIRECTIONAL_CAP_{caps.max_net_directional_pct*100:.0f}%"
                          f"_{net_signed/max(self.state.equity,1e-9)*100:+.0f}%")
        # (e) Sembol başına aynı yönde açık pozisyon tavanı (Madde 12) — config'de ölüydü
        if new_symbol is not None and new_sign != 0:
            same_dir_same_sym = sum(
                1 for p in self.state.open_positions
                if p.symbol == new_symbol and p.directional_sign == new_sign
            )
            if same_dir_same_sym >= CONFIG.safety.max_signals_per_symbol_open:
                return True, (f"SYMBOL_CLUSTER_CAP_{CONFIG.safety.max_signals_per_symbol_open}"
                              f"_{new_symbol}")
        return False, "OK"

    # Madde 27 — Greek bütçesi
    def greek_caps_breach(self, prospective: GreekSet, contracts: int) -> tuple[bool, list[str]]:
        agg = PortfolioGreeks()
        for p in self.state.open_positions:
            if p.greeks:
                agg.add_position(p.greeks, p.quantity)
        agg.add_position(prospective, contracts)
        ok, breaches = agg.within_caps(CONFIG.greeks)
        return (not ok, breaches)

    # Madde 13 — Marjin kontrolü AŞAĞIDA, `_magnitude_cap_units` (a) maddesindedir.
    # Buradaki `margin_ok(notional) -> bool` ayrı bir yöntem olarak duruyordu ve depoda
    # HİÇBİR çağıranı yoktu (Tur-50). Aynı Reg T bağını iki kez tanımlamak iki farklı
    # DAVRANIŞ tanımlamak demekti: `margin_ok` adayı REDDEDER, `_magnitude_cap_units`
    # ise tavana sığacak şekilde KÜÇÜLTÜR (`cap_aware_sizing`) — ve küçültme bilinçli
    # olarak seçilmiş olandır. Ölü olan kopya silindi; kural tek yerde yaşıyor.

    # Madde 13, 14 — Büyüklük tavanlarının izin verdiği MAKSİMUM birim adedi.
    # Sayı tavanları (max_open / sektör / sembol kümelenmesi) BURADA YOK — onlar
    # küçültülemez, ayrı hard-reject olarak kalır. Burası yalnızca büyüklük (notional /
    # risk $) tavanlarını birime çevirir ki cap-aware boyutlama pozisyonu sığacak
    # şekilde küçültebilsin.
    #   unit_notional   : 1 birimin (1 hisse / 1 kontrat) nominal değeri ($)
    #   unit_risk       : 1 birimin risk katkısı ($)
    def _magnitude_cap_units(self, unit_notional: float, unit_risk: float,
                             sign: int) -> tuple[float, str]:
        caps = CONFIG.portfolio
        eq = self.state.equity
        limits: list[tuple[float, str]] = []

        # (a) Buying power (Reg T): Σ|nominal| × başlangıç marjı ≤ özkaynak, yani brüt ≤ 2×özkaynak.
        # Tur-89 (BULGU-89-2): eskiden `2×özkaynak / (marj × birim)` = POZİSYON BAŞINA 4× idi ve açık
        # pozisyonlar hiç düşülmüyordu → brüt 7,8 M$ / özkaynak 226 k$ = 35× ölçüldü (2026-09-15).
        # ponytail: gün-içi PDT 4× alım gücü modellenmedi (muhafazakâr); gerekirse eşik NLV≥25k'ya bağla.
        margin = CONFIG.regulation.reg_t_initial_margin
        if unit_notional > 0 and margin > 0:
            gross0 = sum(abs(p.signed_notional) for p in self.state.open_positions)
            limits.append(((eq / margin - gross0) / unit_notional,
                           "INSUFFICIENT_BUYING_POWER"))

        # (b) Net yönsel headroom: |net0 + units×unit_notional×sign| ≤ equity×cap
        #     allowed = (L − sign×net0) / unit_notional   (türetme README/risk_manager not)
        if unit_notional > 0 and sign != 0:
            net0 = sum(p.signed_notional for p in self.state.open_positions)
            limit = eq * caps.max_net_directional_pct
            limits.append(((limit - sign * net0) / unit_notional,
                           f"NET_DIRECTIONAL_CAP_{caps.max_net_directional_pct*100:.0f}%"))

        # (c) Brüt risk headroom: gross0 + units×unit_risk ≤ equity×gross_cap
        if unit_risk > 0:
            gross0 = sum(p.risk_dollars for p in self.state.open_positions)
            limits.append(((eq * caps.max_gross_risk_pct - gross0) / unit_risk,
                           f"GROSS_RISK_CAP_{caps.max_gross_risk_pct*100:.0f}%"))

        # (d) Pozisyon başına nominal tavanı: units × unit_notional ≤ cap
        if unit_notional > 0 and caps.max_position_notional > 0:
            limits.append((caps.max_position_notional / unit_notional,
                           f"MAX_POSITION_NOTIONAL_{caps.max_position_notional:.0f}"))

        max_units = float("inf")
        binding = "OK"
        for val, name in limits:
            if val < max_units:
                max_units, binding = val, name
        return max_units, binding

    def _apply_cap_aware(self, target_units: int, unit_notional: float,
                         unit_risk: float, sign: int) -> tuple[int, str]:
        """Risk-hedef birim sayısını büyüklük tavanlarına göre küçült.

        Dönüş: (nihai_birim, reason). nihai_birim ≤ 0 ise reason bağlayıcı tavandır.
        cap_aware_sizing kapalıyken küçültme gerekiyorsa 0 döndürüp reddettirir."""
        cap_units_f, binding = self._magnitude_cap_units(unit_notional, unit_risk, sign)
        cap_units = target_units if cap_units_f == float("inf") else int(cap_units_f)
        units = min(target_units, max(cap_units, 0))
        if units <= 0:
            return 0, binding
        if not CONFIG.portfolio.cap_aware_sizing and units < target_units:
            return 0, binding   # eski davranış: küçültme yerine reddet
        min_notional = CONFIG.portfolio.min_position_notional
        if min_notional > 0 and units * unit_notional < min_notional:
            return 0, f"BELOW_MIN_POSITION_{min_notional:.0f}"
        return units, "OK"

    # ---------------------------------------------------------------------
    # Hisse pozisyon boyutu
    # ---------------------------------------------------------------------
    def size_stock(self, entry: float, stop: float, sector: str = "UNKNOWN",
                   *, symbol: str | None = None, side: str = "LONG",
                   confidence_multiplier: float = 1.0) -> RiskCheck:
        blocked = self._entry_blocked()
        if blocked:
            return RiskCheck(False, blocked)

        # A5 — düşük-güven kademeli boyutlama: risk × güven çarpanı (≤1; 0..1'e kırpılır)
        conf = min(max(confidence_multiplier, 0.0), 1.0)
        risk_pct = self._current_risk_pct(CONFIG.risk.per_trade_risk_pct_directional)
        risk_dollars = self.state.equity * risk_pct * conf
        # SHORT'ta stop > entry; mesafe mutlak değerle alınır (aksi halde 0.01'e kırpılıp
        # devasa pozisyon üretiyordu).
        risk_per_share = max(abs(entry - stop), 0.01)
        target_shares = int(risk_dollars // risk_per_share)
        if target_shares <= 0:
            return RiskCheck(False, "ZERO_SIZE")

        # Madde 13, 14 — büyüklük tavanlarına SIĞACAK şekilde küçült (reddetme yerine).
        sign = +1 if side == "LONG" else -1
        shares, why = self._apply_cap_aware(
            target_shares, unit_notional=entry, unit_risk=risk_per_share, sign=sign)
        if shares <= 0:
            return RiskCheck(False, why)

        notional = shares * entry
        # Sayı tavanları (max_open / sektör / sembol kümelenmesi) küçültülemez → hard-reject.
        breach, why = self.portfolio_caps_breach(
            shares * risk_per_share, sector,
            new_signed_notional=notional * sign, new_symbol=symbol, new_sign=sign,
        )
        if breach:
            return RiskCheck(False, why)

        return RiskCheck(True, "OK", sizing={
            "shares": shares, "risk_dollars": shares * risk_per_share,
            "notional": notional, "risk_pct_used": risk_pct,
            "downsized": shares < target_shares, "target_shares": target_shares,
            "confidence_mult": conf,
        })

    # ---------------------------------------------------------------------
    # Opsiyon pozisyon boyutu (uzun opsiyon, primin tamamı = risk)
    # ---------------------------------------------------------------------
    def size_long_option(self, premium: float, greeks: GreekSet, sector: str = "UNKNOWN",
                         *, symbol: str | None = None, option_type: str = "c",
                         confidence_multiplier: float = 1.0) -> RiskCheck:
        blocked = self._entry_blocked()
        if blocked:
            return RiskCheck(False, blocked)

        # A5 — düşük-güven kademeli boyutlama (uzun opsiyonda da risk bütçesini ölçekler)
        conf = min(max(confidence_multiplier, 0.0), 1.0)
        risk_pct = self._current_risk_pct(CONFIG.risk.per_trade_risk_pct)
        risk_dollars = self.state.equity * risk_pct * conf

        # Madde 32 — sabit ücretler dahil
        fee_per_contract = sum(OPTION_FEES.values())
        cost_per_contract = premium * 100 + fee_per_contract
        if cost_per_contract <= 0:
            return RiskCheck(False, "INVALID_PREMIUM")

        target_contracts = int(risk_dollars // cost_per_contract)
        if target_contracts <= 0:
            return RiskCheck(False, "ZERO_CONTRACTS")

        # Madde 14 — long call boğa (+), long put ayı (−)
        sign = +1 if option_type == "c" else -1
        # Madde 13, 14 — büyüklük tavanlarına sığacak şekilde küçült. Opsiyon nominali
        # prim×100×kontrat; birim risk kontrat maliyeti (prim×100 + ücretler).
        contracts, why = self._apply_cap_aware(
            target_contracts, unit_notional=premium * 100,
            unit_risk=cost_per_contract, sign=sign)
        if contracts <= 0:
            return RiskCheck(False, why)

        total_risk = contracts * cost_per_contract
        signed_notional = contracts * premium * 100 * sign
        breach, why = self.portfolio_caps_breach(
            total_risk, sector,
            new_signed_notional=signed_notional, new_symbol=symbol, new_sign=sign,
        )
        if breach:
            return RiskCheck(False, why)

        gbreach, gwhy = self.greek_caps_breach(greeks, contracts)
        if gbreach:
            return RiskCheck(False, "GREEK_CAP: " + " | ".join(gwhy))

        return RiskCheck(True, "OK", sizing={
            "contracts": contracts, "risk_dollars": total_risk,
            "fees_total": contracts * fee_per_contract, "risk_pct_used": risk_pct,
            "downsized": contracts < target_contracts, "target_contracts": target_contracts,
            "confidence_mult": conf,
        })
