import hmac
import json
import os
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from nihai.atomic_io import atomic_write_text, tek_yazar_kilidi
from nihai.config import (
    CONFIG,
    DATA_DIR,
    DATA_PROVIDER,
    INITIAL_EQUITY,
    LOG_DIR,
    OPTION_MODE,
    WEB_CORS_ORIGINS,
    WEB_HOST,
    WEB_PORT,
    WEB_TOKEN,
    get_sector,
)
from nihai.console import ensure_utf8_stdout
from nihai.data.data_provider import get_provider
from nihai.data.market_calendar import is_market_open, kapanmis_gunluk, now_et_naive
from nihai.execution.concurrency import PORTFOLIO_LOCK
from nihai.execution.state import StateStore
from nihai.risk.options_analyzer import GreekSet, PortfolioGreeks
from nihai.risk.position_manager import PositionManager, SlippageModel, sabit_hedefle_cikar
from nihai.runtime_strategy import settings as strategy_settings
from nihai.signals.strategy import Signal, SignalAction, is_quant_label

# Setup FastAPI App
app = FastAPI(title="Nihai Algoritma - Premium Web Panel")

# CORS: '*' + allow_credentials tarayıcılarca reddedilen geçersiz kombinasyondur. Panel
# yerel olduğundan origin'ler localhost'a kısıtlanır (WEB_CORS_ORIGINS ile genişletilebilir).
# Yine de '*' istenirse credentials kapatılır ki spec ihlali olmasın.
_cors_wildcard = "*" in WEB_CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=WEB_CORS_ORIGINS,
    allow_credentials=not _cors_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Madde 44 — Web token koruması. WEB_TOKEN doluysa tüm mutasyon uçları (POST/PUT/
# DELETE/PATCH, /api/*) X-Auth-Token ister; pano JS same-origin token'ı otomatik gönderir
# (bkz. get_dashboard enjeksiyonu). WEB_TOKEN boşsa enforcement YOK → mevcut davranış
# birebir korunur (yalnızca localhost binding geçerli).
_MUTATING_METHODS = {"POST", "PUT", "DELETE", "PATCH"}

# Tur-58 (dış rapor §F) — token karşılaştırması ve KABA KUVVET freni.
#
# İki gerçek kusur vardı: (1) `!=` karşılaştırması ilk farklı baytta döner → uzaktan
# ölçülebilir bir zamanlama kanalıdır; (2) yanlış token için bir DENEME SINIRI yoktu,
# yani token uzunluğu ne olursa olsun sınırsız deneme yapılabiliyordu.
#
# **Yapılmayan ve neden:** 2FA / oturum süresi / RBAC. Bu pano `WEB_HOST` ile localhost'a
# bağlanan TEK operatörlü bir araçtır; rol yoktur, oturum yoktur, ikinci faktörü
# doğrulayacak kimlik sağlayıcısı yoktur. Onları eklemek koruma değil BAKIM YÜKÜ olurdu
# (YAGNI). Pano dış ağa açılacaksa doğru katman ters vekil (reverse proxy) + mTLS'tir,
# uygulama içi rol tablosu değil — bu karar burada yazılıdır ki bir sonraki tur yeniden
# tartışmasın.
_AUTH_FAIL_MAX = 10                 # bu kadar ardışık hatadan sonra
_AUTH_LOCK_SEC = 60.0               # bu kadar saniye kilitle (istemci IP'si başına)
_auth_fails: dict[str, tuple[int, float]] = {}      # ip → (ardışık hata, kilit bitişi)
_auth_lock = threading.Lock()


def _auth_ok(token: str, client: str) -> tuple[bool, str]:
    """(kabul, gerekçe). Sabit-zamanlı karşılaştırma + IP başına kilit."""
    now = time.monotonic()
    with _auth_lock:
        fails, until = _auth_fails.get(client, (0, 0.0))
        if now < until:
            return False, "KILITLI"
        ok = hmac.compare_digest(token or "", WEB_TOKEN)
        if ok:
            _auth_fails.pop(client, None)
            return True, "OK"
        fails += 1
        until = now + _AUTH_LOCK_SEC if fails >= _AUTH_FAIL_MAX else 0.0
        _auth_fails[client] = (fails, until)
        return False, ("KILITLENDI" if until else "GECERSIZ")


@app.middleware("http")
async def _web_token_guard(request: Request, call_next):
    if (
        WEB_TOKEN
        and request.method in _MUTATING_METHODS
        and request.url.path.startswith("/api/")
    ):
        client = request.client.host if request.client else "?"
        ok, why = _auth_ok(request.headers.get("X-Auth-Token", ""), client)
        if not ok:
            logger.warning(f"[auth] {client} {request.url.path} reddedildi ({why})")
            return JSONResponse(
                {"success": False, "message": "Yetkisiz: geçersiz veya eksik X-Auth-Token.",
                 "reason": why},
                status_code=429 if why in ("KILITLI", "KILITLENDI") else 401,
            )
    return await call_next(request)


@app.get("/metrics")
def metrics():
    """Prometheus/OpenTelemetry düz-metin sergisi (dış rapor §H).

    Salt-okunur ve sır İÇERMEZ (süre, RSS/CPU, çağrı sayaçları, döngü gecikmesi) →
    mutasyon kapısının dışındadır, GET uçlarıyla aynı sözleşmededir."""
    from nihai.observability import BUDGET, prometheus_text
    BUDGET.observe()        # pano yoklaması da pencereyi ilerletir (tarayıcı kapalıyken de ölçüm)
    return Response(prometheus_text(), media_type="text/plain; version=0.0.4; charset=utf-8")


class ClosePositionRequest(BaseModel):
    symbol: str
    asset_type: str
    expiry: str | None = None
    strike: float | None = None
    option_type: str | None = None

def get_portfolio_status():
    store = StateStore()
    portfolio = store.load_or_init(INITIAL_EQUITY)
    today = now_et_naive().date()
    store.settle_matured(today)
    return store, portfolio


# Tur-106 (BULGU-106-30) — operatör eylemleri CANLI portföye uygulanır.
# `concurrency.PORTFOLIO_LOCK` panonun ana döngüyle AYNI portföyü değiştirdiğini varsayar; uçlar ise diskten ayrı bir
# kopya yükleyip ona yazıyordu. Ana döngü belleğindeki portföyü her turda (~60 sn) diske geri yazdığı için panodan
# kapatılan pozisyon geri geliyor (sonra İKİNCİ kez kapanıp deftere çift satır yazıyor), stop/hedef düzenlemesi ve
# kill-switch sıfırlaması da siliniyordu.
_CANLI: dict = {}


def canli_baglam_kaydet(store: StateStore, pmgr: PositionManager) -> None:
    """Ana süreç (`main.run_loop`, `--serve-web` ile aynı süreç) canlı durum deposunu ve pozisyon yöneticisini bildirir."""
    _CANLI.update(store=store, pmgr=pmgr)


def _yazma_baglami() -> tuple:
    """Operatör eyleminin uygulanacağı `(store, portfolio, pmgr, canli)`; PORTFOLIO_LOCK altında çağrılır.

    Kayıtlı canlı bağlam varsa O nesneler döner (kapanış defterini pmgr'ın kancası tutar). Yoksa pano tek başınadır:
    ana süreç başka bir süreçte koşuyorsa (`data/main.lock` tutuluyor) disk kopyasına yazmak ezileceği için eylem
    REDDEDİLİR; hiçbir ana süreç yoksa disk tek doğruluk kaynağıdır ve eski yol (yükle → değiştir → kaydet) geçerlidir."""
    if _CANLI:
        store = _CANLI["store"]
        return store, store.portfolio, _CANLI["pmgr"], True
    kilit = tek_yazar_kilidi(DATA_DIR / "main.lock")
    if kilit is None:
        raise HTTPException(status_code=409, detail=(
            "Ana süreç çalışıyor ve bu pano onun canlı portföyüne bağlı değil (ayrı süreç ya da henüz başlıyor): "
            "buradan yapılan değişiklik bir sonraki turda ezilirdi. İşlemi ana sürecin panosundan yapın."))
    kilit.close()
    store = StateStore()
    portfolio = store.load_or_init(INITIAL_EQUITY)
    pmgr = PositionManager(portfolio, SlippageModel())
    pmgr.rehydrate_trackers()
    return store, portfolio, pmgr, False


_EMPTY_TRADES = {
    "stats": {"win_rate": 0, "total_pnl": 0, "total_trades": 0,
              "profit_factor": 0, "wins": 0, "losses": 0},
    "trades": [], "equity_curve": [],
}


def _duration_label(entry_iso: str, exit_iso: str) -> str:
    """'10 Gun 1 Saat 44 Dakika' — log_trade_postmortem'in insan-okur süre biçimi."""
    from datetime import datetime
    try:
        d = datetime.fromisoformat(exit_iso) - datetime.fromisoformat(entry_iso)
    except (ValueError, TypeError):
        return "—"
    hrs, rem = divmod(d.seconds, 3600)
    return f"{d.days} Gun {hrs} Saat {rem // 60} Dakika"


def parse_trade_log():
    """Kapanan işlem defteri — KAYNAK: data/trades.csv (makine-okur, append-only).

    ÖNCESİ logs/trade_log.txt bloklarını ayrıştırıyordu ve pano defteri BOŞTU: yazan
    (`signal_engine.log_trade_postmortem`) başlık ile alan satırları ARASINA da bir
    '='*70 ayracı koyuyor, ayrıştırıcı ise metni aynı ayraçla bölüp yalnız
    'ISLEM TAMAMLANDI RAPORU' geçen bloğu işliyordu → marker başlık bloğunda, alanlar
    BİR SONRAKİ blokta kalıyor, her kayıt sessizce düşüyordu (330 işlem → 0). Testler
    kaçırdı çünkü fixture o orta ayracı içermiyordu (gerçek yazan biçimi değildi).

    Metin yerine `log_trade_csv`'nin AYNI kapanışta yazdığı CSV okunur: ayrıştırma
    kırılgan değil, alanlar tipli, ve trade_log.txt'in aksine CSV rotasyona uğramaz
    (txt boyut eşiğinde arşive taşınır → geçmiş defterden kaybolurdu)."""
    import csv as _csv

    path = DATA_DIR / "trades.csv"
    if not path.exists():
        return dict(_EMPTY_TRADES, stats=dict(_EMPTY_TRADES["stats"]))
    try:
        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        trades, equity_curve = [], []
        total_pnl = total_gain = total_loss = 0.0
        wins = losses = 0

        with path.open(encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                pnl = _f(row.get("realized_pnl"))
                if pnl is None:
                    continue                       # kapanmamış/bozuk satır — sayma
                qty, side = row.get("quantity", ""), row.get("side", "")
                kind = "Option" if row.get("asset_type") == "option" else "Stock"
                label = " ".join(p for p in (side, kind, f"x {qty}") if p)
                mae, mfe = _f(row.get("mae_pct")), _f(row.get("mfe_pct"))
                nlv = _f(row.get("nlv_after"))
                entry_t = (row.get("entry_time") or "").replace("T", " ")[:19]
                exit_t = (row.get("exit_time") or "").replace("T", " ")[:19]
                trades.append({
                    "asset": f"{row.get('symbol', '')} ({label})".strip(),
                    "symbol": row.get("symbol", ""),
                    "side": side,
                    "sector": row.get("sector", ""),
                    "strategy": row.get("strategy", ""),
                    "duration": _duration_label(row.get("entry_time", ""),
                                                row.get("exit_time", "")),
                    "entry_time": entry_t,
                    "exit_time": exit_t,
                    "entry_price": _f(row.get("entry_price")),
                    "exit_price": _f(row.get("exit_price")),
                    "exit_trigger": row.get("exit_trigger", ""),
                    "mae_mfe": (f"{mae:+.1f}% / {mfe:+.1f}%"
                                if mae is not None and mfe is not None else ""),
                    "r_multiple": _f(row.get("r_multiple")),
                    "pnl": round(pnl, 2),
                    "nlv_after": f"{nlv:,.2f} USD" if nlv is not None else "",
                })
                total_pnl += pnl
                if pnl >= 0:
                    wins += 1
                    total_gain += pnl
                else:
                    losses += 1
                    total_loss += abs(pnl)
                if nlv is not None:
                    equity_curve.append({"time": exit_t, "equity": nlv,
                                         "asset": row.get("symbol", "")})

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        profit_factor = (total_gain / total_loss) if total_loss > 0 else (
            total_gain if total_gain > 0 else 1.0)

        trades.reverse()                            # en yeni işlem başta
        return {
            "stats": {
                "win_rate": round(win_rate, 2),
                "total_pnl": round(total_pnl, 2),
                "total_trades": total_trades,
                "profit_factor": round(profit_factor, 2),
                "wins": wins,
                "losses": losses,
            },
            "trades": trades[:50],
            "equity_curve": equity_curve,
        }
    except Exception as e:
        logger.error(f"Trade log parsing error: {e}")
        return dict(_EMPTY_TRADES, stats=dict(_EMPTY_TRADES["stats"]))

@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    # no-store: tarayıcı panoyu önbelleğe almasın → kod değişince eski sayfa takılı
    # kalmaz (yeni sekme/sütunlar hep güncel gelir).
    nocache = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
               "Pragma": "no-cache", "Expires": "0"}
    html_path = Path(__file__).parent / "web_dashboard.html"
    if html_path.exists():
        html = html_path.read_text(encoding="utf-8")
        html = html.replace("<!-- LAB_PANELS -->", html_path.with_name("lab_dashboard.html").read_text(encoding="utf-8"))
        # Same-origin token enjeksiyonu: pano JS apiCall'da X-Auth-Token olarak gönderir.
        # Token yalnızca bu (same-origin) sayfaya gömülür; CORS yanıt gövdesini okumayı
        # engellediğinden çapraz-origin sayfalar token'ı okuyamaz.
        token_script = f"<script>window.__WEB_TOKEN__ = {json.dumps(WEB_TOKEN)};</script>"
        html = html.replace("</head>", token_script + "\n</head>", 1)
        return HTMLResponse(html, headers=nocache)
    return HTMLResponse("<h1>Web Dashboard HTML dosyası bulunamadı. Lütfen "
                        "web_dashboard.html dosyasını oluşturun.</h1>", headers=nocache)


@app.get("/lab.js")
def lab_script():
    return FileResponse(Path(__file__).with_name("lab_dashboard.js"), media_type="text/javascript",
                        headers={"Cache-Control": "no-store"})

@app.get("/i18n.js")
def i18n_script():
    # [Tur-108] İngilizce arayüz: sözlük + motor TEK yanıtta. Betik <head>'de eşzamanlı yüklenir →
    # İngilizce kipte Türkçe metin hiç boyanmaz; Türkçe kipte motor yalnız dil düğmesini ekler.
    kok = Path(__file__).parent
    govde = ("window.__PANO_EN=" + (kok / "pano_en.json").read_text(encoding="utf-8") + ";\n"
             + (kok / "pano_i18n.js").read_text(encoding="utf-8"))
    return Response(govde, media_type="text/javascript", headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def get_status():
    try:
        with PORTFOLIO_LOCK:
            store, portfolio = get_portfolio_status()
            max_pos = CONFIG.portfolio.max_open_positions
            pos_count = len(portfolio.open_positions)
            capacity_pct = (pos_count / max_pos * 100) if max_pos > 0 else 0
            market_open = is_market_open()
            
            pdt_limit = CONFIG.regulation.pdt_max_day_trades_under_threshold
            # Tur-90 (BULGU-90-4): sayım yürüyen pencereden (son N NYSE seansı, ET) okunur ve açılış
            # günü SUNUCUDA hesaplanır — JS `toISOString()` yerel gece yarısını UTC'ye çevirip bir
            # gün geri yazıyordu ("2026-09-26 açılır" = Cumartesi; kodun sıfırladığı gün Pazar'dı).
            from nihai.data.market_calendar import now_et_naive
            _pdt_bugun = now_et_naive().date()
            pdt_count = portfolio.pdt_refresh(_pdt_bugun, CONFIG.regulation.pdt_lookback_days)
            _pdt_acilis = portfolio.pdt_unlock_date(_pdt_bugun, CONFIG.regulation.pdt_lookback_days, pdt_limit)

            # BULGU-62-2 — Tur-59'un beta hedge ÖNERİSİ operatöre hiç ulaşmıyordu:
            # `beta_hedge_suggestion` docstring'i "operatöre gösterilecek bir ÖLÇÜM +
            # öneri" diyor ama depodaki tek çağıranı TESTTİ (Tur-61 BULGU-61-2 ile aynı
            # sınıf: yazılıp okunmayan yüzey). Saf bellek-içi hesap (açık pozisyon
            # döngüsü) — uç zaten PORTFOLIO_LOCK altında, ek maliyet yok.
            #
            # `suggestion=None` İKİ ayrı şeydir ve tek alana yıkılamaz (Tur-60 dersi):
            # betası ölçülmüş pozisyon YOKSA öneri üretilemez, limit içindeyse GEREKMEZ.
            # `measured`/`coverage` bu yüzden ÖNERİYLE BİRLİKTE taşınır.
            from nihai.risk.risk_manager import RiskManager
            _exp = portfolio.beta_exposure()
            beta_hedge = {
                "suggestion": RiskManager(portfolio).beta_hedge_suggestion(),
                "net_beta_dollars": _exp["net_beta_dollars"],
                "limit_dollars": portfolio.equity * CONFIG.portfolio.max_net_directional_pct,
                "measured": _exp["measured"],
                "unmeasured": _exp["unmeasured"],
                "coverage": _exp["coverage"],
            }

            # Tur-106 (BULGU-106-7): pano "Kasa Değeri (NLV)" diye GERÇEKLEŞMİŞ özkaynağı, "Nakit Rezervi"
            # diye `cash_balance`ı basıyordu — ikincisi alımda hiç düşülmez (yalnız kapanışta += K/Z), yani
            # ikisi hep AYNI sayıydı (2 Eki: 12 uzun pozisyon, ~28 k$ nominal, ikisi de 19.500,30 $). NLV
            # piyasa değeridir: nakit + pozisyonların değeri = gerçekleşmiş + açık K/Z. Kurallar (PDT eşiği,
            # boyut) gerçekleşmiş özkaynakla çalışır → o da `equity` olarak ayrıca taşınır.
            acik_kz = portfolio.open_pnl
            bagli = sum(p.notional if p.side == "LONG" else -p.notional for p in portfolio.open_positions)
            return {
                "nlv": portfolio.equity + acik_kz,
                "equity": portfolio.equity,
                "open_pnl": acik_kz,
                "ath": portfolio.ath_equity,
                "cash": portfolio.cash_balance - bagli,
                "cash_committed": bagli,
                "unsettled_cash": portfolio.unsettled_cash,
                "daily_pnl": portfolio.daily_pnl,
                "consecutive_stops": portfolio.consecutive_stops,
                "consecutive_stops_limit": CONFIG.risk.consecutive_stops_cooldown,
                "pdt_enabled": CONFIG.regulation.pdt_enabled,
                "day_trades_count": pdt_count,
                "day_trades_limit": pdt_limit,
                # Tur-89: kural yalnız NLV < eşik iken bağlar ve pencere dolunca sıfırlanır; pano
                # bunlar olmadan "97 / 3 · %3233" basıyordu (kural uygulanıyor mu, ne zaman açılır?).
                "pdt_threshold": CONFIG.regulation.pdt_threshold_equity,
                "pdt_window_start": portfolio.day_trades_window_start.isoformat(),
                "pdt_lookback_days": CONFIG.regulation.pdt_lookback_days,
                "pdt_unlock_date": _pdt_acilis.isoformat() if _pdt_acilis else None,
                "beta_hedge": beta_hedge,
                "kill_switch_active": store.kill_switch_active,
                "position_count": pos_count,
                "max_open_positions": max_pos,
                "capacity_pct": round(capacity_pct, 1),
                "market_open": market_open,
                "active_strategy": strategy_settings()["strategy"],
                "option_mode": OPTION_MODE,
                "data_provider": DATA_PROVIDER,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/positions")
def get_positions():
    try:
        with PORTFOLIO_LOCK:
            _store, portfolio = get_portfolio_status()
            return [
                {
                    "symbol": pos.symbol,
                    "asset_type": pos.asset_type,
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "state_updated_at": _store.path.stat().st_mtime if _store.path.exists() else None,
                    "entry_time": pos.entry_time.isoformat() if hasattr(pos.entry_time, "isoformat") else str(pos.entry_time),
                    "stop": pos.stop,
                    # Tur-106 (BULGU-106-28): quant pozisyonu küresel bayraktan bağımsız izleyen stopla yönetilir ve
                    # sabit hedefle ÇIKMAZ → hedef nominaldir; pano bunu çıkış seviyesi gibi göstermesin.
                    "trailing_enabled": ((CONFIG.risk.enable_trailing or is_quant_label(pos.strategy))
                                         and pos.asset_type == "stock"),
                    "target": pos.target,
                    "hedef_nominal": not sabit_hedefle_cikar(pos.strategy),
                    "option_type": pos.option_type,
                    "strike": pos.strike,
                    "expiry": pos.expiry,
                    "sector": pos.sector,
                    "last_mark": pos.last_mark,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "notional": pos.notional
                }
                for pos in portfolio.open_positions
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/portfolio/timeline")
def portfolio_timeline():
    with PORTFOLIO_LOCK:
        store = StateStore()
        if not store.path.exists():
            return {"measured": False, "reason": "Portföy durum dosyası henüz yok.",
                    "settlements": [], "quarantine": []}
        # Bozuk dosyayı yeni/boş portföy diye sunma.
        raw = json.loads(store.path.read_text(encoding="utf-8"))
        store.unsettled_lots = raw.get("unsettled_lots", [])
        store.loss_history = raw.get("loss_history", [])
        return {"measured": True, "saved_at": raw.get("saved_at"),
                **store.cash_timeline(now_et_naive().date())}

@app.get("/api/var")
def get_var(horizon_days: int = 1, n_sims: int = 10000):
    """A3 — Portföy VaR/CVaR (tarihsel + Monte Carlo). RAPORLAMA katmanı: sinyal-only,
    pozisyon BOYUTUNU kısmaz. Açık pozisyon yoksa ok=False döner (dürüst: tahmin yok).

    var/cvar değerleri POZİTİF $ = beklenen KAYIP büyüklüğü (güven %95/%99)."""
    try:
        from nihai.risk.safety import compute_portfolio_var
        with PORTFOLIO_LOCK:
            _store, portfolio = get_portfolio_status()
            positions = list(portfolio.open_positions)
            equity = float(portfolio.equity)
        if not positions:
            return {"ok": False, "reason": "Açık pozisyon yok — VaR hesaplanamaz.",
                    "equity": equity}
        provider = _analysis_provider()
        result = compute_portfolio_var(
            positions, provider,
            horizon_days=max(1, min(int(horizon_days), 20)),
            n_sims=max(1000, min(int(n_sims), 50000)))
        result["equity"] = equity
        # VaR'ı equity yüzdesi olarak da ver (UI kartı için pratik)
        if result.get("ok") and equity > 0:
            hist = result.get("historical", {}).get("var", {})
            result["var_pct_of_equity"] = {
                k: round(v / equity * 100.0, 2) for k, v in hist.items()}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/signals")
def get_signals():
    try:
        with PORTFOLIO_LOCK:
            store = StateStore()
            store.load_or_init(INITIAL_EQUITY)
            
            # Log dosyalarından son kararları oku ve parse et
            # Son 2 günün dosyalarını listele
            files = sorted(LOG_DIR.glob("decision_log_*.txt"), reverse=True)
            
            # Büyük dosyaları performanslı okumak için tail algoritması
            def tail_file(file_path, max_lines=1000):
                if not file_path.exists():
                    return []
                size = file_path.stat().st_size
                if size < 512 * 1024:  # 512KB altı ise doğrudan oku
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        return f.readlines()
                # 512KB üstü büyük dosyalar için sondan geriye oku
                with open(file_path, "rb") as f:
                    f.seek(0, 2)
                    pointer = f.tell()
                    buffer = bytearray()
                    lines_list = []
                    block_size = 4096
                    while pointer > 0 and len(lines_list) <= max_lines:
                        step = min(block_size, pointer)
                        pointer -= step
                        f.seek(pointer)
                        chunk = f.read(step)
                        buffer = chunk + buffer
                        lines_list = buffer.split(b"\n")
                    return [line.decode("utf-8", errors="ignore") for line in lines_list]

            # Tur-106 (BULGU-106-32): dosyalar EN YENİDEN eskiye, her dosya kendi içinde sondan başa okunur.
            # Eskiden iki dosyanın kuyruğu [bugün…, dün…] diye birleştirilip TERSTEN geziliyordu → ilk 200 satır
            # hep DÜNÜN sonuydu (bugünün kuyruğu 1.500 satırdan kısa okunur: büyük dosyada ~1.000) ve panel seans
            # boyunca dünkü kararları gösteriyordu. Gürültü-yalnız dosyalar (hafta sonu SESSION_CLOSED) atlanabilsin
            # diye en çok 5 dosyaya bakılır.
            parsed_signals = []
            for f_path in files[:5]:
                for line in reversed(tail_file(f_path, 1000)):
                    line = line.strip()
                    if not line or "|" not in line or "ZAMAN" in line or "---" in line or line.startswith("Dosya:"):
                        continue
                    # Seans dışı gürültü filtreleme
                    if "MARKET_CLOSED" in line or "market kapalı" in line:
                        continue
                    parsed_signals.append(line)
                    if len(parsed_signals) >= 200:
                        break
                if len(parsed_signals) >= 200:
                    break
            
            # Öne çıkanlar (kabul edilen sinyal / açılış / kapanış) EN ÜSTE, en yeni önce. Tur-106 (BULGU-106-33):
            # yalnız TARİHLİ satırlar — eski biçim (yalnız saat) günler sonra da güncel görünüyordu.
            from nihai.notifier.terminal_dashboard import tarihli_satirlar
            one_cikan = [s for s in reversed(tarihli_satirlar(store.recent_signals)) if s not in parsed_signals]
            parsed_signals = one_cikan + parsed_signals
                    
            return parsed_signals[:200]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/predictions")
def get_predictions():
    """Canlı fiyat-tahmin servisinin son anlık görüntüsü (önbellekten, hesap yok).

    SİNYAL-ONLY: emir göndermez. P(win) = meta-model güveni (kesin tahmin değil)."""
    try:
        from nihai.execution.live_predict_service import get_service
        return get_service().snapshot()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# Tur-31: TARAYICI YOKLAMASI tam-dosya resolve TETİKLEMEMELİ.
# `pt.resolve()` predictions_log'un TAMAMINI (7,8 MB / 53k satır) csv.DictReader ile
# ayrıştırır ve çözüm varsa geri yazar; maliyeti çözülecek satır sayısıyla değil TOPLAM
# satır sayısıyla ölçeklenir. Pano bunu iki uçtan saniyeler arayla çağırıyordu →
# ÖLÇÜLDÜ: /api/predictions/accuracy 219 ms, tamamına yakını burada (resolve 286 ms/çağrı,
# çözülecek satır 0 iken bile). Kapı: dosya imzası (mtime_ns, size) DEĞİŞMEDİYSE **ve**
# son koşudan bu yana TTL geçmediyse önceki sonucu döndür.
# Zaman bileşeni ŞART: bekleyen bir satır dosya hiç değişmeden, yalnız ZAMAN geçtiği için
# çözülebilir hâle gelir — saf imza-cache'i (summary()'deki gibi) burada yanlış olurdu.
# TTL en hızlı bar (1m) periyodunun altında tutulur → çözüm gecikmesi bir bardan azdır.
# Önbellek YALNIZ bu web yolundadır: `resolve()`'un kendi anlambilimi değişmez, bu yüzden
# adanmış çözücü süreç (forward_harness) ve testler etkilenmez.
_RESOLVE_TTL_SEC = 20.0
_resolve_cache: tuple[tuple[int, int] | None, float, dict | None] | None = None


def _resolve_throttled() -> dict | None:
    """`pt.resolve()` — imza+TTL kapılı. Kapı kapalıysa son sonucu döndürür."""
    global _resolve_cache
    from nihai.execution import prediction_tracker as pt
    sig = pt._file_sig()
    if _resolve_cache is not None:
        onceki_sig, kosuldu, sonuc = _resolve_cache
        if onceki_sig == sig and (time.monotonic() - kosuldu) < _RESOLVE_TTL_SEC:
            return sonuc
    res = pt.resolve(_analysis_provider())
    # İmza resolve'DAN SONRA yeniden okunur: resolve yazdıysa imza değişmiştir ve
    # eski imzayı saklamak her çağrıda cache'i ıskalatırdı.
    _resolve_cache = (pt._file_sig(), time.monotonic(), res)
    return res


@app.get("/api/predictions/accuracy")
def get_prediction_accuracy(resolve: bool = True):
    """Madde 3 — canlı tahminin GERÇEKLEŞEN isabeti (kapalı-döngü, look-ahead'siz).

    PREDICT_TRACK=true ile kaydedilen tahminleri vadeleri gelince gerçekleşenle etiketler
    (resolve) ve özetler (yön isabeti + güven-kovası kalibrasyonu + ort. getiri). DÜRÜST:
    isabeti İDDİA etmez, ÖLÇER — ~%50 ise pano 'öngörü gücü yok' der (EDGE_FINDINGS)."""
    try:
        from nihai.execution import prediction_tracker as pt
        res = None
        if resolve:
            try:
                res = _resolve_throttled()
            except Exception as e:
                res = {"error": str(e)[:80]}
        # D4b (Tur-28): tam-CSV ayrıştırması yerine imza-önbellekli paylaşılan özet.
        # Kopyala — dönen dict PAYLAŞILAN önbellektir, üstüne alan eklemek onu kirletir.
        rep = dict(pt.summary()["report"])
        rep["resolve"] = res
        rep["tracking_enabled"] = os.getenv("PREDICT_TRACK", "").lower() in (
            "1", "true", "yes", "on")
        return rep
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/forward-test")
def get_forward_test(resolve: bool = False):
    """Forward-test (canlı kapalı-döngü doğrulama) OPERASYONEL paneli — projenin gerçek #1
    önceliği. /api/predictions/accuracy isabeti özetler; bu uç KAPSAMA + makine-okunur
    VERDİKT verir: takip açık mı, kaç tahmin birikti, hedefe (target_days) ne kadar kaldı,
    ve dürüst edge verdict'i (NO_DATA/NO_EDGE/WEAK_POSITIVE/INVERTED). İsabet İDDİA etmez,
    ÖLÇER (EDGE_FINDINGS). resolve=true ile vadesi geleni önce etiketler."""
    try:
        from nihai.execution import prediction_tracker as pt
        if resolve:
            try:
                _resolve_throttled()
            except Exception as e:
                logger.warning(f"[web] forward-test resolve atlandı: {e}")
        # D4 (Tur-10): log'u BİR KEZ oku, report+coverage paylaşsın (istek başına 2-3 tam
        # CSV okuması → 1). D4b (Tur-28): o tek okuma da dosya imzasıyla ÖNBELLEKLİ — log
        # değişmediyse ~354 ms'lik ayrıştırma tekrar edilmez, değiştiği an tazelenir.
        ozet = pt.summary()
        rep, cov = ozet["report"], ozet["coverage"]
        target_days = int(os.getenv("FORWARD_TEST_TARGET_DAYS", "14"))
        progress = (round(min(1.0, cov["days_span"] / target_days), 3)
                    if target_days > 0 else None)
        return {
            "tracking_enabled": os.getenv("PREDICT_TRACK", "").lower() in (
                "1", "true", "yes", "on"),
            "coverage": cov,
            "report": rep,
            "verdict": ozet["verdict"],
            "target_days": target_days,
            "progress_to_target": progress,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


_FALLBACK_PROVIDER = None   # tahmin servisi kapalıyken PAYLAŞILAN yedek provider


def _analysis_provider():
    """Tek-sembol analiz/grafik için paylaşılan provider. Tahmin servisi çalışıyorsa
    onun provider'ını (aynı hub/WS bağlantısı) yeniden kullan → ekstra bağlantı YOK.
    Yoksa DATA_PROVIDER'ı cache'li kur — AMA BİR KEZ.

    Tur-30: yedek dal her istekte `get_provider()` çağırıyordu; bu YENİ bir
    `CachingProvider` (boş sözlük) kuruyor, yani "cache'li" provider'ın önbelleği her
    HTTP isteğinde çöpe gidiyordu. Tahmin servisi kapalıyken (şu anki durum) /api/var,
    /api/analyze, /api/bars, /api/intel/*, /api/options/chain hepsi barları ağdan
    YENİDEN çekiyordu. ÖLÇÜLDÜ: aynı süreçte ikinci `compute_portfolio_var` çağrısı
    7 ms, aynı isteği HTTP'den tekrarlamak 2845 ms. CachingProvider zaten thread-safe
    ve paylaşılmak üzere yazılmış (docstring'i öyle diyor); TTL'ler kısa
    (bar 45 sn / quote 1,5 sn) olduğundan bayatlama riski yok."""
    try:
        from nihai.execution.live_predict_service import get_service
        p = get_service()._provider
        if p is not None:
            return p
    except Exception:
        pass
    global _FALLBACK_PROVIDER
    if _FALLBACK_PROVIDER is None:
        # Yarış hâlinde iki thread ikinci bir provider kurabilir — zararsız (ikisi de
        # geçerli, sonuncusu kalır); kilit maliyeti bu kazanca değmez.
        _FALLBACK_PROVIDER = get_provider(DATA_PROVIDER, with_cache=True)
        from nihai.observability import set_provider
        set_provider(_FALLBACK_PROVIDER)   # `/metrics` akış sağlığını buradan okur
    return _FALLBACK_PROVIDER


def _calibration_points(interval: str | None = None):
    """forecast_calibration.json — canlı-tahmin sekmesiyle AYNI kaynak → hisse-detay da
    dürüst 'Olasılık (kal.)' verir.

    Tur-28: burada BİR KEZ yükleyen yerel cache SİLİNDİ. Pano 7/24 çalışır; eğri
    `calibrate_forecast.py` ile tazelenince süreç yeniden başlayana kadar bayat kalıyordu.
    `_load_calibration` artık dosya imzasıyla kendi cache'ini tutuyor (değişmediyse tek
    `stat()`), yani istek başına çağırmak hem TAZE hem ucuz.
    Tur-83: `interval` eğrinin KENDİ aralığıyla eşleşmezse None — 1m/5 bar ufkunda kurulan
    bir olasılık, 1h detay analizinin olasılığı DEĞİLDİR (eskiden ona da uygulanıyordu)."""
    try:
        from nihai.execution.live_predict_service import _load_calibration
        return _load_calibration(interval)
    except Exception:
        return None


def _bars_for_analysis(provider, sym: str, interval: str, min_bars: int = 60):
    """Analiz için bar çek; intraday'de yetersiz bar gelirse pencereyi genişletip
    yeniden dene.

    Madde 1 düzeltmesi: ücretsiz IEX feed'i kısa dilimlerde (özellikle 5dk) dar
    `lookback` penceresinde 60 barın altına düşebiliyordu → `_start_for_lookback`
    penceresi lookback'e bağlı olduğundan, lookback'i kademeli büyütünce takvim
    penceresi de genişler ve seyrek IEX barları yine de 60'ı aşar. En iyi (en uzun)
    sonucu döndür."""
    df = provider.get_bars(sym, interval, lookback_bars=220)
    if (df is None or len(df) < min_bars) and interval in ("1m", "5m", "15m"):
        for lb in (500, 800):
            try:
                wider = provider.get_bars(sym, interval, lookback_bars=lb)
            except Exception:
                wider = None
            if wider is not None and len(wider) >= min_bars:
                return wider
            if wider is not None and (df is None or len(wider) > len(df)):
                df = wider
    return df


def _order_plan(price: float, atr_val: float, action: str,
                stop: float | None, target: float | None, hedef_nominal: bool = False) -> dict:
    """Madde 6 — SİNYAL-ONLY emir öneri seviyeleri (limit/market/stop-limit/bracket/
    çok-bacaklı opsiyon). Broker'a HİÇBİR emir göndermez; yalnızca operatörün manuel
    girebileceği seviyeleri ÖNERİR. Sinyal-only tasarım korunur (bkz. README)."""
    import math as _math
    if action not in ("LONG", "SHORT") or price <= 0:
        return {"note": "Aksiyon WAIT/yok — emir önerisi üretilmedi.", "legs": []}
    is_long = action == "LONG"
    atrv = 0.0 if (atr_val is None or _math.isnan(atr_val)) else float(atr_val)
    side = "BUY" if is_long else "SELL"
    exit_side = "SELL" if is_long else "BUY"
    # Limit giriş: küçük geri-çekilmeyi bekle (long→altta, short→üstte) = daha iyi fiyat
    limit_entry = round(price - 0.25 * atrv, 2) if is_long else round(price + 0.25 * atrv, 2)
    # Stop-limit: tetik = strateji stopu; limit, slipaj tamponuyla tetiğin ötesinde
    stop_trigger = round(float(stop), 2) if stop else None
    stop_limit = None
    if stop_trigger is not None:
        off = 0.1 * atrv
        stop_limit = round(stop_trigger - off, 2) if is_long else round(stop_trigger + off, 2)
    # Tur-106 (BULGU-106-28): hedef nominalse (quant / trailing kipi) pozisyon orada KAPANMAZ → kâr-al bacağı ve
    # hedefe dayalı uzak strike ÖNERİLMEZ; çıkış izleyen stoptur.
    tgt = round(float(target), 2) if target and not hedef_nominal else None
    # Çok-bacaklı opsiyon: yön dikey spread önerisi (debit). Strike'lar YAKLAŞIK (zincir
    # teyidi gerekir); broker emri değil, yapı önerisi.
    k_near = round(price)
    k_far = round(tgt) if tgt else round(price + (atrv if is_long else -atrv))
    opt_type = "CALL" if is_long else "PUT"
    multi_leg = {
        "structure": ("bull_call_spread" if is_long else "bear_put_spread"),
        "note": "Dikey debit spread önerisi — strike'lar yaklaşık, zincir teyidi gerekir.",
        "legs": [
            {"action": "BUY", "option_type": opt_type, "strike_approx": k_near},
            {"action": "SELL", "option_type": opt_type, "strike_approx": k_far},
        ],
    }
    return {
        "side": side,
        "hedef_nominal": bool(hedef_nominal),
        "disclaimer": "SİNYAL-ONLY: seviyeler öneridir, otomatik emir gönderilmez.",
        "market": {"type": "MARKET", "side": side, "ref_price": round(price, 2)},
        "limit": {"type": "LIMIT", "side": side, "price": limit_entry},
        "stop_market": ({"type": "STOP", "side": exit_side, "trigger": stop_trigger}
                        if stop_trigger is not None else None),
        "stop_limit": ({"type": "STOP_LIMIT", "side": exit_side,
                        "trigger": stop_trigger, "limit": stop_limit}
                       if stop_trigger is not None else None),
        "bracket": {  # OCO: giriş + kâr-al + zarar-kes (tek yapı)
            "entry": {"type": "LIMIT", "side": side, "price": limit_entry},
            "take_profit": ({"type": "LIMIT", "side": exit_side, "price": tgt}
                            if tgt is not None else None),
            "stop_loss": ({"type": "STOP", "side": exit_side, "trigger": stop_trigger}
                          if stop_trigger is not None else None),
        },
        "multi_leg_option": multi_leg,
    }


def _bars_payload(df) -> list[dict]:
    """OHLCV DataFrame → lightweight-charts uyumlu liste (time = UNIX saniye, UTC).

    D9 (Tur-10): iterrows (satır başına Series kurulumu) → toplu epoch + sütun listeleri.
    Değer/yuvarlama birebir (Python round korunur; parite: test_perf_parity_round10)."""
    import pandas as pd
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    # as_unit("ns"): asi8 birim-BAĞIMLI (pandas us/ms çözünürlüklü index üretebilir) →
    # ns'e sabitle; //1e9 pozitif zamanlarda int(ts.timestamp()) ile özdeş.
    epochs = idx.as_unit("ns").asi8 // 1_000_000_000
    o, h = df["open"].tolist(), df["high"].tolist()
    lo, c = df["low"].tolist(), df["close"].tolist()
    vol = df["volume"].tolist() if "volume" in df.columns else None
    out = []
    for i in range(len(df)):
        v = float(vol[i]) if vol is not None and vol[i] is not None else 0.0
        out.append({
            "time": int(epochs[i]),
            "open": round(float(o[i]), 4),
            "high": round(float(h[i]), 4),
            "low": round(float(lo[i]), 4),
            "close": round(float(c[i]), 4),
            "volume": v if v else 0.0,
        })
    return out


def _series_payload(s) -> list[dict]:
    """pandas Series → lightweight-charts line serisi ([{time, value}], NaN atlanır).

    D9 (Tur-10): items() yerine toplu epoch + liste; NaN/None atlama kuralı birebir."""
    import pandas as pd
    idx = pd.DatetimeIndex(s.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    epochs = idx.as_unit("ns").asi8 // 1_000_000_000  # asi8 birim-bağımlı → ns'e sabitle
    out = []
    for i, val in enumerate(s.tolist()):
        if val is None or (isinstance(val, float) and val != val):  # NaN
            continue
        out.append({"time": int(epochs[i]), "value": round(float(val), 4)})
    return out


def _hist_payload(s, up: str = "#22c55e", dn: str = "#ef4444") -> list[dict]:
    """pandas Series → lightweight-charts histogram serisi ([{time, value, color}]).

    Pozitif barlar yeşil, negatif barlar kırmızı (MACD histogramı momentum yönünü
    renkle gösterir). NaN değerler atlanır.

    D9 (Tur-10): items() yerine toplu epoch + liste; atlama/renk kuralı birebir."""
    import pandas as pd
    idx = pd.DatetimeIndex(s.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    epochs = idx.as_unit("ns").asi8 // 1_000_000_000  # asi8 birim-bağımlı → ns'e sabitle
    out = []
    for i, val in enumerate(s.tolist()):
        if val is None or (isinstance(val, float) and val != val):  # NaN
            continue
        fv = float(val)
        out.append({"time": int(epochs[i]), "value": round(fv, 4), "color": up if fv >= 0 else dn})
    return out


_BAR_DK = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}


def _gelecek_barlar(son_ts, interval: str, n: int) -> list[int]:
    """Son bardan SONRAKİ `n` seans barının etiket zamanı (UTC epoch sn) — XNYS takvimi.

    Tur-92: tahmin/ileri-bulut noktaları gece, hafta sonu ve tatile düşmesin diye. Gün-içi
    barlar seans açılışına çapalıdır (Tur-91 ızgarası: 1h → 09:30, 10:30 … ET); günlük barın
    etiketi son barın ET saatinden türetilir (sağlayıcı ET gece yarısı etiketler). Bilinmeyen
    aralık → []."""
    import exchange_calendars as xcals
    import pandas as pd
    if n <= 0 or (interval not in _BAR_DK and interval != "1d"):
        return []
    cal = xcals.get_calendar("XNYS")
    ts = pd.Timestamp(son_ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    et = ts.tz_convert("America/New_York")
    gun = et.tz_localize(None).floor("D")   # ET takvim günü (gece yarısı formülü DEĞİL, Tur-84)
    son_seans = cal.last_session                    # takvim ~1 yıl ileri tanımlı: aşmak istisnadır
    if gun > son_seans:
        return []
    seanslar = cal.sessions_in_range(gun, min(gun + pd.Timedelta(days=2 * n + 10), son_seans))
    out: list[int] = []
    if interval == "1d":
        saat = et.tz_localize(None) - gun          # etiketin ET gün-içi konumu
        for s in seanslar:
            t = (pd.Timestamp(s).tz_localize(None) + saat).tz_localize("America/New_York")
            if t > ts:
                out.append(int(t.timestamp()))
            if len(out) >= n:
                break
        return out
    adim = pd.Timedelta(minutes=_BAR_DK[interval])
    for s in seanslar:
        t, kapanis = cal.session_open(s), cal.session_close(s)
        while t < kapanis and len(out) < n:
            if t > ts:
                out.append(int(t.timestamp()))
            t += adim
        if len(out) >= n:
            break
    return out


def _bar_bitisi(etiket: int, interval: str) -> int:
    """Etiketi `etiket` (UTC epoch sn) olan seans barının KAPANIŞ anı: etiket + bar süresi, seans
    kapanışında kırpılır (1h'in son yarım barı 16:00 ET'de, yarım günde 13:00'te biter); 1d →
    o seansın kapanışı."""
    import exchange_calendars as xcals
    import pandas as pd
    cal = xcals.get_calendar("XNYS")
    t = pd.Timestamp(etiket, unit="s", tz="UTC")
    gun = t.tz_convert("America/New_York").tz_localize(None).floor("D")
    kapanis = cal.session_close(gun) if cal.is_session(gun) else None
    if interval == "1d":
        return int(kapanis.timestamp()) if kapanis is not None else etiket
    son = t + pd.Timedelta(minutes=_BAR_DK.get(interval, 0))
    if kapanis is not None and son > kapanis:
        son = kapanis
    return int(son.timestamp())


def _tahmin_bandi(close, atr_pct, ufuk: int, *, min_n: int = 30) -> dict | None:
    """`ufuk` bar sonrası kapanışın ampirik ~%80 aralığı + medyanı ve AYNI kuralın yürüyen sınavı.

    Tur-92 (BULGU-92-1): Hisse Detay'ın tek tahmini defterden kalibre 1dk aralığıydı ve seçicide
    1dk yoktu → hiçbir aralıkta tahmin fiyatı basılmıyordu. Kural tahmin sekmesininkiyle aynı
    aileden: z = (C[t+h]/C[t] − 1)·100 / ATR%[t] (Tur-84 ölçeği), kuantiller o sembolün KENDİ
    barlarından, çapa son bar kapanışı (BULGU-84-4). Yön iddiası yoktur: medyan, yürüyen sınavda
    "fiyat değişmez"den (sıfır getiri) iyi değilse merkez son kapanıştır (BULGU-84-2 dersi).

    Yürüyen sınav: her t'de yalnız t'ye kadar ÇÖZÜLMÜŞ z'ler (s+h ≤ t) kullanılır → kapsama ve
    ortalama mutlak hata (bps) iddia değil ölçümdür. Veri < min_n → None ("—")."""
    import numpy as np
    c = np.asarray(close, dtype=float)
    a = np.asarray(atr_pct, dtype=float)
    h = int(ufuk)
    n = c.size
    if h < 1 or n <= h or a.size != n:
        return None
    ok = (np.isfinite(a[:-h]) & (a[:-h] > 0) & np.isfinite(c[:-h]) & (c[:-h] > 0) & np.isfinite(c[h:]))
    z = np.full(n - h, np.nan)
    z[ok] = (c[h:][ok] / c[:-h][ok] - 1.0) * 100.0 / a[:-h][ok]
    zs = z[np.isfinite(z)]
    if zs.size < min_n or not (np.isfinite(a[-1]) and a[-1] > 0 and c[-1] > 0):
        return None
    icinde, hata_med, hata_sabit = [], [], []
    for t in range(n - h):
        if not np.isfinite(z[t]):
            continue
        gecmis = z[: t - h + 1] if t - h + 1 > 0 else z[:0]
        gecmis = gecmis[np.isfinite(gecmis)]
        if gecmis.size < min_n:
            continue
        lo_t, med_t, hi_t = np.quantile(gecmis, [0.10, 0.50, 0.90])
        icinde.append(lo_t <= z[t] <= hi_t)
        hata_med.append(abs(z[t] - med_t) * a[t] * 100.0)     # % → bps
        hata_sabit.append(abs(z[t]) * a[t] * 100.0)
    q10, q50, q90 = np.quantile(zs, [0.10, 0.50, 0.90])
    son, olcek = float(c[-1]), float(a[-1]) / 100.0
    n_eval = len(icinde)
    mae_med = float(np.mean(hata_med)) if n_eval else None
    mae_sabit = float(np.mean(hata_sabit)) if n_eval else None
    medyan_iyi = bool(n_eval >= min_n and mae_med is not None and mae_sabit is not None
                      and mae_med < mae_sabit)
    return {
        "ufuk": h,
        "merkez": round(son * (1.0 + q50 * olcek) if medyan_iyi else son, 2),
        "merkez_kaynak": "medyan" if medyan_iyi else "son_kapanis",
        "medyan": round(son * (1.0 + q50 * olcek), 2),
        "alt": round(son * (1.0 + q10 * olcek), 2),
        "ust": round(son * (1.0 + q90 * olcek), 2),
        "capa": round(son, 2),
        "n": int(zs.size),
        "sinav": {"n": n_eval,
                  "kapsama": (round(100.0 * float(np.mean(icinde)), 1) if n_eval else None),
                  "mae_medyan_bps": (round(mae_med, 1) if mae_med is not None else None),
                  "mae_sabit_bps": (round(mae_sabit, 1) if mae_sabit is not None else None)},
    }


@app.get("/api/bars")
def get_bars_endpoint(symbol: str, interval: str = "1h", lookback: int = 200):
    """Grafik için ham OHLCV barları (hub/alpaca üzerinden). Canlı mum grafiği bunu çeker.

    Eşzamanlılık notu: bu uç `def` (async DEĞİL) — Starlette bunu kendi worker
    thread-havuzunda (anyio, varsayılan 40 işçi) çalıştırır. Böylece ağ-bağımlı bar
    çekimi tek event-loop thread'ini KİLİTLEMEZ; aynı anda gelen istekler örtüşür."""
    try:
        sym = symbol.upper().strip()
        lookback = max(30, min(int(lookback), 1000))
        provider = _analysis_provider()
        df = provider.get_bars(sym, interval, lookback)
        if df is None or df.empty:
            return {"symbol": sym, "interval": interval, "bars": []}
        return {"symbol": sym, "interval": interval, "bars": _bars_payload(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/quote")
def get_quote_endpoint(symbol: str):
    """Anlık quote (canlı fiyat çizgisi/başlık güncellemesi için hafif uç).

    `def` (async DEĞİL): ağ çağrısı thread-havuzunda koşar, event-loop'u bloklamaz."""
    try:
        provider = _analysis_provider()
        q = provider.get_quote(symbol.upper().strip())
        if q is None:
            return {}
        ts = q.timestamp
        return {"symbol": q.symbol, "bid": q.bid, "ask": q.ask, "last": q.last,
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/llm-events")
def get_llm_events(symbol: str):
    """LLM (8-K olay-çıkarımı) ADAY sinyalleri — SALT cache okur, AI ÇAĞIRMAZ (ucuz).
    KENAR KANITLI DEĞİL; UI 'araştırma/aday' etiketiyle göstermeli (research/llm_event_test.py)."""
    try:
        from nihai.data.llm_event import cached_events
        sym = symbol.upper().strip()
        ev = cached_events(sym)             # SALT cache — AI/EDGAR çağırmaz (sayfa açılışı güvenli)
        if ev is None or ev.empty:
            return {"symbol": sym, "events": []}
        out = []
        for d, r in ev.iterrows():
            score = round(float(r.get("score", 0.0)), 2)
            sig = "LONG" if score >= 0.4 else ("SHORT" if score <= -0.4 else "WAIT")
            out.append({
                "date": str(d),
                "event_type": str(r.get("event_type", "none")),
                "direction": str(r.get("direction", "neutral")),
                "magnitude": round(float(r.get("magnitude", 0.0)), 2),
                "confidence": round(float(r.get("confidence", 0.0)), 2),
                "score": score, "signal": sig,
                "rationale": str(r.get("rationale", "")),
            })
        out.reverse()                       # en yeni üstte
        return {"symbol": sym, "events": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/llm-events/refresh")
def refresh_llm_events(symbol: str, max_filings: int = 10):
    """Yeni 8-K'ları LLM ile çıkar — AI BURADA çalışır (kullanıcı tetikler, rate-limitli).
    Ücretsiz Gemini kotası: arka arkaya basma."""
    try:
        from nihai.data.llm_event import llm_events
        sym = symbol.upper().strip()
        ev = llm_events(sym, refresh=True, max_filings=max(1, min(int(max_filings), 40)))
        return {"symbol": sym, "refreshed": 0 if ev is None or ev.empty else int(len(ev))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/analyze")
def analyze_symbol(symbol: str, interval: str = "1h"):
    """TEK sembol derin analizi: mevcut fiyat, yön, öngörü/hedef ('olması gereken')
    fiyat, stop, RR, kalite, P(win), rejim + tüm gösterge konsensüsü.

    SİNYAL-ONLY: emir göndermez. 'Öngörü aralığı' ufuk sonundaki fiyatın kalibre ~%80 aralığı
    (yalnız kalibre edildiği aralıkta; yön iddiası yok — Tur-84),
    'hedef' stratejinin RR hedefi, P(win) meta-model güvenidir — hiçbiri garanti değil.

    Eşzamanlılık/çekirdek notu: bu uç bilinçli olarak `def` (async DEĞİL). Detay
    sekmesinin maliyeti CPU değil AĞ I/O'dur (sağlayıcıdan bar çekme); göstergeler
    ~220 bar üzerinde milisaniyeler sürer. `def` olduğundan Starlette analizi worker
    thread-havuzunda koşturur → ağ beklerken event-loop boşa çıkar, 5sn'de bir gelen
    status/positions/logs istekleri donmaz ve birden çok sembol analizi I/O'da örtüşür.
    Tek sembolü çok çekirdeğe bölmek HIZ KAZANDIRMAZ (paralelleştirilecek CPU işi yok,
    süreç açma maliyeti net kayıp olur); asıl paralellik kazancı ÇOK sembol tarayan
    screener/backtest tarafındadır."""
    try:
        import numpy as np

        from nihai.execution.live_predict_service import (
            _apply_band,
            _apply_calibration,
            _bias_and_confidence,
            _et_gunu,
            _htf_direction,
            _last_valid,
            _load_band,
            _regime_tilt,
            _seans_acilisi,
            _yon_kipi,
        )
        from nihai.signals.indicators import ichimoku, indicator_panel
        from nihai.signals.market_regime import compute_regime
        from nihai.signals.meta_gate import get_gate
        from nihai.signals.strategy import (
            SignalAction,
            _SharedIndicators,
            generate_signal,
        )

        sym = symbol.upper().strip()
        if not sym:
            raise HTTPException(status_code=400, detail="Sembol boş olamaz.")

        provider = _analysis_provider()
        df = _bars_for_analysis(provider, sym, interval, min_bars=60)
        if df is None or df.empty or len(df) < 60:
            got = 0 if df is None else len(df)
            hint = (" Kısa dilimlerde (1dk/5dk) ücretsiz IEX feed'i seyrek bar üretir; "
                    "15dk veya 1s deneyin." if interval in ("1m", "5m") else "")
            return {"symbol": sym, "ok": False,
                    "error": f"{sym} {interval} için yeterli veri yok "
                             f"({got}/60 bar).{hint}"}

        close = df["close"]
        last_close = float(close.iloc[-1])

        # Canlı fiyat (quote) — yoksa son kapanış
        bid = ask = 0.0
        live_price = 0.0
        try:
            q = provider.get_quote(sym)
            if q:
                live_price = float(getattr(q, "last", 0) or 0)
                bid = float(getattr(q, "bid", 0) or 0)
                ask = float(getattr(q, "ask", 0) or 0)
        except Exception:
            pass
        price = live_price if live_price > 0 else last_close

        # Gün-içi değişim — SEANS açılışından (Tur-84, BULGU-84-3; tahmin sekmesiyle aynı kaynak).
        # Eskiden barların UTC gece yarısından sonraki ilki okunuyordu: 1m/5m/1h'de seans öncesi bir
        # IEX işlemi ya da pencerenin başı. Açılış bilinmiyorsa None ("—"), uydurma yok.
        try:
            day_open = _seans_acilisi(provider, sym, _et_gunu(df.index[-1]))
        except Exception:
            day_open = None
        chg_pct = round((price - day_open) / day_open * 100.0, 2) if day_open else None

        # Göstergeler — Tur-7 paylaşımlı önbelleği sembol başına BİR kez kurulur ve
        # AŞAĞIDAKİ generate_signal + meta kapısı + açıklama kartı da onu kullanır.
        # Tur-43 öncesi bu beş gösterge `/api/analyze` içinde DÖRT KEZ hesaplanıyordu
        # (burada, generate_signal'in attığı iç önbellekte, kapının extract_features'ında
        # ve explain'in İKİNCİ extract_features'ında). Değerler birebir aynı.
        shared = _SharedIndicators(df)
        e20 = _last_valid(shared.ema(20))
        e50 = _last_valid(shared.ema(50))
        r = _last_valid(shared.rsi14())
        a = _last_valid(shared.atr14())
        adx_v = _last_valid(shared.adx_series())

        # Üst zaman dilimi + rejim
        htf_dir = 0
        try:
            htf_dir = _htf_direction(provider.get_bars(sym, "15m", lookback_bars=80))
        except Exception:
            htf_dir = 0
        regime = None
        try:
            regime = compute_regime(provider.get_bars("SPY", "1d", lookback_bars=220)).regime
        except Exception:
            regime = None

        # Tur-83: yönün fiyat girdisi canlı tahmin sekmesiyle AYNI kural (`_yon_kipi`):
        # varsayılan KARAR BARININ KAPANIŞI. Burada kotasyon sağlık denetimi bile yoktu —
        # EMA20'nin öbür yanına geçen bir kotasyon yönü çeviriyordu ve canlı defterde o
        # çevrilen yönler %26 isabet etti (BULGU-83-1). `price` gösterim/ATR hedefi için kalır.
        yon_px = last_close if _yon_kipi() == "bar" else price
        direction, _bias, conf_base = _bias_and_confidence(
            yon_px, e20, e50, r, adx_v, htf_dir=htf_dir, regime_tilt=1.0)
        confidence = float(np.clip(conf_base * _regime_tilt(regime, direction), 0.0, 1.0))
        # DÜRÜST metrik: 'Yön Güveni' ham skordur; 'Olasılık (kal.)' canlı defterde
        # GERÇEKLEŞEN isabete kalibre edilmiş hâlidir ve girdisi YAYINLANAN güvendir (Tur-83).
        # Eğri yalnız kendi aralığında uygulanır; başka aralıkta None ("—") döner.
        prob_cal = _apply_calibration(_calibration_points(interval), confidence)

        # Tur-84 (BULGU-84-2): `price ± ATR` yön-işaretli nokta tahmini SİLİNDİ (1m defterde "fiyat
        # değişmez"den ölçülerek kötüydü). Yerine tahmin sekmesiyle AYNI kalibre aralık; kuantiller
        # yalnız KENDİ aralığında (1m / defter ufku) geçerlidir — başka aralıkta "—".
        # Çapa son BAR KAPANIŞI (tahmin sekmesiyle aynı; kotasyon çapası ölçülerek kötüydü, BULGU-84-4).
        atr_pct = (a / last_close * 100.0) if (last_close > 0 and np.isfinite(a) and a > 0) else None
        # Kalibrasyon alanı AÇIK seanstır (tahmin sekmesiyle aynı kural) → kapalı piyasada "—".
        bant = _apply_band(_load_band(interval), sym, atr_pct) if is_market_open() else None
        band_lo = round(last_close * (1.0 + bant[0] / 100.0), 2) if bant else None
        band_hi = round(last_close * (1.0 + bant[1] / 100.0), 2) if bant else None
        exp_move = round(bant[2], 2) if bant else None

        # Tur-92 (BULGU-92-1): HER aralıkta tahmin fiyatı — ufuk tahmin defteriyle aynı (5 bar),
        # ölçek ATR14% (yukarıdaki `a` ile aynı seri), çapa son bar kapanışı. Gelecek bar zamanları
        # takvimden: tahmin noktası ve ileri Ichimoku bulutu grafikte doğru saate düşsün.
        # Kuantiller UZUN pencereden (1000 bar): 8 sembolde yürüyen kapsama 220 barda %66,7–83,3
        # saçılırken 1000 barda %74,6–81,2 (ölçüldü). Grafik/sinyal penceresi DEĞİŞMEZ; uzun seri
        # aynı son bara sahip değilse (önbellek anı farkı) kısa pencereye düşülür — çapa tek.
        from nihai.execution.prediction_tracker import HORIZON_BARS
        try:
            df_t = provider.get_bars(sym, interval, lookback_bars=1000)
        except Exception:
            df_t = None
        if df_t is None or len(df_t) < len(df) or df_t.index[-1] != df.index[-1]:
            df_t, sh_t = df, shared
        else:
            sh_t = _SharedIndicators(df_t)
        c_t = df_t["close"].to_numpy(dtype=float)
        tahmin = _tahmin_bandi(c_t, sh_t.atr14().to_numpy(dtype=float) / c_t * 100.0, HORIZON_BARS)
        try:
            gelecek = _gelecek_barlar(df.index[-1], interval, max(26, HORIZON_BARS))
        except Exception as e:
            logger.debug(f"[web] {sym} gelecek bar zamanları kurulamadı: {e}")
            gelecek = []
        if tahmin is not None:
            # hedef_ts = ufuk barının ETİKETİ (grafik ekseni); tahmin o barın KAPANIŞI içindir
            # (hedef_bitis_ts: etiket + bar süresi, seans kapanışında kırpılmış; 1d → seans kapanışı).
            tahmin["hedef_ts"] = gelecek[HORIZON_BARS - 1] if len(gelecek) >= HORIZON_BARS else None
            tahmin["hedef_bitis_ts"] = (_bar_bitisi(tahmin["hedef_ts"], interval)
                                        if tahmin["hedef_ts"] is not None else None)

        # Strateji sinyali (aksiyon/stop/hedef/RR/kalite + meta)
        # Tur-106 (BULGU-106-27): canlı taramayla AYNI strateji kaynağı ve AYNI günlük barlar. quant_trend günlük
        # barla karar verir (yalnız kapanmış günler, BULGU-106-17); bu uç onları geçmediği için kart gün-içi aralıkta
        # hep QUANT_GUNLUK_BAR_GEREKLI, 1d'de ise oluşmakta olan barın kararını gösteriyordu.
        from nihai.runtime_strategy import apply as _strateji_sec
        from nihai.signals.quant import CANLI_GUNLUK_BAR
        strateji = _strateji_sec()
        df_gunluk = None
        if strateji == "quant_trend":
            try:
                df_gunluk = kapanmis_gunluk(provider.get_bars(sym, "1d", lookback_bars=CANLI_GUNLUK_BAR))
            except Exception as e:
                logger.debug(f"[web] {sym} günlük barlar okunamadı: {e}")
        sig = generate_signal(sym, df, shared=shared, strategy_mode=strateji, df_daily=df_gunluk)
        pwin = None
        meta_explain = None
        # Meta kapı quant sinyalinde kapsam dışıdır (BULGU-97-4: TA ile eğitilen model ona p_win=0,00 basar).
        if (sig.action in (SignalAction.LONG, SignalAction.SHORT)
                and not is_quant_label((sig.metadata or {}).get("strategy"))):
            try:
                gate = get_gate()
                pwin = gate.probability(df, sig, shared=shared)
                # D1: SHAP-tarzı P(win) açıklaması (Tur-43: kendi içinde artık ikinci
                # bir extract_features koşmuyor; olasılık aynı vektörden türetiliyor).
                meta_explain = gate.explain(df, sig, shared=shared)
            except Exception:
                pwin = None

        # 'Olması gereken fiyat' referansları (çoklu — tek bir sihirli sayı değil)
        ma, _, _ = shared.bb20()   # detay panelinde yalnız orta-bant gösterilir
        panel = indicator_panel(df, live_price=price)
        ich = ichimoku(df)
        overlays = {k: _series_payload(v) for k, v in ich.items()}
        # İleri bulut (Tur-92): Senkou A/B tanım gereği 26 bar İLERİ çizilir; `ichimoku()` "şimdi"
        # değerini verdiği için grafikte bulut son barda bitiyordu. Son 26 kaydırılmamış değer
        # gelecek seans barlarına yerleşir — yalnız geçmiş veriden hesaplanır (look-ahead yok).
        ileri = gelecek[:26]
        ham_a = ((ich["tenkan"] + ich["kijun"]) / 2.0).to_numpy(dtype=float)[-26:]
        ham_b = ((df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2.0).to_numpy(dtype=float)[-26:]
        overlays["span_a_ileri"] = [{"time": t, "value": round(float(v), 4)}
                                    for t, v in zip(ileri, ham_a, strict=False) if np.isfinite(v)]
        overlays["span_b_ileri"] = [{"time": t, "value": round(float(v), 4)}
                                    for t, v in zip(ileri, ham_b, strict=False) if np.isfinite(v)]

        # Quant Beyin görüşü (Tur-92) — canlı tarama yolundaki `ai_advisor` ile AYNI motor ve AYNI
        # girdiler (özellik vektörü, meta p_win, kapanışlar); taraf sinyal WAIT ise yön
        # göstergesinden. `persist=False`: pano sorgusu karar defterine/anıya yazmaz.
        quant: dict | None
        try:
            from nihai.ai import decision as _decision
            from nihai.execution import ai_advisor as _aia
            from nihai.execution.ai_advisor import resolve_mode as _ai_kip
            from nihai.signals.meta_features import extract_features
            sinyal_tarafi = sig.action in (SignalAction.LONG, SignalAction.SHORT)
            taraf = sig.action.value if sinyal_tarafi else ("LONG" if direction == "+" else "SHORT")
            raw_pwin = (sig.metadata or {}).get("meta_p_win")
            if _aia.kapsam_disi(sig, strateji):   # Tur-99: canlı köprüyle AYNI kapsam kuralı (ve AYNI strateji kaynağı)
                quant = {"kip": _ai_kip(), "taraf": taraf, "taraf_kaynagi": "sinyal" if sinyal_tarafi else "yon",
                         "action": _aia.KAPSAM_DISI, "posterior": None, "karar": _aia.KAPSAM_NEDENI,
                         "kanit": [_aia.KAPSAM_NEDENI]}
            else:
                qd = _decision.decide(
                    sym, strateji, extract_features(df, sig, shared=shared),
                    p_win=float(raw_pwin) if raw_pwin is not None else None,
                    closes=close.to_numpy(dtype=float), side_hint=taraf,
                    record_episode=False, persist=False)
                quant = {"kip": _ai_kip(), "taraf": taraf,
                         "taraf_kaynagi": "sinyal" if sinyal_tarafi else "yon",
                         "action": qd.action, "posterior": round(qd.posterior, 3),
                         "prior": round(qd.prior_mean, 3), "conviction": round(qd.conviction, 3),
                         "utility": (round(qd.utility, 5) if qd.utility is not None else None),
                         "karar": qd.evidence[-1] if qd.evidence else "",
                         "kanit": qd.evidence}
        except Exception as e:
            logger.debug(f"[web] {sym} Quant görüşü üretilemedi: {e}")
            quant = {"hata": f"{type(e).__name__}: {e}"}

        # TD Sequential rakam-overlay (her anlamlı barın üstüne/altına setup/countdown no)
        import pandas as _pd

        from nihai.signals.indicators_ext import td_sequential_series
        td_markers = []
        for m in td_sequential_series(df):
            idx = df.index[m["pos"]]
            ts = _pd.Timestamp(idx)
            epoch = int(ts.tz_localize("UTC").timestamp() if ts.tzinfo is None else ts.timestamp())
            buy = m["dir"] == "buy"
            if m["kind"] == "cd":
                color = "#22d3ee" if buy else "#fb923c"   # countdown: cyan/turuncu
            else:
                color = "#22c55e" if buy else "#ef4444"   # setup: yeşil/kırmızı
            td_markers.append({
                "time": epoch,
                "position": "belowBar" if buy else "aboveBar",
                "color": color,
                "shape": "circle",
                "text": str(m["num"]),
            })
        fair_value = {
            "strategy_target": round(float(sig.target), 2) if sig.target else None,
            "ema50": round(e50, 2) if not np.isnan(e50) else None,
            # `ma.dropna().empty` yeni bir Series+Index tahsis eder; `count()` NA-olmayanı
            # C düzeyinde sayar (Tur-43, BULGU-43-2 deseninin üçüncü kopyası).
            "bb_mid": round(float(ma.iloc[-1]), 2) if ma.count() else None,
            "pivot": panel["pivots"].get("pivot") if panel.get("pivots") else None,
            "vwap": next((row["value"] for row in panel["rows"] if row["name"].startswith("VWAP")), None),
        }

        # Alt-grafikli göstergeler: fiyat ekseniyle aynı ölçeğe sığmayan osilatörler
        # (MACD, RSI) kendi alt panelinde çizilir. Detay sekmesi bunları zaman ekseni
        # senkron alt grafikler olarak gösterir. Yeni grafikli gösterge eklemek için
        # buraya bir seri payload'ı ekleyip HTML'de drawSubcharts'a bir blok eklemek yeter.
        macd_line, macd_signal, macd_hist = shared.macd()
        subcharts = {
            "macd": {
                "title": "MACD (12, 26, 9)",
                "macd": _series_payload(macd_line),
                "signal": _series_payload(macd_signal),
                "hist": _hist_payload(macd_hist),
            },
            "rsi": {
                "title": "RSI (14)",
                "rsi": _series_payload(shared.rsi14()),   # aynı seri, tekrar-hesap yok
                "levels": [30.0, 50.0, 70.0],
            },
        }

        return {
            "symbol": sym,
            "ok": True,
            "interval": interval,
            "as_of": df.index[-1].isoformat() if hasattr(df.index[-1], "isoformat") else str(df.index[-1]),
            "source": getattr(provider, "name", None) or DATA_PROVIDER,
            "bars_count": len(df),
            "price": round(price, 2),
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "last_close": round(last_close, 2),
            "chg_pct": chg_pct,
            "direction": direction,
            "confidence": round(confidence, 2),
            "prob_cal": (round(prob_cal, 3) if prob_cal is not None else None),
            "exp_move_pct": exp_move,
            "band_lo": band_lo,
            "band_hi": band_hi,
            "tahmin": tahmin,
            "gelecek": gelecek,
            "quant": quant,
            "regime": regime,
            "htf_dir": htf_dir,
            "signal": {
                "action": sig.action.value,
                "stop": round(float(sig.stop), 2) if sig.stop else None,
                "target": round(float(sig.target), 2) if sig.target else None,
                "hedef_nominal": not sabit_hedefle_cikar((sig.metadata or {}).get("strategy")),
                "rr": round(float(sig.rr_ratio or 0), 2),
                "quality": round(float(sig.quality_score or 0), 2),
                "pwin": (round(pwin, 3) if pwin is not None else None),
                "reasons": sig.reasons[:6],
                "strategy": sig.metadata.get("strategy") if sig.metadata else None,
                "meta_explain": meta_explain,
            },
            "order_plan": _order_plan(price, a, sig.action.value, sig.stop, sig.target,
                                      not sabit_hedefle_cikar((sig.metadata or {}).get("strategy"))),
            "fair_value": fair_value,
            "indicators": panel,
            "overlays": overlays,
            "subcharts": subcharts,
            "td_markers": td_markers,
            "bars": _bars_payload(df),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[web] analiz hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/watchlist")
def get_watchlist_data():
    try:
        import nihai.execution.watchlist as watchlist
        with PORTFOLIO_LOCK:
            return {
                "current": watchlist.current(),
                "pins": watchlist.pins(),
                "blocks": watchlist.blocks(),
                "last_scan": watchlist.last_scan()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/greeks")
def get_greeks():
    try:
        with PORTFOLIO_LOCK:
            _store, portfolio = get_portfolio_status()
            agg = PortfolioGreeks()
            for p in portfolio.open_positions:
                if p.greeks:
                    agg.add_position(p.greeks, p.quantity)
                elif p.asset_type == "stock":
                    sign = 1.0 if p.side == "LONG" else -1.0
                    hisseler_greek = GreekSet(delta=sign, gamma=0.0, theta=0.0, vega=0.0, iv=0.0)
                    agg.add_position(hisseler_greek, p.quantity, multiplier=1)
                    
            caps = CONFIG.greeks
            return {
                "delta": {
                    "value": round(agg.delta, 2),
                    "limit": caps.max_net_delta,
                    "pct": round(abs(agg.delta) / caps.max_net_delta * 100, 1) if caps.max_net_delta > 0 else 0
                },
                "gamma": {
                    "value": round(agg.gamma, 4),
                    "limit": caps.max_net_gamma,
                    "pct": round(abs(agg.gamma) / caps.max_net_gamma * 100, 1) if caps.max_net_gamma > 0 else 0
                },
                "theta": {
                    "value": round(agg.theta, 2),
                    "limit": caps.max_net_theta,
                    "pct": round(abs(agg.theta) / abs(caps.max_net_theta) * 100, 1) if caps.max_net_theta != 0 else 0
                },
                "vega": {
                    "value": round(agg.vega, 2),
                    "limit": caps.max_net_vega,
                    "pct": round(abs(agg.vega) / caps.max_net_vega * 100, 1) if caps.max_net_vega > 0 else 0
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/sector-clustering")
def get_sectors():
    try:
        with PORTFOLIO_LOCK:
            _store, portfolio = get_portfolio_status()
            sector_counts = {}
            for pos in portfolio.open_positions:
                sector_counts[pos.sector] = sector_counts.get(pos.sector, 0) + 1
            
            limit = CONFIG.portfolio.max_per_sector_positions
            return {
                "limit": limit,
                "distribution": sector_counts
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/trades")
def get_trades():
    try:
        return parse_trade_log()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/equity")
def get_equity(limit: int = 1000):
    """Temiz bakiye eğrisi — data/equity_curve.csv (main.append_equity_point yazar).
    trade_log.txt parse'ından (parse_trade_log) daha sağlam kaynak; grafik bunu tercih
    eder, boşsa /api/trades equity_curve'üne düşer. SİNYAL-ONLY, yalnız okuma."""
    import csv as _csv
    path = DATA_DIR / "equity_curve.csv"
    if not path.exists():
        return {"points": []}
    try:
        points = []
        with path.open(encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                ts = (row.get("timestamp") or "").strip()
                try:
                    eq = float(row["equity"])
                except (KeyError, ValueError, TypeError):
                    continue
                # renderEquityChart pt.time'ı ' ' ile böler → "YYYY-MM-DD HH:MM" formatla
                points.append({"time": ts.replace("T", " ")[:16], "equity": round(eq, 2)})
        if limit > 0:
            points = points[-limit:]
        return {"points": points}
    except Exception as e:
        logger.warning(f"[web] equity_curve.csv okunamadı: {e}")
        return {"points": []}

@app.get("/api/tearsheet")
def get_tearsheet(refresh: bool = False):
    """P0-2 — quantstats tear-sheet (HTML). nihai.web.tearsheet.build_daily_returns
    kapanan sanal işlemlerin (data/trades.csv) GERÇEKLEŞMİŞ getirisinden günlük seri
    kurar; bu GEÇMİŞ MUHASEBE raporudur, ileri kestirim değildir (SİNYAL-ONLY).

    data/tearsheet.html <6 saatlik ise onu servis eder; yoksa/eskiyse veya ?refresh=1
    verilirse önce yeniden üretir. Üretim bloklayan dosya-IO + (ilk seferde) quantstats/
    matplotlib CPU işi olduğundan /api/var deseninde `def` (async def DEĞİL). quantstats
    kurulu değilse 503 JSON döner (kurulum ipucuyla)."""
    import time

    from nihai.web import tearsheet as ts

    path = ts.TEARSHEET_HTML
    stale = True
    if path.exists():
        stale = (time.time() - path.stat().st_mtime) >= 6 * 3600

    if refresh or stale:
        try:
            ts.generate_tearsheet(path)
        except ImportError as e:
            return JSONResponse(
                {"ok": False, "error": f"quantstats kurulu değil: {e}",
                 "hint": "Kurulum: .venv/Scripts/python.exe -m pip install quantstats"},
                status_code=503,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    if not path.exists():
        raise HTTPException(status_code=500, detail="tearsheet üretilemedi (dosya bulunamadı)")
    return HTMLResponse(path.read_text(encoding="utf-8"))

@app.get("/api/logs")
def get_logs():
    try:
        path = LOG_DIR / "system.log"
        if not path.exists():
            return []
            
        # Performanslı tail okuma algoritması (bellek ve CPU dostu)
        size = path.stat().st_size
        if size < 256 * 1024:  # 256KB altı ise doğrudan oku
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                return [line.strip() for line in lines[-100:]]
                
        # 256KB üstü büyük log dosyaları için sondan geriye hızlı oku
        #
        # Tur-81 (Codex triyajı, madde C): eski gövde her adımda `buffer = chunk + buffer`
        # ile TAMPONUN TAMAMINI kopyalıyor ve `buffer.split(b"\n")` ile TAMAMINI yeniden
        # tarıyordu → okunan kuyruk B bayt / K blok iken O(B·K) kopyalama+tarama. Parçalar
        # listede TUTULUR, yalnız YENİ parça sayılır, birleştirme SONDA bir kez yapılır
        # (Tur-35'in `tail_lines` dersinin bu dosyadaki kopyası). Durma koşulu BİREBİR
        # korunur: `split` daima `satır_sonu + 1` öğe verir, yani eski `len(lines_list)<=150`
        # ile yeni `newlines < 150` AYNI adımda durur. UTF-8 `ignore` davranışı da aynıdır.
        with path.open("rb") as f:
            f.seek(0, 2)
            pointer = f.tell()
            chunks: list[bytes] = []
            newlines = 0
            block_size = 4096
            # En az 150 satır bulana kadar geriye oku (strip olunca 100 kalması için)
            while pointer > 0 and newlines < 150:
                step = min(block_size, pointer)
                pointer -= step
                f.seek(pointer)
                chunk = f.read(step)
                chunks.append(chunk)
                newlines += chunk.count(b"\n")
            lines_list = b"".join(reversed(chunks)).split(b"\n")

            decoded_lines = [line.decode("utf-8", errors="ignore").strip() for line in lines_list]
            valid_lines = [line for line in decoded_lines if line]
            return valid_lines[-100:]
    except Exception as e:
        return [f"Log okuma hatası: {e!s}"]

@app.get("/api/config")
def get_config():
    try:
        return {
            "per_trade_risk_pct": CONFIG.risk.per_trade_risk_pct,
            "daily_max_loss_pct": CONFIG.risk.daily_max_loss_pct,
            "consecutive_stops_cooldown": CONFIG.risk.consecutive_stops_cooldown,
            "max_gross_risk_pct": CONFIG.portfolio.max_gross_risk_pct,
            "max_net_directional_pct": CONFIG.portfolio.max_net_directional_pct,
            "wash_sale_days": CONFIG.regulation.wash_sale_days,
            "settlement_days": CONFIG.regulation.settlement_days,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/close-position")
def close_position(req: ClosePositionRequest):
    symbol = req.symbol.upper()
    try:
        # Boş string ve case-sensitivity normalizasyonu
        expiry = None if not req.expiry or req.expiry.strip() == "" else req.expiry
        option_type = None if not req.option_type or req.option_type.strip() == "" else req.option_type.lower()
        strike = req.strike
        
        with PORTFOLIO_LOCK:
            store, portfolio, pmgr, canli = _yazma_baglami()
            
            pos = next((p for p in portfolio.open_positions if 
                        p.symbol.upper() == symbol and 
                        p.asset_type == req.asset_type and
                        (p.expiry == expiry if expiry is not None else True) and
                        (p.strike == strike if strike is not None else True) and
                        (p.option_type.lower() == option_type if (option_type is not None and p.option_type is not None) else True)), None)
            if not pos:
                raise HTTPException(status_code=404, detail=f"{symbol} ({req.asset_type}) açık pozisyonu bulunamadı.")
            
            provider = get_provider(DATA_PROVIDER)
            
            # Fiyatı belirle
            if pos.asset_type == "option":
                from nihai.risk.option_strategy import get_option_premium_now
                price = None
                if pos.option_type and pos.strike and pos.expiry:
                    price = get_option_premium_now(
                        provider, pos.symbol, pos.option_type, pos.strike, pos.expiry)
                if price is None or price <= 0:
                    price = pos.last_mark or pos.entry_price
            else:
                q = provider.get_quote(pos.symbol)
                price = q.last if q and q.last > 0 else (pos.last_mark or pos.entry_price)
                
            closed = pmgr.force_close(pos, price, "WEB_MANUAL_CLOSE", now_et_naive())
            if closed:
                # Kapanış sonrası state'e kaydet (callbacks thread-safe içinde tetiklenecektir, 
                # ama state kalıcılığı için save() yapmalıyız)
                if not canli:   # canlıda defteri pmgr'ın kapanış kancası tutar (main._apply_close_bookkeeping)
                    store.record_trade_pnl(closed.realized_pnl)
                    store.register_close_for_wash_sale(closed.pos.symbol, closed.realized_pnl,
                                                       closed.exit_time, closed.pos.asset_type)
                    mult = 100 if closed.pos.asset_type == "option" else 1
                    store.add_unsettled_proceeds(closed.exit_price * closed.pos.quantity * mult,
                                                 closed.exit_time.date(), closed.pos.side)
                
                if store.kill_switch_active:
                    logger.warning("[web] Kill-switch aktif oldu!")
                
                store.save()
                return {"success": True, "message": f"{symbol} pozisyonu {closed.exit_price:.2f} fiyattan kapatıldı. PnL: {closed.realized_pnl:+.2f}"}
            raise HTTPException(status_code=500, detail="Pozisyon kapatılamadı.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[web] Kapatma hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/flatten")
def flatten_all_positions():
    try:
        with PORTFOLIO_LOCK:
            store, portfolio, pmgr, canli = _yazma_baglami()
            
            if not portfolio.open_positions:
                return {"success": True, "message": "Kapatılacak açık pozisyon bulunmamaktadır."}
                
            provider = get_provider(DATA_PROVIDER)
            
            closed_list = pmgr.flatten_all(provider, now_et_naive(), reason="WEB_MANUAL_FLATTEN")
            
            for ce in ([] if canli else closed_list):   # canlıda kapanış kancası tutar
                store.record_trade_pnl(ce.realized_pnl)
                store.register_close_for_wash_sale(ce.pos.symbol, ce.realized_pnl, ce.exit_time,
                                                  ce.pos.asset_type)
                mult = 100 if ce.pos.asset_type == "option" else 1
                store.add_unsettled_proceeds(ce.exit_price * ce.pos.quantity * mult, ce.exit_time.date(),
                                            ce.pos.side)
                
            store.save()
            return {"success": True, "message": f"{len(closed_list)} açık pozisyon başarıyla kapatıldı (Flatten All)."}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[web] Flatten hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/reset-killswitch")
def reset_killswitch():
    try:
        with PORTFOLIO_LOCK:
            store = _yazma_baglami()[0]
            store.kill_switch_active = False
            store.recent_trades_pnl.clear()
            store.save()
            return {"success": True, "message": "Kill-switch ve son işlemler geçmişi başarıyla sıfırlandı."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

class UpdateConfigRequest(BaseModel):
    per_trade_risk_pct: float
    daily_max_loss_pct: float
    consecutive_stops_cooldown: int
    max_gross_risk_pct: float
    max_net_directional_pct: float
    wash_sale_days: int
    settlement_days: int

@app.post("/api/update-config")
def update_config(req: UpdateConfigRequest):
    try:
        def update_env_file(key: str, val: str):
            env_path = Path(__file__).resolve().parents[2] / ".env"  # repo kökü
            if not env_path.exists():
                return
            content = env_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            updated = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{key}="):
                    lines[i] = f"{key}={val}"
                    updated = True
                    break
            if not updated:
                lines.append(f"{key}={val}")
            # Atomik yazım (Tur-11): yazma ortasında kesinti .env'i yarım bırakmasın —
            # tüm servisler bu dosyayla açılıyor. Ara dosya YAZANA özel olmalı (Tur-28):
            # sabit '.env.tmp' iki eşzamanlı yazanda yarım içerik yayınlıyordu.
            atomic_write_text(env_path, "\n".join(lines) + "\n")

        # 1. OS Env ve .env Dosyasını Güncelle
        os.environ["PER_TRADE_RISK_PCT"] = str(req.per_trade_risk_pct)
        os.environ["DAILY_MAX_LOSS_PCT"] = str(req.daily_max_loss_pct)
        os.environ["CONSECUTIVE_STOPS_COOLDOWN"] = str(req.consecutive_stops_cooldown)
        os.environ["MAX_GROSS_RISK_PCT"] = str(req.max_gross_risk_pct)
        os.environ["MAX_NET_DIRECTIONAL_PCT"] = str(req.max_net_directional_pct)
        os.environ["WASH_SALE_DAYS"] = str(req.wash_sale_days)
        os.environ["SETTLEMENT_DAYS"] = str(req.settlement_days)

        # .env yazımları + CONFIG override TEK kilit altında (Tur-11): eşzamanlı iki POST'un
        # read-modify-write turları birbirinin anahtarını ezmesin.
        with PORTFOLIO_LOCK:
            update_env_file("PER_TRADE_RISK_PCT", str(req.per_trade_risk_pct))
            update_env_file("DAILY_MAX_LOSS_PCT", str(req.daily_max_loss_pct))
            update_env_file("CONSECUTIVE_STOPS_COOLDOWN", str(req.consecutive_stops_cooldown))
            update_env_file("MAX_GROSS_RISK_PCT", str(req.max_gross_risk_pct))
            update_env_file("MAX_NET_DIRECTIONAL_PCT", str(req.max_net_directional_pct))
            update_env_file("WASH_SALE_DAYS", str(req.wash_sale_days))
            update_env_file("SETTLEMENT_DAYS", str(req.settlement_days))

            # 2. CONFIG Nesnesini Runtime'da Güncelle (frozen=True dataclass override)
            object.__setattr__(CONFIG.risk, 'per_trade_risk_pct', req.per_trade_risk_pct)
            object.__setattr__(CONFIG.risk, 'daily_max_loss_pct', req.daily_max_loss_pct)
            object.__setattr__(CONFIG.risk, 'consecutive_stops_cooldown', req.consecutive_stops_cooldown)
            
            object.__setattr__(CONFIG.portfolio, 'max_gross_risk_pct', req.max_gross_risk_pct)
            object.__setattr__(CONFIG.portfolio, 'max_net_directional_pct', req.max_net_directional_pct)
            
            object.__setattr__(CONFIG.regulation, 'wash_sale_days', req.wash_sale_days)
            object.__setattr__(CONFIG.regulation, 'settlement_days', req.settlement_days)

        logger.info(f"[web] Sistem parametreleri başarıyla güncellendi ve kaydedildi. Cooldown stop: {req.consecutive_stops_cooldown}")
        return {"success": True, "message": "Yapılandırma parametreleri başarıyla güncellendi ve kalıcı olarak kaydedildi."}
    except Exception as e:
        logger.exception(f"[web] Config update error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

class UpdatePositionLimitsRequest(BaseModel):
    symbol: str
    asset_type: str
    stop: float
    target: float
    expiry: str | None = None
    strike: float | None = None
    option_type: str | None = None

class OpenManualPositionRequest(BaseModel):
    symbol: str
    asset_type: str
    side: str
    quantity: int
    price: float | None = None
    stop: float
    target: float
    expiry: str | None = None
    strike: float | None = None
    option_type: str | None = None

@app.post("/api/update-position-limits")
def update_position_limits(req: UpdatePositionLimitsRequest):
    symbol = req.symbol.upper()
    try:
        with PORTFOLIO_LOCK:
            store, portfolio, pmgr, _canli = _yazma_baglami()
            
            # Parametre normalizasyonu
            expiry = None if not req.expiry or req.expiry.strip() == "" else req.expiry
            option_type = None if not req.option_type or req.option_type.strip() == "" else req.option_type.lower()
            
            # Eşleşen açık pozisyonu bul
            pos = next((p for p in portfolio.open_positions if 
                        p.symbol.upper() == symbol and 
                        p.asset_type == req.asset_type and
                        (p.expiry == expiry if expiry is not None else True) and
                        (p.strike == req.strike if req.strike is not None else True) and
                        (p.option_type.lower() == option_type if (option_type is not None and p.option_type is not None) else True)), None)
            
            if not pos:
                raise HTTPException(status_code=404, detail=f"{symbol} ({req.asset_type}) açık pozisyonu bulunamadı.")
            
            # Stop & target güncelle
            pos.stop = req.stop
            pos.target = req.target
            
            # PositionManager'daki tracker'ı da güncellemeliyiz ki check_exit düzgün çalışsın
            
            # Tracker key bul ve stop/target güncelle
            # PMGR tracker key formatı: key = f"{p.symbol}_{p.entry_time.isoformat()}_{p.side}_{id(p)}"
            for tracker in pmgr._trackers.values():
                if tracker.pos is pos:
                    tracker.initial_stop = req.stop
                    break
            
            store.save()
            logger.info(f"[web] {symbol} pozisyon limitleri başarıyla güncellendi: Stop={req.stop}, Target={req.target}")
            return {"success": True, "message": f"{symbol} limitleri güncellendi (Stop: ${req.stop:.2f}, Hedef: ${req.target:.2f})"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[web] Limit güncelleme hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/open-position")
def open_manual_position(req: OpenManualPositionRequest):
    symbol = req.symbol.upper()
    side = req.side.upper()
    if side not in ("LONG", "SHORT"):
        raise HTTPException(status_code=400, detail="Geçersiz yön. LONG veya SHORT olmalıdır.")
        
    try:
        with PORTFOLIO_LOCK:
            store, portfolio, pmgr, _canli = _yazma_baglami()
            
            provider = get_provider(DATA_PROVIDER)
            
            # Opsiyon normalizasyonları
            expiry = None if not req.expiry or req.expiry.strip() == "" else req.expiry
            option_type = None if not req.option_type or req.option_type.strip() == "" else req.option_type.lower()
            
            # Fiyatı belirle
            price = req.price
            if not price or price <= 0:
                if req.asset_type == "option":
                    from nihai.risk.option_strategy import get_option_premium_now
                    # Opsiyon premium'u ancak tüm kontrat alanları (tür/strike/vade) verilmişse
                    # çekilebilir; eksikse None bırak → aşağıda 400 ile elle fiyat istenir.
                    if option_type and req.strike and expiry:
                        price = get_option_premium_now(provider, symbol, option_type, req.strike, expiry)
                    else:
                        price = None
                else:
                    q = provider.get_quote(symbol)
                    price = q.last if q and q.last > 0 else None
                    
            if not price or price <= 0:
                raise HTTPException(status_code=400, detail=f"{symbol} için anlık fiyat alınamadı. Lütfen giriş fiyatını elle girin.")
                
            sig_action = SignalAction.LONG if side == "LONG" else SignalAction.SHORT
            sig = Signal(
                symbol=symbol,
                action=sig_action,
                price=price,
                stop=req.stop,
                target=req.target,
                reasons=["MANUAL_OPERATOR_ORDER"]
            )
            
            sizing = {
                "shares": req.quantity if req.asset_type == "stock" else 0,
                "contracts": req.quantity if req.asset_type == "option" else 0,
                "risk_dollars": abs(price - req.stop) * req.quantity
            }
            
            greeks = None
            if req.asset_type == "option":
                greeks = GreekSet(delta=0.5 if side == "LONG" else -0.5, gamma=0.0, theta=0.0, vega=0.0, iv=0.3)
                
            pmgr.open_from_signal(
                signal=sig,
                sizing=sizing,
                now=now_et_naive(),
                sector=get_sector(symbol),
                asset_type=req.asset_type,
                option_type=option_type,
                strike=req.strike,
                expiry=expiry,
                greeks=greeks,
                iv=0.3 if req.asset_type == "option" else None,
                strategy_name="Operator Manual Order"
            )
            
            store.save()
            logger.info(f"[web] Operatör manuel işlem açtı: {symbol} {side} {req.quantity}x @ {price:.2f}")
            return {"success": True, "message": f"{symbol} ({side}) manuel pozisyonu {price:.2f} fiyattan başarıyla açıldı."}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[web] Manuel işlem açma hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

class WatchlistAddRequest(BaseModel):
    symbol: str
    pin: bool = True

class WatchlistRemoveRequest(BaseModel):
    symbol: str
    block: bool = True

@app.post("/api/watchlist/add")
def watchlist_add(req: WatchlistAddRequest):
    symbol = req.symbol.upper().strip()
    try:
        import nihai.execution.watchlist as watchlist
        with PORTFOLIO_LOCK:
            changed = watchlist.add(symbol, pin=req.pin, source="web")
            if changed:
                return {"success": True, "message": f"{symbol} izleme listesine eklendi (Pin={req.pin})."}
            return {"success": False, "message": f"{symbol} zaten listede veya eklenemedi."}
    except Exception as e:
        logger.exception(f"[web] Watchlist ekleme hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/watchlist/remove")
def watchlist_remove(req: WatchlistRemoveRequest):
    symbol = req.symbol.upper().strip()
    try:
        import nihai.execution.watchlist as watchlist
        with PORTFOLIO_LOCK:
            changed = watchlist.remove(symbol, block=req.block)
            if changed:
                return {"success": True, "message": f"{symbol} izleme listesinden çıkarıldı (Block={req.block})."}
            return {"success": False, "message": f"{symbol} listede bulunamadı."}
    except Exception as e:
        logger.exception(f"[web] Watchlist çıkarma hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

# Tur-18 — backend'de VAR olup panoda GÖRÜNMEYEN yeteneklerin uçları (Quant Beyin,
# piyasa istihbaratı, opsiyon zinciri, araçlar, sistem sağlığı). Salt-okunur/gölge;
# sinyal/execution'a bağlanmaz. En sonda import edilir: api_extra `_analysis_provider`'ı
# çağrı anında geç-import ettiğinden döngü oluşmaz.
from nihai.web.api_extra import router as _extra_router  # noqa: E402

app.include_router(_extra_router)

# Tur-56 — QUANT MODELLERİ: quant_math/options_math/advanced_math/microstructure/
# portfolio/regime_hmm çıktılarının GRAFİĞE HAZIR yüzeyi. Yeni matematik yok; var olan
# fonksiyonların seri/ızgara çıktıları. Aynı geç-import gerekçesi (api_extra ile aynı).
from nihai.web.api_quant import router as _quant_router  # noqa: E402

app.include_router(_quant_router)

from nihai.web.api_lab import router as _lab_router  # noqa: E402
from nihai.web.api_operations import router as _operations_router  # noqa: E402

app.include_router(_lab_router)
app.include_router(_operations_router)

# Tur-103 — Emir Akışı (footprint · hacim profili · baloncuklar · kriptoda likidite ısı haritası). KARAR-DIŞI.
from nihai.web.api_orderflow import router as _orderflow_router  # noqa: E402

app.include_router(_orderflow_router)


# Threaded starter function
def start_web_server():
    """Arka planda FastAPI sunucusunu uvicorn ile başlatır."""
    logger.info(f"[web] Web sunucusu başlatılıyor: http://{WEB_HOST}:{WEB_PORT}")
    # Canlı fiyat-tahmin servisini arka planda başlat (tek paylaşılan provider).
    try:
        from nihai.execution.live_predict_service import get_service
        get_service().start()
    except Exception as e:
        logger.warning(f"[web] tahmin servisi başlatılamadı: {e}")
    config = uvicorn.Config(app, host=WEB_HOST, port=WEB_PORT, log_level="warning")
    server = uvicorn.Server(config)
    
    # Thread daemon olarak ayarlanır, böylece main.py durduğunda web sunucusu da durur.
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return t

if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    # Tek başına test etmek için
    logger.info("Web sunucusu bağımsız olarak başlatılıyor...")
    try:
        from nihai.execution.live_predict_service import get_service
        get_service().start()
    except Exception as e:
        logger.warning(f"[web] tahmin servisi başlatılamadı: {e}")
    uvicorn.run(app, host="127.0.0.1", port=8765)
