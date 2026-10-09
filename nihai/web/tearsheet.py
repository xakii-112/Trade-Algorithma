"""
tearsheet.py — P0-2: quantstats ile GERÇEKLEŞMİŞ getiri tear-sheet'i (HTML).

VERİ KAYNAĞI SEÇİMİ (gerekçe)
------------------------------
Aranan şey: kapanan SANAL pozisyonların gerçekleşmiş PnL'i + kapanış zamanı.
Adaylar ve neden ELENDİLER:

  * data/predictions_log.csv (nihai/execution/prediction_tracker.py çözülmüş satırlar) —
    bu dosya YÖN tahmininin isabetini ölçer (`fwd_ret_pct` = horizon bar sonraki HAM fiyat
    hareketi) ve kapı-engelli (meta_gate ile elenmiş) WAIT adaylarını da içerir. Stop/target/
    trailing/slipaj/komisyon/pozisyon-boyutu bu getiriye YANSIMAZ (modülün kendi docstring'i
    "paper-trade eşdeğeri" bir yön-testi olduğunu söylüyor). Gerçek portföy PnL'i değil →
    performans tear-sheet'i için uygunsuz kaynak.
  * logs/trade_log.txt — insan-okur post-mortem blokları. web_dashboard.parse_trade_log()
    bunu zaten kırılgan string-split ile çözüyor; aynı kapanış olayının makine-okur ikizi
    (aşağıya bak) dururken tekrar regex/parse riski almaya gerek yok.
  * data/equity_curve.csv (main.append_equity_point) — canlı döngüde PERİYODİK poll ile
    yazılır, ardışık-aynı-değer epsilon'la dedup'lanır ve yalnız seans açıkken eklenir
    (main.py "C6" notu). Pano çizgi-grafiği ve research/postmortem.py'nin kaba max-drawdown
    hesabı için yeterli; ama işlem-başı `realized_pnl` YOK ve iki kapanış aynı poll
    aralığına denk gelirse tek satıra sıkışabilir → "kaç işlem kapandı" (min_trades kapısı)
    burada güvenilir sayılamaz.
  * data/trades.csv (SEÇİLEN) — nihai/risk/position_manager.py `_close()` HER kapanışta
    nihai/signals/signal_engine.py `log_trade_csv()` ile senkron/eşzamanlı yazar (aynı
    fonksiyon insan-okur trade_log.txt bloğunu da üretir — trades.csv onun makine-okur
    ikizidir). Şema (signal_engine.TRADES_CSV_HEADER): exit_time, symbol, asset_type, side,
    strategy, entry_time, entry_price, exit_price, quantity, risk_dollars, planned_rr,
    realized_pnl, r_multiple, mae_pct, mfe_pct, exit_trigger, sector, nlv_after. Bu dosya
    zaten research/postmortem.py'de "tek temiz kaynak" (post-mortem/dashboard/VaR analizleri)
    olarak benimsenmiş — tear-sheet aynı aileye katılıyor. Kritik olarak `nlv_after` sütunu
    KAPANIŞ ANINDAKİ portföy sermayesini taşır → "kaynakta sermaye bilgisi" burada hazır
    (aşağıdaki EQUITY TABANI bölümüne bak).

EQUITY TABANI VARSAYIMI
------------------------
`nlv_after` TÜM kapanan işlemlerde sayısal/doluysa equity işaretleri DOĞRUDAN ondan alınır
(kaynağın kendi sermaye muhasebesi — compounding, komisyon, slipaj zaten içinde). Herhangi
BİR satırda `nlv_after` eksik/bozuksa (eski şema, kısmi/bozuk yazım) TÜM seri karışık
kaynaktan süreksizlik taşımasın diye `nihai.config.INITIAL_EQUITY` (env `INITIAL_EQUITY`,
varsayılan $25000 — repo'daki TEK sabit sermaye varsayımı, web_dashboard/state.py da aynısını
kullanır) tabanından `realized_pnl` kümülatif toplamıyla YENİDEN kurulur — ya hep ya hiç.

GÜNLÜK SERİ: birden çok işlem aynı takvim gününde kapanabilir; her gün için GÜN SONU (o
günün en son kapanışı) equity işareti alınır ve pct_change ile yüzde getiriye çevrilir.
İşlem kapanmayan günler seride YOKTUR (yapay 0-getirili satır eklenmez — quantstats seyrek
tarih indeksiyle çalışabilir; dürüst boşluk, sahte doluluktan iyidir).

SİNYAL-ONLY / DÜRÜSTLÜK: bu GEÇMİŞ MUHASEBE raporudur (gerçekleşmiş PnL), ileri kestirim
DEĞİLDİR. Alfa iddiası taşımaz.
"""
from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from nihai.atomic_io import atomic_replace, atomic_write_text, tmp_path
from nihai.config import DATA_DIR, INITIAL_EQUITY

TRADES_CSV = DATA_DIR / "trades.csv"
TEARSHEET_HTML = DATA_DIR / "tearsheet.html"

# build_daily_returns/_load_closed_trades'in okuduğu/ürettiği minimum sütun kümesi.
_TRADE_COLUMNS = ("exit_time", "realized_pnl", "nlv_after")


def _normalize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """exit_time/realized_pnl/nlv_after'ı sayısallaştır, geçersiz satırları at, zamana göre
    sırala. Eksik sütun (eski şema) NaN ile eklenir — KeyError yerine nazik bozulma."""
    df = df.copy()
    for col in _TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    df["nlv_after"] = pd.to_numeric(df["nlv_after"], errors="coerce")
    df = df.dropna(subset=["exit_time", "realized_pnl"])
    return df.sort_values("exit_time").reset_index(drop=True)


def _load_closed_trades(csv_path: str | Path | None = None) -> pd.DataFrame:
    """data/trades.csv'yi (veya verilen yolu) oku + temizle. Dosya yoksa boş DataFrame
    döner (exception değil — canlı akışı asla bozmayan log_trade_csv'nin okuma tarafındaki
    simetriği)."""
    path = Path(csv_path) if csv_path is not None else TRADES_CSV
    if not path.exists():
        return pd.DataFrame(columns=list(_TRADE_COLUMNS))
    return _normalize_trades(pd.read_csv(path))


def build_daily_returns(trades: pd.DataFrame | str | Path | None = None) -> pd.Series:
    """Kapanan sanal işlemlerden GÜNLÜK yüzde getiri serisi (quantstats girdisi).

    trades:
      - None (varsayılan)  → data/trades.csv okunur (üretim yolu).
      - str/Path           → o CSV okunur (aynı şema varsayılır).
      - pd.DataFrame       → doğrudan kullanılır (testler sentetik veri için böyle çağırır);
                              en az `exit_time` + `realized_pnl` sütunu gerekir, `nlv_after`
                              opsiyoneldir (yoksa/eksikse aşağıdaki varsayıma düşülür).

    Equity tabanı: modül docstring'indeki "EQUITY TABANI VARSAYIMI" bölümüne bak — kaynakta
    (nlv_after) sermaye bilgisi TAM doluysa o kullanılır, değilse INITIAL_EQUITY + kümülatif
    realized_pnl varsayımına düşülür.

    Döner: DatetimeIndex'li (gün, normalize edilmiş) pd.Series; değerler ondalık yüzde
    getiri (0.01 = %1). Kapanan işlem yoksa boş Series (dtype=float)."""
    df = _normalize_trades(trades) if isinstance(trades, pd.DataFrame) else _load_closed_trades(trades)
    if df.empty:
        return pd.Series(dtype=float, name="returns")

    # Kaynakta sermaye bilgisi (nlv_after) TAM doluysa onu equity işareti olarak kullan;
    # aksi halde TÜM seri için INITIAL_EQUITY tabanlı kümülatif toplama düş (karışık
    # kaynaktan doğacak süreksizliği önlemek için ya hep ya hiç).
    if df["nlv_after"].notna().all():
        marks = df["nlv_after"].to_numpy(dtype=float)
    else:
        marks = INITIAL_EQUITY + df["realized_pnl"].cumsum().to_numpy(dtype=float)

    # İlk işlem ÖNCESİ equity: marks[0]'dan o işlemin PnL'i çıkarılarak geri hesaplanır.
    # nlv_after dalında bu kaynağın kendi taban sermayesini verir; cumsum dalında tam olarak
    # INITIAL_EQUITY'e eşittir (marks[0] = INITIAL_EQUITY + pnl[0]) — iki dal da tutarlı.
    prior_equity = float(marks[0] - df["realized_pnl"].iloc[0])

    day_index = df["exit_time"].dt.normalize()
    daily_close = pd.Series(marks, index=day_index).groupby(level=0).last()

    base_day = daily_close.index[0] - pd.Timedelta(days=1)
    equity_curve = pd.concat([pd.Series([prior_equity], index=[base_day]), daily_close])

    returns = equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    returns.index.name = "date"
    returns.name = "returns"
    return returns


def _ai_appendix_html() -> str:
    """Tear-sheet'in AI KALİBRASYON EKİ (Tur-37) — KARAR-DIŞI, SALT OKUNUR.

    NEDEN BURADA: tear-sheet GERÇEKLEŞMİŞ getirinin muhasebesidir ("ne oldu"). Sistemin
    kendi TAHMİNLERİNİN kalibrasyonu ("söylediğim %60 gerçekten %60 muydu") tamamen ayrı
    bir soruydu ve yalnız `nihai/ai` artefaktlarında yaşıyordu — raporun okuyucusu ikisini
    hiçbir yerde YAN YANA göremiyordu. Getiri iyi ama kalibrasyon bozuksa (ya da tersi)
    okunması gereken şey tam olarak bu ikilidir.

    `calibration_loop` ve `adaptive_calibration` modüllerinin ikinci üretim çağırıcısıdır;
    yalnız KAYITLI artefaktı okur, hiçbir kalibratörü yeniden fit ETMEZ (rapor üretimi bir
    ölçüm koşusu tetiklemez — gece görevi `daily_summary` üzerinden yürür).

    AI katmanı yok/artefakt boş → dürüst "ölçülmedi" satırı. Hata → boş string (tear-sheet
    hiçbir koşulda düşmez)."""
    try:
        from nihai.ai.observatory import calibration_line
        line = calibration_line()
    except Exception:
        return ""
    if not line:
        return ""
    safe = html.escape(line)
    return f"""
<div style="font-family: system-ui, sans-serif; background:#0d1124; color:#94a3b8;
            padding:1rem 1.5rem; margin:1.5rem auto; max-width:900px; border-radius:8px;
            border-left:4px solid #22d3ee; font-size:0.9rem; line-height:1.6;">
<strong style="color:#f8fafc;">AI kalibrasyon eki (KARAR-DIŞI)</strong><br>
<code style="color:#22d3ee;">{safe}</code><br>
<span style="font-size:0.8rem;">Bu satır sistemin kendi olasılık tahminlerinin
KALİBRASYON HİJYENİDİR — bir kenar (alfa) iddiası DEĞİLDİR. Yukarıdaki getiri raporu
gerçekleşmiş sanal muhasebedir; bu ek, o tahminlerin ne kadar dürüst olduğunu ölçer.</span>
</div>
"""


def _with_ai_appendix(page: str) -> str:
    """Eki `</body>`'den hemen önce yerleştir; kapanış etiketi yoksa sona ekle.

    quantstats'ın ürettiği HTML'in iç yapısına DOKUNULMAZ (sürüm değişirse kırılacak bir
    seçici/regex yok) — tek varsayım `</body>`'nin varlığı, o da yoksa düz ekleme."""
    block = _ai_appendix_html()
    if not block:
        return page
    idx = page.rfind("</body>")
    return page + block if idx < 0 else page[:idx] + block + page[idx:]


def _insufficient_data_html(n_trades: int, n_days: int, min_trades: int) -> str:
    """Yetersiz veri yolu — exception yerine Türkçe açıklamalı basit HTML."""
    return f"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<title>Nihai Algoritma — Tear-Sheet</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #0d1124; color: #e2e8f0;
        padding: 2.5rem; max-width: 640px; margin: 0 auto; }}
h1 {{ color: #f8fafc; font-size: 1.4rem; }}
p {{ line-height: 1.6; color: #94a3b8; }}
.stat {{ color: #22d3ee; font-weight: 600; }}
</style>
</head>
<body>
<h1>Tear-Sheet henüz üretilemiyor</h1>
<p>İstatistiksel olarak anlamlı bir performans raporu üretmek için yeterli kapanmış sanal
işlem birikmedi.</p>
<p>Kapanan işlem: <span class="stat">{n_trades}</span> (gerekli ≥ {min_trades}) ·
Farklı gün: <span class="stat">{n_days}</span> (gerekli ≥ 2)</p>
<p>Bu sayfa, sistem yeni sinyal kapanışları biriktirdikçe otomatik olarak dolacaktır.
SİNYAL-ONLY: bu rapor gerçekleşmiş sanal işlemlerin geçmiş muhasebesidir, ileri kestirim
değildir.</p>
</body>
</html>
"""


def generate_tearsheet(output_path: str | Path, min_trades: int = 10) -> Path:
    """data/trades.csv'den quantstats HTML tear-sheet'i üretir → output_path.

    Yetersiz veri (kapanan işlem < min_trades VEYA farklı gün < 2): exception ATMAZ,
    Türkçe açıklamalı basit bir HTML üretir (_insufficient_data_html). Bu yolda quantstats'a
    HİÇ dokunulmaz.

    quantstats YALNIZ bu fonksiyon içinde, ve yalnız yeterli veri varsa lazy import edilir
    (modül yükleme anında quantstats zorunlu olmasın). matplotlib Agg backend'i quantstats'ın
    kendi pyplot importundan ÖNCE ayarlanır — nihai.notifier.chart_render.render_price_png'
    deki desen: sunucu (arka plan thread'i) içinde GUI backend'i hataya/askıya yol açabilir.

    Yazım OneDrive/AV kilidine dayanıklı: geçici dosyaya yazılır, sonra nihai.atomic_io ile
    hedefe atomik taşınır (yarım/kilitli dosya asla dışa görünmez).

    Döner: yazılan dosyanın Path'i (output_path ile aynı)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = _load_closed_trades()
    n_trades = len(df)
    returns = build_daily_returns(df)
    n_days = int(returns.shape[0])

    if n_trades < min_trades or n_days < 2:
        atomic_write_text(out, _with_ai_appendix(
            _insufficient_data_html(n_trades, n_days, min_trades)))
        return out

    import matplotlib
    matplotlib.use("Agg")          # quantstats'ın iç pyplot importundan ÖNCE (server-safe)
    import quantstats as qs

    tmp = tmp_path(out)   # yazana özel ara dosya (iki istek aynı anda üretirse çakışmasın)
    qs.reports.html(returns, output=str(tmp), title="Nihai Algoritma — Sanal Portföy Tear-Sheet")
    # AI kalibrasyon eki quantstats YAZDIKTAN SONRA enjekte edilir (kütüphanenin şablonuna
    # dokunmadan). Okuma/yazma hata verirse quantstats çıktısı OLDUĞU GİBİ kalır.
    try:
        tmp.write_text(_with_ai_appendix(tmp.read_text(encoding="utf-8")), encoding="utf-8")
    except OSError:
        pass
    atomic_replace(tmp, out)
    return out
