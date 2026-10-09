"""
signal_engine.py — Karar + işlem loglama, PDF formatına uygun.

PDF Referansları:
- Madde 19 : Kapsamlı karar/işlem loglaması (alınan + pas geçilen)
- Madde 37 : Karar günlüğü formatı (decision_log_YYYY_MM_DD.txt)
- Madde 38 : İşlem post-mortem formatı (trade_log.txt)
- Madde 50 : Realized slippage izleme (simülasyonda da modellenir)
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger

from nihai.atomic_io import atomic_replace, tmp_path
from nihai.config import DATA_DIR, LOG_DIR
from nihai.execution.cost_attribution import COST_FIELDS
from nihai.risk.options_analyzer import GreekSet
from nihai.risk.risk_manager import VirtualPosition
from nihai.signals.strategy import Signal


@dataclass
class TradeRecord:
    asset_label: str               # "AAPL 2026-06-19 C 200 (Long Call)"
    strategy: str                  # "Momentum Breakout V2 (BRK)"
    entry_time: datetime
    entry_price: float
    entry_greeks: GreekSet | None
    entry_iv: float | None
    entry_iv_rank: float | None
    dte_at_entry: int | None
    risk_dollars: float
    planned_rr: float
    contracts_or_shares: int

    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_trigger: str | None = None
    mae_pct: float | None = None   # Maximum Adverse Excursion
    mfe_pct: float | None = None   # Maximum Favorable Excursion
    iv_change: float | None = None
    realized_pnl: float | None = None
    nlv_after: float | None = None
    entry_reasons: list[str] = field(default_factory=list)
    realized_slippage: float = 0.0  # Madde 50
    cost_attribution: dict | None = None


def _today_log() -> Path:
    return LOG_DIR / f"decision_log_{datetime.now():%Y_%m_%d}.txt"


TRADE_LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB → rotasyon

# Retention (env-ayarlanabilir): rotasyon ÇIKTILARI sınırsız birikmesin diye en yeni
# N dosya tutulur, eskileri silinir. 0 = sınırsız (eski davranış). Tek-seferlik temizlik
# ile aynı politika; canlı çalışmada her rotasyon/başlangıçta otomatik uygulanır.
KEEP_TRADE_ARCHIVES = int(os.getenv("LOG_KEEP_TRADE_ARCHIVES", "5"))
KEEP_DECISION_LOGS = int(os.getenv("LOG_KEEP_DECISION_LOGS", "30"))
KEEP_SCREEN_CSV = int(os.getenv("LOG_KEEP_SCREEN_CSV", "30"))

# Launcher (PowerShell) yeniden-yönlendirme dosyaları Python tarafından yazılmıyor:
# hub-watchdog.ps1 → hub_server.{out,err}.log, harici başlatıcı → _server_restart.log.
# Bunlar rotasyonsuz birikiyordu (gözlem: hub_server.err.log ~500MB). prune_logs()
# başlangıçta bunları boyut tavanına kırpar (silmez — yazan süreç handle'ı geçerli kalsın).
LOG_TRUNCATE_MAX_BYTES = int(os.getenv("LOG_TRUNCATE_MAX_BYTES", str(50 * 1024 * 1024)))
_TRUNCATE_TARGETS = ("hub_server.err.log", "hub_server.out.log", "_server_restart.log")


def _truncate_big(name: str, max_bytes: int) -> int:
    """Dosya max_bytes'ı aşıyorsa içeriğini sıfırla (truncate). 1 = kırpıldı, 0 = dokunulmadı.
    Aktif yazan süreç inode'u korur (delete değil truncate); hata yutulur (best-effort)."""
    if max_bytes <= 0:
        return 0
    path = LOG_DIR / name
    try:
        if path.exists() and path.stat().st_size > max_bytes:
            path.open("w").close()
            return 1
    except Exception:
        pass
    return 0


def _prune_old(pattern: str, keep: int) -> int:
    """En yeni `keep` dosyayı tut, eskilerini sil. Silinen dosya sayısını döndürür.
    keep<=0 ise no-op (sınırsız). mtime'a göre sıralar; hata yutulur (best-effort)."""
    if keep <= 0:
        return 0
    try:
        files = sorted(LOG_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return 0
    removed = 0
    for old in files[keep:]:
        try:
            old.unlink()
            removed += 1
        except Exception:
            pass
    return removed


def prune_logs() -> dict[str, int]:
    """Tüm log ailelerinde retention uygula (başlangıçta bir kez çağrılır).
    Döner: {aile: silinen_dosya_sayısı}. trade_log.txt (aktif) ve system.log dokunulmaz."""
    return {
        "trade_archives": _prune_old("trade_log_*.txt", KEEP_TRADE_ARCHIVES),
        "decision_logs": _prune_old("decision_log_*.txt", KEEP_DECISION_LOGS),
        "screen_csv": _prune_old("auto_screen_*.csv", KEEP_SCREEN_CSV),
        "system_archives": _prune_old("system.*.log", KEEP_TRADE_ARCHIVES),
        "truncated_big": sum(_truncate_big(n, LOG_TRUNCATE_MAX_BYTES) for n in _TRUNCATE_TARGETS),
    }


def _trade_log() -> Path:
    """trade_log.txt yolu. Boyut eşiği aşılırsa zaman damgalı arşive döndürür (rotasyon),
    sonra eski arşivleri retention sınırına kırpar."""
    path = LOG_DIR / "trade_log.txt"
    try:
        if path.exists() and path.stat().st_size > TRADE_LOG_MAX_BYTES:
            archive = LOG_DIR / f"trade_log_{datetime.now():%Y%m%d_%H%M%S}.txt"
            path.rename(archive)
            _prune_old("trade_log_*.txt", KEEP_TRADE_ARCHIVES)
    except Exception:
        pass
    return path


def log_decision(action: str, detail: str) -> None:
    """Madde 37 formatı: ZAMAN | EYLEM | GEREKÇE."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} | {action:<13} | {detail}\n"
    path = _today_log()
    new = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if new:
            f.write(f"Dosya: {path.name}\n")
            f.write(f"{'ZAMAN':<19} | {'EYLEM':<13} | GEREKCE / DURUM DETAYI\n")
            f.write("-" * 80 + "\n")
        f.write(line)


def _zincire_yaz(tur: str, kayit: dict) -> None:
    """Madde 244 — kararı SHA-256 karar zincirine ekle (fail-soft, KARAR-DIŞI).

    NEDEN HER SATIR DEĞİL: `log_decision` tarayıcının her turunda her sembol için
    çağrılır (ölçüldü: tek günlük `decision_log` 35 MB, Tur-58). Zincirin amacı
    "her satırı say" değil, KARAR kaydının sonradan sessizce düzenlenemediğini
    göstermektir; bu yüzden yalnız KABUL EDİLEN sinyal ve KAPANAN işlem halkaya
    girer (canlı defterde ~510 işlem / 6 hafta).

    İDDİANIN SINIRI değişmedi: zincir kazara bozulmayı ve seçici düzenlemeyi
    yakalar, dosyaya yazma yetkisi olan kötü niyetli bir yazara karşı kanıt DEĞİLDİR
    (gerçek değiştirilemezlik `merkle_root`un dışarı taşınmasını ister).
    """
    if os.getenv("AUDIT_CHAIN", "on").strip().lower() in ("0", "off", "false", "no"):
        return
    try:
        from nihai import audit_chain
        audit_chain.append({"tur": tur, **kayit})
    except Exception as e:                       # noqa: BLE001 — karar yolu düşmemeli
        logger.debug(f"[zincir] {tur} eklenemedi: {type(e).__name__}: {e}")


def log_signal(signal: Signal, accepted: bool, extra: str = "") -> None:
    """Madde 19 — alınan VE pas geçilen kararlar."""
    action = "SIGNAL" if accepted else "SKIP"
    detail = f"{signal.symbol}: {signal.action.value} @ {signal.price:.2f} | " \
             f"RR={signal.rr_ratio:.2f} Q={signal.quality_score:.2f} | " \
             f"{' / '.join(signal.reasons)}"
    if extra:
        detail += f" || {extra}"
    log_decision(action, detail)
    if accepted:
        _zincire_yaz("sinyal", {
            "symbol": signal.symbol, "action": signal.action.value,
            "price": round(float(signal.price), 4),
            "rr": round(float(signal.rr_ratio), 4),
            "quality": round(float(signal.quality_score), 4),
            "reasons": list(signal.reasons)[:8],
        })


def log_block(reason_code: str, detail: str) -> None:
    """Risk/safety bloklamaları."""
    log_decision(reason_code, detail)


# ---------------------------------------------------------------------------
# KARAR HUNİSİ (Tur-77, Madde 6.4b) — yazıcıyla AYNI modülde: biçim tek kaynaktadır
# (Tur-29: yazan ile ayrıştıran farklı biçim varsayınca pano 330 kaydı sessizce düşürmüştü).
# KARAR-DIŞI rapordur; `scan_symbol`'ün elenme sırası `execution/main.py`'deki guard sırasıdır.
# ---------------------------------------------------------------------------
# Tur-106 (BULGU-106-15): Tur-97/101'in üç quant guard'ı ve Tur-89'un dolum kapısı huniye hiç eklenmemişti →
# günde ~42 bin deneme `taninmayan`a düşüyor, pano onu göstermediği için "taranan deneme" yarıya iniyordu ve
# `FILL_SLIPPAGE`in SKIP satırı STRATEJI/LONG sayılıyordu. `QUANT_ODAK_DISI` sembol başına GÜNDE BİR yazılır
# (sayımı deneme değil sembol); diğerleri her denemede.
HUNI_ASAMALARI: tuple[str, ...] = (
    "KILL_SWITCH_ACTIVE", "SESSION_GUARD", "QUANT_POZISYON_VAR", "QUANT_ODAK_DISI", "QUANT_AYNI_GUN",
    "COOLDOWN", "EVENT_GUARD", "CORP_ACTION", "DATA_EMPTY", "HALT_LULD", "QUOTE_UNHEALTHY", "STRATEJI",
    "STALE_SIGNAL", "FILL_SLIPPAGE", "SHADOW_VETO", "BORROW_BLOCK", "AI_VETO", "CIRCUIT_BREAKER",
    "RISK_REJECT",
)
# Aynı denemede hem `SKIP ... || extra` hem ayrı blok satırı yazan veto/çapa dalları: SKIP satırı
# SAYILMAZ (blok satırı sayılır) — yoksa her veto iki kez elenmiş görünürdü.
_CIFT_YAZILAN = ("STALE_SIGNAL", "FILL_SLIPPAGE", "SHADOW_VETO", "AI_VETO")
_BILGI_SATIRI = {"CHAIN_HEALTH", "IV_RANK", "SHADOW", "SSR_NOTICE", "BORROW_NOTICE", "WASH_SALE",
                 "AI_ADVICE", "GOOD_FAITH", "POSITION_OPEN", "POSITION_CLOSE", "SESSION_CLOSED"}


def karar_hunisi(path: Path | str | None = None) -> dict | None:
    """Bir günün karar logundan ELENME HUNİSİ: her tarama denemesi tam BİR uç durumda biter.

    Birim TARAMA DENEMESİdir (sembol değil — aynı sembol gün içinde her turda yeniden taranır);
    ayrıca her sembolün O GÜN ULAŞTIĞI EN İLERİ aşama `sembol_en_ileri` altında sayılır.
    Kabul = `SIGNAL ... || sizing=` (sanal pozisyon yolu); `SIGNAL_ALERT` bir aşama değil ayrı
    sayaçtır (risk reddinden ÖNCE gönderilebilir). Tanınmayan eylem kodu `taninmayan`a düşer —
    sessizce yutulmaz. Dosya yoksa/okunamazsa None."""
    p = Path(path) if path is not None else _today_log()
    try:
        fh = p.open(encoding="utf-8", errors="replace")
    except OSError:
        return None
    elenen: dict[str, int] = {a: 0 for a in HUNI_ASAMALARI}
    alt: dict[str, dict[str, int]] = {a: {} for a in HUNI_ASAMALARI}
    taninmayan: dict[str, int] = {}
    en_ileri: dict[str, int] = {}
    kabul = uyari = acilis = 0
    sira = {a: i for i, a in enumerate(HUNI_ASAMALARI)}
    kabul_sira = len(HUNI_ASAMALARI)

    def _isle(sym: str, asama: str, neden: str) -> None:
        elenen[asama] += 1
        alt[asama][neden] = alt[asama].get(neden, 0) + 1
        if sym:
            en_ileri[sym] = max(en_ileri.get(sym, -1), sira[asama])

    with fh:
        for ln in fh:
            parcalar = ln.rstrip("\n").split(" | ", 2)
            if len(parcalar) != 3 or not parcalar[0][:4].isdigit():
                continue
            eylem, detay = parcalar[1].strip(), parcalar[2]
            sym, ayrac, govde = detay.partition(": ")
            sym = sym.strip() if ayrac else ""
            if eylem == "CIRCUIT_BREAKER":            # sembolsüz yazılır: log_block(kod, why)
                sym, govde = "", detay
            if eylem in _BILGI_SATIRI:
                acilis += eylem == "POSITION_OPEN"
                continue
            if eylem in ("SIGNAL", "SKIP"):
                ekstra = detay.rpartition(" || ")[2] if " || " in detay else ""
                if eylem == "SIGNAL":
                    if ekstra.startswith("SIGNAL_ALERT"):
                        uyari += 1
                    else:
                        kabul += 1
                        if sym:
                            en_ileri[sym] = max(en_ileri.get(sym, -1), kabul_sira)
                    continue
                if ekstra in _CIFT_YAZILAN:
                    continue
                if ekstra.startswith("RISK_REJECT"):
                    _isle(sym, "RISK_REJECT", ekstra.partition(": ")[2][:40] or "RISK_REJECT")
                elif ekstra in ("NOT_SHORTABLE", "HARD_TO_BORROW"):
                    _isle(sym, "BORROW_BLOCK", ekstra)
                else:
                    yon = govde.split(" @ ", 1)[0].strip() or "?"
                    _isle(sym, "STRATEJI", yon)
                continue
            if eylem in sira:
                neden = govde.split("_IN_")[0].split(":")[0].strip()[:40] or eylem
                _isle(sym, eylem, neden)
            else:
                taninmayan[eylem] = taninmayan.get(eylem, 0) + 1

    toplam = sum(elenen.values()) + kabul
    kalan = toplam
    asamalar = []
    for a in HUNI_ASAMALARI:
        kalan -= elenen[a]
        en_cok = sorted(alt[a].items(), key=lambda kv: -kv[1])[:5]
        asamalar.append({"asama": a, "elenen": elenen[a], "kalan": kalan, "alt_neden": dict(en_cok)})
    ileri_sayim: dict[str, int] = {}
    for idx in en_ileri.values():
        ad = "KABUL" if idx == kabul_sira else HUNI_ASAMALARI[idx]
        ileri_sayim[ad] = ileri_sayim.get(ad, 0) + 1
    return {"dosya": p.name, "toplam_deneme": toplam, "asamalar": asamalar, "kabul": kabul,
            "uyari": uyari, "pozisyon_acilis": acilis, "n_sembol": len(en_ileri),
            "sembol_en_ileri": ileri_sayim, "taninmayan": taninmayan}


# ---------------------------------------------------------------------------
# Madde 38 — Tamamlanan işlem otopsi bloğu
# ---------------------------------------------------------------------------
def log_trade_postmortem(tr: TradeRecord) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append(" " * 18 + "ISLEM TAMAMLANDI RAPORU")
    lines.append("=" * 70)
    lines.append(f"Varlik / Kontrat   : {tr.asset_label}")
    lines.append(f"Uygulanan Strateji : {tr.strategy}")
    if tr.exit_time:
        dur = tr.exit_time - tr.entry_time
        days = dur.days
        hrs, rem = divmod(dur.seconds, 3600)
        mins = rem // 60
        lines.append(f"Pozisyon Suresi    : {days} Gun {hrs} Saat {mins} Dakika")
    lines.append("-" * 70)
    lines.append(f"Giris Zamani       : {tr.entry_time:%Y-%m-%d %H:%M:%S}")
    lines.append(f"Giris Fiyat / Prim : {tr.entry_price:.2f} USD" +
                 (f"   (Slippage: {tr.realized_slippage:+.3f})" if tr.realized_slippage else ""))
    if tr.entry_greeks:
        g = tr.entry_greeks
        lines.append(f"Giris Greekleri    : Delta {g.delta:+.2f} | Theta {g.theta:+.2f} | Vega {g.vega:+.2f}")
    if tr.entry_iv is not None:
        ivr_str = f"{tr.entry_iv_rank:.0f}" if tr.entry_iv_rank is not None else "N/A"
        dte_str = f"{tr.dte_at_entry} gun" if tr.dte_at_entry is not None else "N/A"
        lines.append(f"Giris IV / IVR     : %{tr.entry_iv*100:.0f} / {ivr_str}   | DTE (giris): {dte_str}")
    lines.append(f"Risk (planlanan)   : {tr.risk_dollars:.2f} USD   | Planlanan R:R: {tr.planned_rr:.2f}")
    lines.append("-" * 70)
    if tr.exit_time:
        lines.append(f"Cikis Zamani       : {tr.exit_time:%Y-%m-%d %H:%M:%S}")
        lines.append(f"Cikis Fiyat / Prim : {tr.exit_price:.2f} USD")
        lines.append(f"Cikis Tetikleyici  : {tr.exit_trigger}")
        lines.append("-" * 70)
        lines.append("Performans Metrikleri:")
        if tr.mae_pct is not None and tr.mfe_pct is not None:
            lines.append(f"MAE / MFE          : {tr.mae_pct:+.1f}% / {tr.mfe_pct:+.1f}%")
        if tr.iv_change is not None:
            lines.append(f"IV Degisimi        : {tr.iv_change*100:+.1f}%")
        if tr.realized_pnl is not None:
            lines.append(f"Net PnL Sonucu     : {tr.realized_pnl:+.2f} USD")
        if tr.nlv_after is not None:
            lines.append(f"Guncel Kasa (NLV)  : {tr.nlv_after:,.2f} USD")
    lines.append(f"Giris Gerekcesi    : {' | '.join(tr.entry_reasons)}")
    lines.append("=" * 70 + "\n")

    with _trade_log().open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    _zincire_yaz("kapanis", {
        "asset": tr.asset_label, "strategy": tr.strategy,
        "entry_time": f"{tr.entry_time:%Y-%m-%d %H:%M:%S}",
        "entry_price": round(float(tr.entry_price), 4),
        "exit_time": (f"{tr.exit_time:%Y-%m-%d %H:%M:%S}" if tr.exit_time else None),
        "exit_price": (None if tr.exit_price is None else round(float(tr.exit_price), 4)),
        "exit_trigger": tr.exit_trigger,
        "pnl": (None if tr.realized_pnl is None else round(float(tr.realized_pnl), 2)),
    })


# ---------------------------------------------------------------------------
# Yapılandırılmış işlem CSV'si (data/trades.csv)
# ---------------------------------------------------------------------------
# log_trade_postmortem insan-okur bloklar yazar (parse'ı kırılgan). Bu fonksiyon AYNI
# kapanışı makine-okur tek satır olarak data/trades.csv'ye ekler → post-mortem
# (research/postmortem.py), dashboard ve VaR analizleri tek temiz kaynaktan beslenir.
# pos verilirse symbol/asset_type/side/sector doğrudan ondan alınır (asset_label parse
# edilmez). Best-effort: CSV yazımı asla kapanış akışını bozmaz.
TRADES_CSV = DATA_DIR / "trades.csv"
TRADES_CSV_HEADER = [
    "exit_time", "symbol", "asset_type", "side", "strategy",
    "entry_time", "entry_price", "exit_price", "quantity",
    "risk_dollars", "planned_rr", "realized_pnl", "r_multiple",
    "mae_pct", "mfe_pct", "exit_trigger", "sector", "nlv_after",
    # BULGU-59-4 — pozisyonu AÇAN katman ("odak"/"genis"/""). DUAL_TIER_ENABLED açıkken
    # iki BAĞIMSIZ strateji katmanı bu tek deftere yazıyordu ve katman kolonu olmadığı
    # için katman-başı brüt maruziyet/kaldıraç/expectancy AYRIŞTIRILAMIYORDU — Tur-50 bunu
    # "brüt/özkaynak 5,80× okundu ama okuma GEÇERSİZ" diye açık kalem bırakmıştı.
    *COST_FIELDS, "tier",
]


def _ensure_trades_header() -> None:
    """trades.csv başlığını güncel TRADES_CSV_HEADER'a yükselt (BULGU-59-5).

    KUSUR: `log_trade_csv` yalnız "dosya var mı?" diye soruyordu. Başlık genişlediğinde
    ESKİ başlıklı dosyaya YENİ alan sırasıyla append edilir → 18 sütunluk başlığın altına
    19 değer yazılır ve o satırdan itibaren TÜM kolonlar KAYAR (okuyanlar: pano Günlük
    Defteri, `tearsheet`, `research/postmortem`, VaR). `prediction_tracker._ensure_schema`
    bu sınıfı 2026-07'de kapatmıştı; KARDEŞ defter (Tur-47 kuralı: bir kural yazınca
    `grep` ile aynı şekildeki tabloları ara) açıkta kalmıştı.
    Yükseltme atomiktir: eksik yeni sütunlar '' olur, bilinmeyen eski sütunlar düşer."""
    if not TRADES_CSV.exists() or TRADES_CSV.stat().st_size == 0:
        return
    with TRADES_CSV.open(encoding="utf-8", newline="") as f:
        try:
            header = next(csv.reader(f))
        except StopIteration:
            return
    if header == TRADES_CSV_HEADER:
        return
    with TRADES_CSV.open(encoding="utf-8", newline="") as f:
        old = list(csv.DictReader(f))
    tmp = tmp_path(TRADES_CSV)
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADES_CSV_HEADER, extrasaction="ignore")
        w.writeheader()
        w.writerows(old)
    atomic_replace(tmp, TRADES_CSV)


def log_trade_csv(tr: TradeRecord, pos: VirtualPosition | None = None) -> None:
    """Kapanan işlemi data/trades.csv'ye tek satır olarak yazar (log_trade_postmortem
    ile yan yana çağrılır)."""
    try:
        r_mult = ""
        if tr.realized_pnl is not None and tr.risk_dollars:
            r_mult = f"{tr.realized_pnl / tr.risk_dollars:.3f}"
        symbol = pos.symbol if pos else (tr.asset_label.split() or [""])[0]
        row = {
            "exit_time": tr.exit_time.isoformat() if tr.exit_time else "",
            "symbol": symbol,
            "asset_type": pos.asset_type if pos else "",
            "side": pos.side if pos else "",
            "strategy": tr.strategy,
            "entry_time": tr.entry_time.isoformat() if tr.entry_time else "",
            "entry_price": f"{tr.entry_price:.4f}",
            "exit_price": f"{tr.exit_price:.4f}" if tr.exit_price is not None else "",
            "quantity": tr.contracts_or_shares,
            "risk_dollars": f"{tr.risk_dollars:.2f}",
            "planned_rr": f"{tr.planned_rr:.2f}",
            "realized_pnl": f"{tr.realized_pnl:.2f}" if tr.realized_pnl is not None else "",
            "r_multiple": r_mult,
            "mae_pct": f"{tr.mae_pct:.2f}" if tr.mae_pct is not None else "",
            "mfe_pct": f"{tr.mfe_pct:.2f}" if tr.mfe_pct is not None else "",
            "exit_trigger": tr.exit_trigger or "",
            "sector": getattr(pos, "sector", "") if pos else "",
            "nlv_after": f"{tr.nlv_after:.2f}" if tr.nlv_after is not None else "",
            "tier": getattr(pos, "tier", "") if pos else "",
        }
        costs = tr.cost_attribution or {}
        row.update({key: f"{costs[key]:.6f}" if costs.get(key) is not None else "" for key in COST_FIELDS})
        _ensure_trades_header()
        new = not TRADES_CSV.exists() or TRADES_CSV.stat().st_size == 0
        with TRADES_CSV.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRADES_CSV_HEADER)
            if new:
                w.writeheader()
            w.writerow(row)
    except Exception as e:  # best-effort — kapanışı asla bozma
        logger.debug(f"[trades_csv] yazılamadı: {e}")
