"""
reasoner.py — Yapılandırılmış AKIL YÜRÜTME (FAZ AI). KARAR-DIŞI.

Ne yapar: forward_eval/history.csv, predictions_log.csv ve live_features_log.csv'den UCUZ
"facts" toplar; bunları Türkçe, yapılandırılmış bir DEĞERLENDİRMEYE çevirir:
  - "Durum"   : bugünkü metrikler (verdikt/çözülen/isabet/bekleyen/canlı özellik).
  - "Değişim" : hafızadaki DÜNKÜ digest'e göre delta (çözülen +N, isabet ±pp, verdikt değişimi).
  - "Dikkat"  : operasyonel uyarılar (history'de boşluk, bekleyen birikimi, servis bayatlığı).
  - Kapanış   : "Bu değerlendirme karar vermez ve kenar iddiası içermez (ACI DERS #3/#4)."

Önce TAMAMEN deterministik üretilir. router varsa YALNIZ ifade akıcılaştırılır; DÜRÜSTLÜK
GUARDI: LLM çıktısında deterministik metnin kritik değerleri (çözülen, isabet %, verdikt
stringi) ve kapanış cümlesi birebir geçmiyorsa LLM çıktısı ATILIR (sayı uydurma koruması).

KARAR-DIŞI: hiçbir çıktı sinyal/eşik/META_GATE'e bağlanmaz; "kenar" ima edilmez.
"""
from __future__ import annotations

import csv
import os
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from nihai.ai import router
from nihai.atomic_io import tail_lines
from nihai.config import DATA_DIR

# Modül-düzeyi yollar: testler monkeypatch'ler (daily_summary/prediction_tracker deseni).
HISTORY_CSV = DATA_DIR / "forward_eval" / "history.csv"
PRED_LOG = DATA_DIR / "predictions_log.csv"
FEATS_LOG = DATA_DIR / "live_features_log.csv"

CLOSING = "Bu değerlendirme karar vermez ve kenar iddiası içermez (ACI DERS #3/#4)."
# Bekleyen sayımı: TÜM predictions_log'u parse etmek pahalı (canlıda 2M+ satır). Bekleyenler
# EN YENİ satırlardır (eskiler çözülür) → yalnız son N satır taranır (ucuz + yeterince doğru).
PRED_TAIL_N = int(os.getenv("AI_REASONER_PRED_TAIL", "500"))
HISTORY_GAP_HOURS = float(os.getenv("AI_REASONER_GAP_HOURS", "26"))


# ---------------------------------------------------------------------------
# Savunmacı çeviriciler (CSV alanları string; boş/bozuk/NaN → None)
# ---------------------------------------------------------------------------
def _f(x: object) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if v == v else None


def _i(x: object) -> int | None:
    v = _f(x)
    return int(v) if v is not None else None


# ---------------------------------------------------------------------------
# UCUZ okuyucular (büyük CSV'yi tamamen parse etme)
# ---------------------------------------------------------------------------
def _tail_history(path: Path) -> tuple[dict | None, dict | None]:
    """history.csv (son, önceki) satır; yok/boş/bozuk → (None, None). Küçük dosya (günlük)."""
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None, None
    if not rows:
        return None, None
    return dict(rows[-1]), (dict(rows[-2]) if len(rows) >= 2 else None)


def _count_data_lines(path: Path) -> int | None:
    """Başlık HARİÇ veri satırı sayısı (akışsal newline sayımı; büyük dosyada ucuz).
    Son satırda newline yoksa 1 eksik sayılabilir (kabaca; tam-sayım gerekmez)."""
    try:
        with path.open("rb") as f:
            n = 0
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                n += chunk.count(b"\n")
        return max(0, n - 1)
    except OSError:
        return None


def _read_header(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", newline="") as f:
            first = f.readline().strip()
    except OSError:
        return []
    return next(csv.reader([first])) if first else []


def _tail_lines(path: Path, n: int) -> list[str]:
    """`atomic_io.tail_lines`e ince sarmalayıcı (Tur-35: üç birebir kopya tek kaynağa indi).
    Doğru artımlı sayım deseni zaten 20 satır yukarıdaki `_count_data_lines`'taydı — ders
    MODÜLE değil DESENE aitti. Ad testler/çağıranlar için korundu."""
    return tail_lines(path, n)


def _gecikmis(bar_ts: str, horizon: str, interval: str, simdi: datetime) -> bool:
    """Bekleyen tahmin çözülmüş OLMALI mıydı? Hedef anından (bar + ufuk) sonraki ilk seans dakikası 30 dk'dan önce
    geçtiyse evet. Kapanışa ufuktan az kala verilen tahmin ertesi seansın ilk barını bekler
    (`prediction_tracker._forward_price`) — o yapısal kuyruk birikim değildir. Ölçülemeyen satır (1d, bozuk) → False."""
    try:
        import pandas as pd

        from nihai.config import INTERVAL_MINUTES
        from nihai.data.market_calendar import ilk_bar_ani

        t = pd.Timestamp(bar_ts)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")    # saat dilimsiz bar_ts = UTC (_canon_ts)
        hedef = t + pd.Timedelta(minutes=int(float(horizon)) * INTERVAL_MINUTES[interval])
        return pd.Timestamp(simdi).tz_convert("UTC") - pd.Timedelta(minutes=30) > ilk_bar_ani(hedef)
    except Exception:  # noqa: BLE001 — uyarı ölçüsü; ölçülemeyen satır alarm üretmez
        return False


def _pred_stats(path: Path, tail_n: int) -> dict:
    """predictions_log: toplam satır + SON tail_n içindeki BEKLEYEN (resolved_at boş) sayısı ve bunların
    çözülmüş olması GEREKENLERİ (`_gecikmis`). UNRESOLVABLE/çözülmüş satırlar bekleyen sayılmaz. Dosya yok → None."""
    if not path.exists():
        return {"pred_total": None, "pred_pending_recent": None, "pred_overdue_recent": None, "pred_tail_n": tail_n}
    header = _read_header(path)
    total = _count_data_lines(path)
    idx = header.index("resolved_at") if "resolved_at" in header else None
    ib, ih, ii = (header.index(k) if k in header else -1 for k in ("bar_ts", "horizon_bars", "interval"))
    pend: int | None = None
    gec: int | None = None
    if idx is not None:
        pend = gec = 0
        simdi = datetime.now(UTC)
        for row in csv.reader(_tail_lines(path, tail_n)):
            if len(row) > idx and not row[idx].strip():
                pend += 1
                if min(ib, ih, ii) >= 0 and len(row) > max(ib, ih, ii) and _gecikmis(row[ib], row[ih], row[ii], simdi):
                    gec += 1
    return {"pred_total": total, "pred_pending_recent": pend, "pred_overdue_recent": gec, "pred_tail_n": tail_n}


def seans_esigi(now: datetime | None = None) -> str:
    """Canlı özellik satırlarının sayılacağı alt sınır: en az 30 dk önce açılmış son seansın yerel açılış damgası
    (`market_calendar.son_seans_acilisi_yerel`, BULGU-106-36). Takvim okunamazsa yerel bugünün tarihi (eski davranış)."""
    an = now or datetime.now()
    try:
        from nihai.data.market_calendar import son_seans_acilisi_yerel
        return son_seans_acilisi_yerel(an if an.tzinfo else an.astimezone())   # saat dilimsiz → makinenin yerel saati
    except Exception as e:  # noqa: BLE001 — özet takvim yüzünden düşmez
        logger.debug(f"[ai.reasoner] seans eşiği okunamadı, yerel güne düşüldü: {e}")
        return an.date().isoformat()


def _rows_today(path: Path, esik: str) -> int | None:
    """live_features_log'da logged_at'ı `esik`ten (yerel ISO tarih ya da tarih-saat) ÖNCE OLMAYAN satır sayısı (ilk
    sütun ISO ts; başlık atlanır). Dosya yok → None. Tek akışsal geçiş (csv parse yok)."""
    if not path.exists():
        return None
    try:
        cnt = 0
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # başlık
                if line[:len(esik)] >= esik:
                    cnt += 1
        return cnt
    except OSError:
        return None


# ---------------------------------------------------------------------------
# FACTS
# ---------------------------------------------------------------------------
def gather_facts(now: datetime | None = None) -> dict:
    """Değerlendirme için ucuz facts topla. Dosya yoksa ilgili alan None (çökme yok).

    `live_features_today` adı korunur (digest alanı) ama Tur-106'dan (BULGU-106-36) beri SON SEANSIN satırlarını
    sayar; hangi seans olduğu `live_features_day`dedir (yaz saatinde 23:30 koşusunda bugünün kendisi)."""
    last, prev = _tail_history(HISTORY_CSV)
    pred = _pred_stats(PRED_LOG, PRED_TAIL_N)
    simdi = now or datetime.now()
    today = simdi.date().isoformat()
    esik = seans_esigi(simdi)
    return {
        "today": today,
        "history_last": last,
        "history_prev": prev,
        "history_last_run_at": (last or {}).get("run_at") if last else None,
        "pred_total": pred["pred_total"],
        "pred_pending_recent": pred["pred_pending_recent"],
        "pred_overdue_recent": pred["pred_overdue_recent"],
        "pred_tail_n": pred["pred_tail_n"],
        "live_features_today": _rows_today(FEATS_LOG, esik),
        "live_features_day": esik[:10],
    }


def digest_of(facts: dict) -> dict:
    """facts'in hafızaya yazılacak kompakt özeti (dünkü ile delta için tek kaynak)."""
    last = facts.get("history_last") or {}
    return {
        "today": facts.get("today"),
        "verdict": ((last.get("verdict") or "").strip() or None),
        "resolved": _i(last.get("resolved")),
        "hit_rate": _f(last.get("hit_rate")),
        "pred_total": facts.get("pred_total"),
        "pred_pending_recent": facts.get("pred_pending_recent"),
        "live_features_today": facts.get("live_features_today"),
    }


# ---------------------------------------------------------------------------
# Deterministik metin bölümleri
# ---------------------------------------------------------------------------
def seans_etiketi(veri: dict, *, ek: str = "") -> str:
    """Canlı özellik sayımının ait olduğu seans: yerel bugünse "bugün", değilse "son seans<ek> (GG)"."""
    gun = veri.get("live_features_day")
    return "bugün" if not gun or gun == veri.get("today") else f"son seans{ek} ({gun})"


def _durum_line(facts: dict) -> str:
    last = facts.get("history_last") or {}
    verdict = (last.get("verdict") or "?").strip() or "?"
    parts = [f'verdikt "{verdict}"']
    r = _i(last.get("resolved"))
    if r is not None:
        parts.append(f"çözülen {r}")
    hit = _f(last.get("hit_rate"))
    if hit is not None:
        parts.append(f"isabet %{hit * 100:.1f}")
    pend = facts.get("pred_pending_recent")
    if pend is not None:
        parts.append(f"bekleyen(son {facts.get('pred_tail_n')}) {pend}")
    lf = facts.get("live_features_today")
    if lf is not None:
        parts.append(f"{seans_etiketi(facts)} canlı özellik +{lf}")
    return "Durum: " + " · ".join(parts)


def _degisim_line(facts: dict, prev: dict | None) -> str:
    if not prev:
        return "Değişim: karşılaştırılacak önceki digest yok (ilk koşu)."
    last = facts.get("history_last") or {}
    bits: list[str] = []
    r_now, r_old = _i(last.get("resolved")), _i(prev.get("resolved"))
    if r_now is not None and r_old is not None:
        bits.append(f"çözülen {r_now - r_old:+d}")
    h_now, h_old = _f(last.get("hit_rate")), _f(prev.get("hit_rate"))
    if h_now is not None and h_old is not None:
        bits.append(f"isabet {(h_now - h_old) * 100:+.1f}pp")
    v_now = (last.get("verdict") or "").strip()
    v_old = (prev.get("verdict") or "").strip()
    if v_now and v_old:
        bits.append("verdikt " + (f"aynı ({v_now})" if v_now == v_old else f"{v_old}→{v_now}"))
    if not bits:
        return "Değişim: önceki digest ile karşılaştırılabilir alan yok."
    return "Değişim: " + " · ".join(bits)


def _dikkat_lines(facts: dict, prev: dict | None, now: datetime) -> list[str]:
    warns: list[str] = []
    run_at = facts.get("history_last_run_at")
    if run_at:
        try:
            gap = (now - datetime.fromisoformat(str(run_at))).total_seconds() / 3600.0
            if gap > HISTORY_GAP_HOURS:
                warns.append(f"forward geçmişinde ~{gap:.0f}s boşluk (son koşu {run_at}).")
        except ValueError:
            pass
    # Tur-106 (BULGU-106-18): eskiden "bekleyen dünkünden fazla mı?" soruluyordu; her koşuda kapanış kuyruğu (son
    # 4 dk × ~30 sembol, ertesi seansı bekler) yapısal olarak ~90-120 satır → 60 koşunun 10'unda yanlış alarm.
    gec = facts.get("pred_overdue_recent")
    if isinstance(gec, int) and gec > 0:
        warns.append(f"bekleyen birikimi: çözülmüş olması gereken {gec} tahmin bekliyor "
                     f"(son {facts.get('pred_tail_n')}; çözücü/servis takılmış olabilir).")
    # Tur-106 (BULGU-106-6 → 106-36): sayım yerel bugüne aitken uyarı seans açılmadan (12:00 koşusu, hafta sonu) her
    # gün yanlış alarm veriyordu; "bugünün seansı açıldıysa" kapısı ise gece yarısını aşan kış seansında ve öğle
    # yedeğinde uyarıyı tümden susturuyordu. Sayım artık AÇILMIŞ son seansa ait (`gather_facts`) → 0 her zaman anlamlı.
    # Seansı bilinmeyen sayım (anahtar yok) alarm üretmez.
    if facts.get("live_features_today") == 0 and facts.get("live_features_day"):
        warns.append(f"{seans_etiketi(facts, ek='ta')} canlı özellik logu artmadı (kayıt servisi durmuş olabilir).")
    if not warns:
        return ["Dikkat: belirgin uyarı yok."]
    return ["Dikkat: " + w for w in warns]


def _critical_tokens(facts: dict, prev: dict | None) -> list[str]:
    """LLM'in AYNEN korumak ZORUNDA olduğu değerler (dürüstlük guardı). Spec: çözülen,
    isabet %, verdikt stringi."""
    toks: list[str] = []
    last = facts.get("history_last") or {}
    v = (last.get("verdict") or "").strip()
    if v:
        toks.append(v)
    r = _i(last.get("resolved"))
    if r is not None:
        toks.append(str(r))
    hit = _f(last.get("hit_rate"))
    if hit is not None:
        toks.append(f"%{hit * 100:.1f}")
    return toks


def deterministic_reasoning(facts: dict, prev: dict | None, *,
                            now: datetime | None = None) -> str:
    """TAM deterministik değerlendirme (Durum + Değişim + Dikkat + kapanış). IO/ağ yok."""
    now = now or datetime.now()
    lines = [_durum_line(facts), _degisim_line(facts, prev)]
    lines.extend(_dikkat_lines(facts, prev, now))
    lines.append(CLOSING)
    return "\n".join(lines)


def summary_sections(facts: dict, prev: dict | None, *, now: datetime | None = None) -> str:
    """daily_summary için YALNIZ Değişim + Dikkat (Durum build_summary'de zaten var; kapanış
    da orada var → burada tekrar edilmez). Deterministik; router çağrılmaz."""
    now = now or datetime.now()
    lines = [_degisim_line(facts, prev)]
    lines.extend(_dikkat_lines(facts, prev, now))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM akıcılaştırma + DÜRÜSTLÜK GUARDI
# ---------------------------------------------------------------------------
def _apply_llm(det_text: str, must_tokens: list[str]) -> str:
    """router varsa ifadeyi akıcılaştır; kritik sayı/verdikt VEYA kapanış cümlesi kaybolursa
    LLM çıktısını AT, deterministik metni döndür (sayı uydurma koruması). router None → det."""
    prompt = ("Aşağıdaki Türkçe sistem-değerlendirmesini akıcılaştır. SAYILARI, yüzdeleri ve "
              "verdikt etiketlerini (ör. NO_EDGE) AYNEN koru; yeni iddia/yorum EKLEME. Son "
              "satırdaki kapanış cümlesini AYNEN koru:\n\n" + det_text)
    try:
        out = router.complete("summary", prompt)
    except Exception as e:  # noqa: BLE001 — LLM hiçbir koşulda değerlendirmeyi düşürmesin
        logger.debug(f"[ai.reasoner] router hatası: {e}")
        return det_text
    if not out:
        return det_text
    if CLOSING not in out or any(tok not in out for tok in must_tokens):
        logger.debug("[ai.reasoner] LLM çıktısı sayı/verdikt/kapanış korumadı → deterministik")
        return det_text
    return out


def reason(facts: dict, memory: object) -> str:
    """Değerlendirmeyi üret (deterministik → opsiyonel LLM akıcılaştırma) ve bugünün digest'ini
    hafızaya yaz. `memory` = nihai.ai.memory modülü (veya .last/.append sağlayan nesne)."""
    prev_rec = memory.last("daily_digest")  # type: ignore[attr-defined]
    prev = prev_rec.get("payload") if isinstance(prev_rec, dict) else None
    det = deterministic_reasoning(facts, prev)
    text = _apply_llm(det, _critical_tokens(facts, prev))
    try:
        memory.append("daily_digest", digest_of(facts))  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001 — digest yazımı değerlendirmeyi engellemesin
        logger.debug(f"[ai.reasoner] digest yazılamadı: {e}")
    return text
