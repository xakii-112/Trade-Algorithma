"""
daily_summary.py — Günlük sistem-sağlığı ÖZETİ (FAZ AI-4). Karar VERMEZ, ÖZETLER.

Ne yapar: data/forward_eval/history.csv'nin son iki satırından verdikt + çözülen-delta,
varsa hazır gate_sim/coverage json'ları, hub/kayıtçı servis sağlığı (socket connect) ve
bugünkü canlı özellik-logu satır sayısını 5-8 satırlık Türkçe metne çevirir. Verdikt
etiketleri AYNEN alıntılanır ("NO_EDGE" vb.); kenar İDDİASI eklenmez (ACI DERS #3/#4).

Tasarım: build_summary SAF (dict → str, IO yok) · collect() tüm IO'yu toplar · main() CLI.
prediction_tracker İMPORT EDİLMEZ (execution katmanına bağımlılık yok): gate_sim/coverage
yalnız data/forward_eval/ altındaki HAZIR json'lardan okunur; yoksa "veri yok" denir.

Kullanım:
    python -m nihai.notifier.daily_summary            # stdout + data/daily_summary_last.txt
    python -m nihai.notifier.daily_summary --send     # + Telegram (token yoksa not düşer)
    python -m nihai.notifier.daily_summary --llm      # + LLM akıcılaştırma (HER hatada
                                                      #   deterministik metne düşer)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from loguru import logger

from nihai.config import DATA_DIR

# Modül-düzeyi yollar: testler monkeypatch'ler (prediction_tracker.FEATS_LOG deseni).
HISTORY_CSV = DATA_DIR / "forward_eval" / "history.csv"
GATE_SIM_JSON = DATA_DIR / "forward_eval" / "gate_sim.json"
COVERAGE_JSON = DATA_DIR / "forward_eval" / "coverage.json"
SHADOW_PROMOTE_JSON = DATA_DIR / "forward_eval" / "shadow_promote.json"
FEATS_LOG = DATA_DIR / "live_features_log.csv"
SUMMARY_TXT = DATA_DIR / "daily_summary_last.txt"

# Servis sağlığı: hub (`nihai.data.hub_server`, proje-ac/gözcü) + kayıtçı (web pano/kayıt süreci).
SERVICE_PORTS: dict[str, int] = {
    "hub": int(os.getenv("HUB_PORT", "8799")),
    "kayitci": int(os.getenv("WEB_PORT", "8765")),
}


# ---------------------------------------------------------------------------
# Küçük savunmacı çeviriciler (history.csv alanları string gelir; boş/bozuk → None)
# ---------------------------------------------------------------------------
def _f(x: object) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # NaN → None


def _i(x: object) -> int | None:
    v = _f(x)
    return int(v) if v is not None else None


# Yan rapor bu kadar günden eskiyse özet BAYAT olduğunu söyler (B-1'in devamı: dosyanın
# VARLIĞI tazeliğini garanti etmez; üretici görev ölürse özet eski verdikti taze sanardı).
STALE_DAYS = 2


def _bayatlik(rapor: dict, today: str) -> str:
    """gate_sim/coverage json'unun yaşı için kısa ek: taze/bilinmeyen → "", bayat → " (N gün önce)"."""
    damga = str(rapor.get("generated_at") or "")[:10]
    if not damga:
        return " (yaş bilinmiyor)"
    try:
        yas = (date.fromisoformat(today) - date.fromisoformat(damga)).days
    except ValueError:
        return " (yaş bilinmiyor)"
    return f" (BAYAT: {yas} gün önce)" if yas >= STALE_DAYS else ""


# ---------------------------------------------------------------------------
# Saf çekirdek — collect() çıktısını (veya testte sahte dict'i) metne çevirir
# ---------------------------------------------------------------------------
def build_summary(data: dict) -> str:
    """Toplanmış girdilerden 5-8 satırlık Türkçe özet (SAF; IO/ağ yok).

    Sözleşme: verdikt AYNEN alıntılanır, çözülen delta (+N) gösterilir, servis durumu
    AÇIK/KAPALI yazılır, hiçbir satır kenar İDDİASI taşımaz."""
    lines: list[str] = []
    today = str(data.get("today") or datetime.now().strftime("%Y-%m-%d"))
    lines.append(f"Günlük sistem özeti — {today}")

    svc = data.get("services") or {}
    _d = "AÇIK" if svc.get("hub") else "KAPALI"
    _k = "AÇIK" if svc.get("kayitci") else "KAPALI"
    lines.append(f"Servisler: hub {_d} · kayıtçı {_k}")

    last = data.get("history_last")
    prev = data.get("history_prev")
    if last:
        verdict = str(last.get("verdict") or "?").strip()
        resolved = _i(last.get("resolved"))
        delta = ""
        prev_res = _i(prev.get("resolved")) if prev else None
        if resolved is not None and prev_res is not None:
            delta = f" ({resolved - prev_res:+d})"
        parts = [f'Forward: verdikt "{verdict}"']
        if resolved is not None:
            parts.append(f"çözülen {resolved}{delta}")
        hit = _f(last.get("hit_rate"))
        if hit is not None:
            parts.append(f"isabet %{hit * 100:.1f}")
        ret = _f(last.get("avg_signed_ret_pct"))
        if ret is not None:
            parts.append(f"ort. işaretli getiri %{ret:.3f}")
        lines.append(" | ".join(parts))
        cal = [f"{lbl} {v:.3f}"
               for k, lbl in (("auc_pwin", "AUC(pwin)"), ("brier_pwin", "Brier"), ("ece_pwin", "ECE"))
               if (v := _f(last.get(k))) is not None]
        if cal:
            lines.append("Kalibrasyon: " + " · ".join(cal))
    else:
        lines.append("Forward: veri yok (history.csv bulunamadı/boş).")

    gs = data.get("gate_sim")
    if gs:
        label = str(gs.get("label") or gs.get("verdict") or "?")
        up = gs.get("uplift_pp")
        tail = f" (uplift {up:+.3f}pp)" if isinstance(up, int | float) else ""
        lines.append(f'Meta-kapı canlı sim: "{label}"{tail}{_bayatlik(gs, today)}')
    else:
        lines.append("Meta-kapı canlı sim: veri yok.")

    cov = data.get("coverage")
    if cov:
        tot, res, pen = _i(cov.get("total")), _i(cov.get("resolved")), _i(cov.get("pending"))
        lines.append(f"Kapsama: toplam {tot if tot is not None else '?'} · "
                     f"çözülen {res if res is not None else '?'} · "
                     f"bekleyen {pen if pen is not None else '?'}")

    sp = data.get("shadow_promote")
    if sp:
        yuz = sp.get("surfaces") or {}
        sayim: dict[str, int] = {}
        for s_ in yuz.values():
            lbl = str(s_.get("label") or "?")
            sayim[lbl] = sayim.get(lbl, 0) + 1
        terfi = [str((yuz.get(k) or {}).get("name") or k) for k in (sp.get("promote") or [])]
        ozet = " · ".join(f"{k} {v}" for k, v in sorted(sayim.items()))
        lines.append(f"Gölge terfi ({len(yuz)} yüzey): {ozet}{_bayatlik(sp, today)}")
        # TERFİ EDİLEBİLİR satırı kanıt kapısını GEÇEN yüzeyi adıyla söyler; boşsa
        # "yok" DENİR (sessizce atlanmaz — okunmayan bir kapı kapı değildir, Tur-38).
        lines.append("  → terfiye hazır: " + (", ".join(terfi) if terfi else "YOK (hepsi gölgede)"))
    else:
        lines.append("Gölge terfi: veri yok (research.forward_harness koşmadı).")

    feats = data.get("live_features_today")
    if feats is None:
        lines.append("Canlı özellik logu: dosya yok / bugünkü satır bilgisi yok.")
    else:
        gun = data.get("live_features_day")     # BULGU-106-36: sayım son SEANSA ait (yerel bugün olmayabilir)
        etiket = "bugün" if not gun or gun == today else f"son seans ({gun})"
        lines.append(f"Canlı özellik logu: {etiket} +{int(feats)} satır.")

    lines.append("Not: Bu özet karar vermez ve kenar iddiası içermez (ACI DERS #3/#4).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# IO katmanı
# ---------------------------------------------------------------------------
def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """127.0.0.1:port'a TCP connect denemesi — dinleyen süreç var mı? (0.5s tavan)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _tail_history(path: Path) -> tuple[dict | None, dict | None]:
    """history.csv'nin (son, önceki) satırları; dosya yok/boş/bozuk → zarif (None, None)."""
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None, None
    if not rows:
        return None, None
    last = dict(rows[-1])
    prev = dict(rows[-2]) if len(rows) >= 2 else None
    return last, prev


def _read_json(path: Path) -> dict | None:
    """Varsa hazır json raporu oku (gate_sim/coverage); yok/bozuk → None ('veri yok')."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _live_rows_today(path: Path, esik: str) -> int | None:
    """live_features_log.csv'de logged_at'ı `esik`ten ÖNCE OLMAYAN satır sayısı; dosya yoksa None.

    logged_at = datetime.now().isoformat(timespec='seconds') (prediction_tracker) → yerel, saat dilimsiz ISO;
    `esik` aynı biçimde (tarih ya da tarih-saat) → sözlük sırası zaman sırasıdır (`reasoner.seans_esigi`)."""
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            return sum(1 for r in csv.DictReader(fh)
                       if str(r.get("logged_at") or "")[:len(esik)] >= esik)
    except OSError:
        return None


def collect(*, ports: dict[str, int] | None = None, host: str = "127.0.0.1",
            timeout: float = 0.5) -> dict:
    """Özet girdilerini topla (tüm IO burada; build_summary saf kalır).

    ports inject edilebilir (test: dinleyen sahte soket / kapalı port). prediction_tracker
    import edilmez — gate_sim/coverage yalnız hazır json'lardan (yoksa None)."""
    last, prev = _tail_history(HISTORY_CSV)
    port_map = SERVICE_PORTS if ports is None else ports
    day = date.today().isoformat()
    try:                                           # BULGU-106-36: sayım yerel güne değil son SEANSA ait
        from nihai.ai.reasoner import seans_esigi
        esik = seans_esigi()
    except Exception:  # noqa: BLE001 — AI katmanı yüklenemezse özet yerel güne düşer
        esik = day
    return {
        "today": day,
        "history_last": last,
        "history_prev": prev,
        "gate_sim": _read_json(GATE_SIM_JSON),
        "coverage": _read_json(COVERAGE_JSON),
        "shadow_promote": _read_json(SHADOW_PROMOTE_JSON),
        "services": {name: _port_open(host, p, timeout) for name, p in port_map.items()},
        "live_features_today": _live_rows_today(FEATS_LOG, esik),
        "live_features_day": esik[:10],
    }


# ---------------------------------------------------------------------------
# Opsiyonel LLM akıcılaştırma (--llm) — artık nihai.ai.router üzerinden (FAZ AI):
# sağlayıcı/model seçimi + 429/kota cooldown + anthropic→gemini fallback zinciri router'da.
# HER hatada / anahtar-yoksa deterministik metin (router None döner → text).
# ---------------------------------------------------------------------------
def _llm_polish(text: str) -> str:
    """Deterministik özeti router ile akıcılaştır. Anahtar/kota/ağ/boş-yanıt dahil HER
    durumda girdi metni AYNEN döner — özet asla LLM'e bağımlı değildir. Sağlayıcı
    çözümlemesi ve kota-yönetimi nihai.ai.router'a devredilir (llm_event SDK deseni)."""
    from nihai.ai import router
    prompt = ("Aşağıdaki günlük sistem-sağlığı özetini akıcı, kısa Türkçe metne çevir. "
              "SAYILARI ve verdikt etiketlerini (örn. NO_EDGE, GATE_HURTS) AYNEN koru; "
              "yeni iddia/yorum EKLEME:\n\n" + text)
    try:
        out = router.complete("summary", prompt)
    except Exception:
        return text  # zarif degrade: LLM hiçbir koşulda özeti düşürmesin
    if not out:
        return text
    # DÜRÜSTLÜK GUARDI (reasoner._apply_llm ile aynı ilke): verdikt etiketi, çözülen sayısı,
    # isabet yüzdesi ve kapanış notu LLM çıktısında AYNEN geçmiyorsa çıktı ATILIR (sayı
    # uydurma koruması — özet asla LLM'in hayal gücüne emanet edilmez).
    import re
    toks = [t for grp in re.findall(r'"([A-Z_]+)"|çözülen (\d+)|isabet (%[\d.]+)', text)
            for t in grp if t]
    closing = text.rsplit("\n", 1)[-1]
    if closing not in out or any(tok not in out for tok in toks):
        logger.debug("[daily_summary] LLM çıktısı sayı/verdikt/kapanışı korumadı → deterministik")
        return text
    return out


# ---------------------------------------------------------------------------
# FAZ AI — reasoner "Değişim/Dikkat" bloğu (deterministik) + AI-0 kabul görünürlüğü.
# Hepsi OPSİYONEL ve KARAR-DIŞI: AI katmanı/harness başarısız olsa bile özet gövdesi ve
# "(telegram yapılandırılmamış — gönderim atlandı)" davranışı bozulmaz.
# ---------------------------------------------------------------------------
def _reasoning_block() -> str:
    """reasoner'ın deterministik Değişim/Dikkat bölümleri + bugünün digest'ini hafızaya yaz.
    router ÇAĞRILMAZ (ağ yok). Herhangi bir hata → '' (özet asla düşmez)."""
    try:
        from nihai.ai import memory, reasoner
        facts = reasoner.gather_facts()
        prev_rec = memory.last("daily_digest")
        prev = prev_rec.get("payload") if isinstance(prev_rec, dict) else None
        block = reasoner.summary_sections(facts, prev)
        try:
            memory.append("daily_digest", reasoner.digest_of(facts))
        except Exception as e:
            logger.debug(f"[daily_summary] digest yazılamadı: {e}")
        return block
    except Exception as e:
        logger.debug(f"[daily_summary] akıl yürütme bloğu atlandı: {e}")
        return ""


def _brain_block() -> str:
    """Quant Beyin bölümü (--brain, FAZ AI v2): öğrenme raporu + gölge-karar özeti
    (+ Tur-17: son öz-değerlendirme KARNESİ tek satır). KARAR-DIŞI ve fail-soft: hata → ''
    (özet asla düşmez). Durum satırı ve kapanış cümlesi TEKRAR EDİLMEZ (ikisi de
    build_summary gövdesinde zaten var).

    Karne satırı SALT-OKUNUR gelir (`selfcheck.last_card_line`): günlük özet üç sınavı
    yeniden KOŞMAZ — yalnız en son yazılmış karneyi gösterir. Karne hiç üretilmemişse
    (dosya yok) satır eklenmez; bölümün geri kalanı etkilenmez."""
    try:
        from nihai.ai import decision, learning, selfcheck
        parts = [learning.learning_report(), decision.decision_report(days=7)]
        try:
            parts.append(selfcheck.last_card_line() or "")
        except Exception as e:  # noqa: BLE001 — karne satırı bölümü düşürmesin
            logger.debug(f"[daily_summary] karne satırı atlandı: {e}")
        return "\n".join(p for p in parts if p)
    except Exception as e:
        logger.debug(f"[daily_summary] beyin bloğu atlandı: {e}")
        return ""


def _calibration_block(refresh: bool = False) -> str:
    """Kalibrasyon döngülerinin GECE GÖREVİ + rapor bloğu (Tur-37). KARAR-DIŞI.

    `nihai/ai/calibration_loop.py` (Tur-25) ve `nihai/ai/adaptive_calibration.py` (Tur-27)
    yazıldı, test edildi, artefakt üretti — ve üretimde HİÇBİR ÇAĞIRANI OLMADI. Tazeleme
    yalnız operatör elle CLI koşturunca oluyordu; koşmadığı gün artefakt bayatlıyor ve
    pano/rapor AYLAR ÖNCEKİ verdikti "şimdiki durum" gibi gösteriyordu. `adaptive_calibration`
    modül docstring'i bu kusuru zaten TARİF EDİYOR ("rejim değişince eşleme bayatlar ve kimse
    elle CLI koşana kadar öyle kalır") ama kendi tazeleyicisi yoktu. Bu blok o çağıranı kurar.

    `--calib` VERİLMEDİKÇE yeniden ölçüm YAPILMAZ, yalnız kayıtlı artefakt okunur: yeniden
    fit `predictions_log.csv` kuyruğunu tarar ve prequential döngü on binlerce fit yapar —
    günlük özetin varsayılan yolu bunu ödememelidir (Tur-35 dersi: ölçüm bütçesini OKUYAN
    da öder). Gece görevi `daily_summary --calib` ile koşar.

    KARAR-DIŞI: ne `calibrate()` çağrılır ne bir olasılık düzeltilir — yalnız ölçülür ve
    yazılır. Hata → '' (özet asla düşmez, `_brain_block` deseniyle aynı)."""
    try:
        from nihai.ai import observatory
        if refresh:
            observatory.refresh_calibration()
        from nihai.ai.adaptive_calibration import report as adaptive_report
        from nihai.ai.calibration_loop import calibration_report
        return "\n".join(p for p in (calibration_report(), adaptive_report()) if p)
    except Exception as e:
        logger.debug(f"[daily_summary] kalibrasyon bloğu atlandı: {e}")
        return ""


def _ai0_acceptance_line(path: Path | None = None, *, today: date | None = None) -> str:
    """history.csv'nin son günlerine bakıp 'AI-0 kabul: son 3 günün X'ünde forward koşusu var'
    bilgisini üret (YALNIZ bilgi — karar/kenar değil). Dosya yok/bozuk/boş → '' (özet bozulmaz)."""
    path = path or HISTORY_CSV
    today = today or date.today()
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return ""
    if not rows:
        return ""
    days_seen: set[date] = set()
    for r in rows:
        try:
            days_seen.add(datetime.fromisoformat(str(r.get("run_at") or "")).date())
        except ValueError:
            continue
    window = {today - timedelta(days=k) for k in range(3)}   # bugün + son 2 gün = son 3 gün
    hit = len(window & days_seen)
    return f"AI-0 kabul: son 3 günün {hit}'ünde forward koşusu var (bilgi)."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Özeti üret → stdout + DATA_DIR/daily_summary_last.txt. --send Telegram, --llm LLM."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass  # capsys/pipe: reconfigure yoksa da çalış
    ap = argparse.ArgumentParser(description="Günlük sistem-sağlığı özeti (karar vermez).")
    ap.add_argument("--send", action="store_true", help="özeti Telegram'a da gönder")
    ap.add_argument("--llm", action="store_true", help="metni LLM ile akıcılaştır (opsiyonel)")
    ap.add_argument("--brain", action="store_true",
                    help="Quant Beyin bölümü ekle (öğrenme + gölge kararlar; karar-dışı)")
    ap.add_argument("--calib", action="store_true",
                    help="kalibrasyon döngülerini YENİDEN ÖLÇ + raporla (gece görevi; "
                         "karar-dışı). Verilmezse yalnız kayıtlı artefakt okunur.")
    args = ap.parse_args(argv)

    text = build_summary(collect())
    # FAZ AI: Değişim/Dikkat bloğu + AI-0 kabul satırı (+ opsiyonel Beyin bölümü) kapanış
    # "Not:" satırının ÖNÜNE girer. Hepsi karar-DIŞI ve fail-soft: hata → blok atlanır.
    extras = [b for b in (_reasoning_block(), _ai0_acceptance_line(),
                          _calibration_block(refresh=args.calib)) if b]
    if args.brain:
        bb = _brain_block()
        if bb:
            extras.append(bb)
    if extras:
        lines = text.split("\n")
        lines[-1:-1] = extras
        text = "\n".join(lines)
    if args.llm:
        text = _llm_polish(text)

    print(text)
    try:
        SUMMARY_TXT.write_text(text + "\n", encoding="utf-8")
    except OSError as e:
        print(f"(uyarı) özet dosyaya yazılamadı: {e}")

    if args.send:
        from nihai.notifier import telegram_bot
        if telegram_bot.is_enabled():
            telegram_bot.send(text)
            print("(Telegram'a gönderildi)")
        else:
            print("(telegram yapılandırılmamış — gönderim atlandı)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
