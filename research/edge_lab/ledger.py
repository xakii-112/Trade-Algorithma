"""
ledger.py — Kenar-keşif programının KALICI deney defteri. KARAR-DIŞI/ARAŞTIRMA.

`BACKLOG.md` şu kuralı YAZIYOR ama uygulamasını İNSANA bırakıyordu:

    "**aileler arası** çoklu-test bu sayaçla insan tarafından izlenir. Sayaç büyüdükçe
     tek bir PASS'in kanıt değeri DÜŞER; 5-6 aileden sonra gelen ilk PASS'e, birinciye
     gösterilenden çok daha fazla şüpheyle bakılmalıdır."

`harness.run_candidate` ise DSR'yi HER koşuda sabit `n_trials=4` ile (yalnız kendi eşik
taraması) deflate ediyordu: adayın programda KAÇINCI olduğu kanıt eşiğine hiç etki
etmiyordu. Yani 1. adayla 10. adaya AYNI çıta uygulanıyor, "denedikçe şansla PASS alma
olasılığı artar" gerçeği yalnız markdown tablosunda yaşıyordu. Elle tutulan sayaç ayrıca
unutulabilir ve geriye dönük düzenlenebilir — kanıt kapısının en zayıf halkası oydu.

Bu modül o sayacı MAKİNEYE çevirir (Tur-24/27/31 deseni: elle uygulanan kuralı ADAYI
OLMAYAN bir turda koda taşı — aday FAIL ettikten sonra kural değiştirmek, sonuca göre
kural değiştirmektir). Defter yalnız-ekleme JSONL'dir; `run_candidate` her koşuda bir
satır ekler ve bir SONRAKİ koşunun DSR deflasyonu bu defterden okunur:

    n_trials = (adayın kendi eşik taraması) x (defterdeki AYRIK aday adı sayısı, bu koşu dahil)

BOŞ defterde bu, eski sabitin ta kendisidir (1 aday x 4 eşik = 4) → bugüne kadarki üç
ailenin koşusuyla BİT-DENK başlar; yalnız program aradıkça sıkışır. Kanıt kapısı bu yolla asla
GEVŞEYEMEZ (ACI DERSLER: sıkılaştırma yeni sonuç ÜRETEMEZ, yanıltıcıyı eler).

Sayım birimi AYRIK ADAY ADI'dır, koşu sayısı değil: aynı adayı hata düzeltmek için üç kez
koşmak deflasyonu şişirmemeli (geliştirme döngüsü kanıt eşiğini kaçırmamalı), ama defter
her koşuyu yine de kaydeder (denetim izi). TAVAN: ad seçimi araştırmacının elindedir —
aynı adı FARKLI özellik kümesiyle yeniden kullanmak sayacı atlatır; `features` ve
`dataset` her satırda saklanır ki denetim bunu yakalayabilsin (otomatik tespit YOK).

Kullanım:
    python -m research.edge_lab      # programın güncel durumu (elle sayaç yerine)
"""
from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LEDGER_FILENAME = "kosu_defteri.jsonl"
# Defter `data/` altında DEĞİL, paketin YANINDA durur ve SÜRÜM KONTROLÜNDEDİR — çünkü
# `/data/` .gitignore'dadır: orada tutulsaydı taze bir klon (ya da silinen bir data dizini)
# çoklu-test sayacını SIFIRLAR ve kanıt kapısı sessizce GEVŞERDİ. Tam da bu modülün
# engellemek için var olduğu şey. Yeri `BACKLOG.md` ve `SONUC_*.md` ile aynı: kurumsal kayıt.
_DEFAULT_PATH = Path(__file__).resolve().parent / LEDGER_FILENAME
_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def ledger_path(path: str | Path | None = None) -> Path:
    """Defter yolu: açık argüman > `NIHAI_EDGE_LEDGER` env > paket içi `kosu_defteri.jsonl`.

    Env import ANINDA değil ÇAĞRI anında okunur: testler her test için ayrı defter
    işaretleyebilsin (kök `conftest.py` autouse fixture). Aksi halde tek bir pytest
    koşusunda biriken adaylar SONRAKİ testin DSR eşiğini değiştirir ve kapı sıra-bağımlı
    olur — kanıt kapısında sıra-bağımlılık, kapının kendisinin bozulmasıdır."""
    if path is not None:
        return Path(path)
    env = os.getenv("NIHAI_EDGE_LEDGER")
    return Path(env) if env else _DEFAULT_PATH


def read_entries(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Defterdeki tüm satırlar. Bozuk/yarım satır ATLANIR (`registry._read_ab_rows` deseni):
    yalnız-ekleme günlüğünde çökme en kötü ihtimalle SON satırı yarım bırakır ve bir
    araştırma defteri okunamadığı için koşu düşmemeli."""
    p = ledger_path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def distinct_candidates(path: str | Path | None = None) -> list[str]:
    """Defterde adı geçen AYRIK adaylar (ilk görülme sırasında)."""
    seen: list[str] = []
    for row in read_entries(path):
        name = str(row.get("candidate") or "")
        if name and name not in seen:
            seen.append(name)
    return seen


def n_trials_for(name: str, per_candidate_trials: int,
                 path: str | Path | None = None) -> int:
    """Bu koşuda kullanılacak PROGRAM-DÜZEYİ çoklu-test sayısı (DSR deflasyonu için).

    `per_candidate_trials` adayın KENDİ içindeki tarama sayısıdır (harness'taki eşik
    adaylarının sayısı) ve çarpan olarak GEÇİLİR — böylece bu modül harness'a bağımlı
    olmaz (döngüsel import yok). Boş defterde dönüş = `per_candidate_trials` (eski sabit)."""
    names = set(distinct_candidates(path))
    names.add(str(name))
    return max(1, int(per_candidate_trials)) * len(names)


def prior_runs(name: str, path: str | Path | None = None) -> dict[str, Any]:
    """Bu adayın GEÇMİŞİ: kaç kez koşuldu, kaçı FAIL, en son ne zaman.

    ACI-7 dört kez tekrarlandı; "bu aileyi zaten çürütmüştük" bilgisi verdiktin YANINDA
    durmalı ki aynı aile yeni veri/gerekçe olmadan yeniden keşfedilmesin."""
    rows = [r for r in read_entries(path) if str(r.get("candidate") or "") == str(name)]
    fails = sum(1 for r in rows if not r.get("all_passed"))
    return {
        "n": len(rows),
        "fail": fails,
        "pass": len(rows) - fails,
        "last": rows[-1].get("ts") if rows else None,
    }


def record(verdict: dict[str, Any], *, feature_cols: list[str] | None = None,
           note: str | None = None, path: str | Path | None = None) -> dict[str, Any]:
    """Verdikti deftere EKLE (tek satır, O(satır) — `registry._append_line` deseni).

    Yazım fail-soft'tur: defter yazılamıyorsa (salt-okunur disk, kilitli dosya) koşu
    DÜŞMEZ, yalnız o satır kaydedilmez. Dönen satır her hâlükârda çağırana verilir.

    `note`: satırın KÖKENİ (örn. geriye dönük doldurulan tarihsel koşunun kaynak belgesi).
    Ölçülmüş bir koşu ile belgeden aktarılmış bir kayıt defterde ayırt edilebilmelidir."""
    gates = verdict.get("gates") or []
    dsr_metrics: dict[str, Any] = {}
    kapsam_metrics: dict[str, Any] = {}
    for g in gates:
        if g.get("name") == "dsr":
            dsr_metrics = g.get("metrics") or {}
        elif g.get("name") == "veri_kapsami":
            kapsam_metrics = g.get("metrics") or {}
    row: dict[str, Any] = {
        "ts": _utc_now_iso(),
        "candidate": str(verdict.get("candidate") or ""),
        "dataset": str(verdict.get("dataset") or ""),
        "features": list(feature_cols or []),
        "n_rows": int(verdict.get("n_rows") or 0),
        "all_passed": bool(gates) and all(bool(g.get("passed")) for g in gates),
        "failed_gates": [str(g.get("name")) for g in gates if not g.get("passed")],
        "dsr": verdict.get("dsr"),
        "n_trials": dsr_metrics.get("n_trials"),
        "exec_lag_effect": verdict.get("exec_lag_effect"),
        "generated_at": verdict.get("generated_at"),
        # Tur-63/BULGU-63-2: defter satır SAYISINI tutuyordu ama DÖNEMİ tutmuyordu —
        # "bu aileyi 2021-2023 için de çürütmüş müydük?" sorusu makineden YANITLANAMIYORDU.
        # Geçmiş satırlar bu alanı taşımaz (defter yalnız-eklemedir, geriye doldurulmaz).
        "kapsam": {k: kapsam_metrics.get(k) for k in
                   ("data_start", "data_end", "expected_start", "expected_end",
                    "warmup_days", "head_trunc_days", "tail_trunc_days", "expected_declared")
                   } if kapsam_metrics else None,
    }
    if note:
        row["not"] = str(note)
    p = ledger_path(path)
    try:
        with _LOCK:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
    except OSError:
        pass
    return row


def summary(path: str | Path | None = None) -> str:
    """Programın güncel durumu — `BACKLOG.md`'deki elle tutulan sayacın makine karşılığı."""
    rows = read_entries(path)
    if not rows:
        return (f"Defter BOŞ ({ledger_path(path)}) — henüz aday koşulmadı.\n"
                "Bir sonraki adayın DSR deflasyonu: 1 aday x eşik taraması (eski sabit).")
    names = distinct_candidates(path)
    passed = [r for r in rows if r.get("all_passed")]
    lines = [
        f"Defter: {ledger_path(path)}",
        f"Koşu: {len(rows)} | AYRIK aday: {len(names)} | tüm kapıları geçen koşu: {len(passed)}",
        "",
        f"{'aday':<28} {'kosu':>5} {'FAIL':>5} {'son':<10} {'DSR':>8} {'n_trials':>9}",
    ]
    for name in names:
        mine = [r for r in rows if str(r.get("candidate") or "") == name]
        son = mine[-1]
        fails = sum(1 for r in mine if not r.get("all_passed"))
        dsr_v = son.get("dsr")
        dsr_s = "-" if dsr_v is None else f"{float(dsr_v):.4f}"
        lines.append(f"{name[:28]:<28} {len(mine):>5} {fails:>5} "
                     f"{('TUM-KAPI' if son.get('all_passed') else 'FAIL'):<10} "
                     f"{dsr_s:>8} {str(son.get('n_trials') or '-'):>9}")
    lines += [
        "",
        f"Bir sonraki YENİ adayın çoklu-test sayacı: {len(names) + 1} ayrık aday "
        "(DSR eşiği bu kadarla deflate edilir).",
        "Sayaç büyüdükçe tek bir PASS'in kanıt değeri DÜŞER (BACKLOG.md çoklu-test muhasebesi).",
    ]
    return "\n".join(lines)
