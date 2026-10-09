"""
test_tur52_insider_veri_seti.py — BACKLOG #5 (Form 4 açık-piyasa alım) kilitleri.

Bu ailenin dört ayrı yoldan sessizce "kenar" uydurması mümkündü; dördü de burada çapalanır:

* **feature-lag 1 (ACI DERS #7).** Filing günü akşam yayımlanmış olabilir. Giriş barı
  filing gününün KENDİSİ olursa, ölçülen şey sinyal değil o günün kendi hareketidir.
* **Yalnız kod "P" + edinim "A", ve `4/A` HARİÇ.** Hibe/icra/satış bir kanaat beyanı
  değildir; düzeltme ise aynı işlemi ikinci kez bildirir (çift sayım).
* **Örtüşmesizlik.** Aynı sembolde `horizon` bardan yakın olaylar aynı ileri pencereyi
  paylaşır; sayılırsa t-istatistiği şişer (Tur-27: bağımsızlık birimi gözlem değildir).
* **BETA kontrolü (§6).** Ham ileri getiri bir kenar iddiası DEĞİLDİR — boğa piyasasında
  her olay ailesi pozitif görünür. Fazlalık kolonu ŞEMADA olmalı ki unutulamasın.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from research.edge_lab import build_insider_dataset as B

TAB = "\t"
NL = "\n"


def _px(n: int = 400, start: str = "2021-01-04") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    return pd.DataFrame({"date": idx, "close": close, "volume": 1_000_000.0})


def _ev(dates: list[str], kume: int = 1, deger: float = 500_000.0) -> pd.DataFrame:
    return pd.DataFrame({"buy_value": [deger] * len(dates),
                         "n_filings": [kume] * len(dates)},
                        index=pd.to_datetime(dates)).sort_index()


def test_giris_bari_filing_gununden_SONRAKI_bardir():
    """feature-lag 1: filing günü fiyatına girilmez."""
    px = _px()
    filing = px["date"].iloc[100]
    rows = B._rows_for("TEST", _ev([str(filing.date())]), px, horizon=21, cost_pct=0.0)
    assert len(rows) == 1
    assert rows[0]["ts"] == px["date"].iloc[101], "giriş filing gününe düşmüş (ACI-7)"


def test_tatil_gununde_filing_SONRAKI_islem_gunune_kayar():
    px = _px()
    i = int(np.flatnonzero(px["date"].dt.dayofweek.to_numpy() == 4)[20])   # bir CUMA
    cumartesi = px["date"].iloc[i] + pd.Timedelta(days=1)
    rows = B._rows_for("TEST", _ev([str(cumartesi.date())]), px, horizon=21, cost_pct=0.0)
    assert rows and rows[0]["ts"] == px["date"].iloc[i + 1], "pazartesiye kaymadı"


def test_ortusen_olaylar_ELENIR():
    """`horizon` bardan yakın ikinci olay aynı ileri pencereyi paylaşır → sayılmaz."""
    px = _px()
    d1, d2, d3 = (str(px["date"].iloc[i].date()) for i in (100, 105, 150))
    rows = B._rows_for("TEST", _ev([d1, d2, d3]), px, horizon=21, cost_pct=0.0)
    assert len(rows) == 2, "örtüşen olay sayıldı (t şişer)"
    assert rows[0]["ts"] == px["date"].iloc[101]
    assert rows[1]["ts"] == px["date"].iloc[151]


def test_ileri_getiri_yalnizca_GELECEGE_bakar():
    """`fwd_ret`, giriş barından SONRAKİ horizon barı ölçmeli — öncekiler etkilememeli."""
    px = _px()
    d = str(px["date"].iloc[100].date())
    rows = B._rows_for("TEST", _ev([d]), px, horizon=21, cost_pct=0.0)
    assert np.isclose(rows[0]["fwd_ret"],
                      px["close"].iloc[101 + 21] / px["close"].iloc[101] - 1.0)
    bozuk = px.copy()
    bozuk.loc[:99, "close"] *= 3.0            # giriş barından ÖNCEYİ boz
    rows2 = B._rows_for("TEST", _ev([d]), bozuk, horizon=21, cost_pct=0.0)
    assert np.isclose(rows[0]["fwd_ret"], rows2[0]["fwd_ret"])


def test_maliyet_pnl_den_DUSULUR_ve_etiket_pnl_den_turer():
    px = _px()
    d = str(px["date"].iloc[100].date())
    ham = B._rows_for("TEST", _ev([d]), px, horizon=21, cost_pct=0.0)[0]
    net = B._rows_for("TEST", _ev([d]), px, horizon=21, cost_pct=0.1)[0]
    assert np.isclose(net["pnl"], ham["fwd_ret"] - 0.001)
    assert net["label"] == int(net["pnl"] > 0)


def _zip_bytes() -> bytes:
    sub = TAB.join(["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE",
                    "ISSUERTRADINGSYMBOL"]) + NL
    sub += TAB.join(["a", "01-MAR-2021", "4", "AAA"]) + NL
    sub += TAB.join(["b", "02-MAR-2021", "4", "AAA"]) + NL
    sub += TAB.join(["c", "03-MAR-2021", "4/A", "AAA"]) + NL        # düzeltme → HARİÇ
    trx = TAB.join(["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES",
                    "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"]) + NL
    trx += TAB.join(["a", "P", "100", "10", "A"]) + NL              # SAYILIR: 1000 $
    trx += TAB.join(["a", "S", "900", "10", "D"]) + NL              # satış
    trx += TAB.join(["b", "A", "500", "10", "A"]) + NL              # hibe
    trx += TAB.join(["b", "M", "500", "10", "A"]) + NL              # opsiyon icrası
    trx += TAB.join(["c", "P", "999", "10", "A"]) + NL              # 4/A içindeki alım
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", sub)
        z.writestr("NONDERIV_TRANS.tsv", trx)
    return buf.getvalue()


def test_yalnizca_kod_P_ve_edinim_A_sayilir(tmp_path, monkeypatch):
    """Hibe/icra/satış/düzeltme bir açık-piyasa alımı DEĞİLDİR."""
    monkeypatch.setattr(B, "CACHE_DIR", tmp_path)
    payload = _zip_bytes()

    class _Yanit:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(B.urllib.request, "urlopen", lambda *a, **k: _Yanit())
    out = B._fetch_quarter("2021q1")
    assert len(out) == 1, "yalnız 'P'+'A' içeren, 4/A OLMAYAN gün kalmalı"
    assert float(out["buy_value"].iloc[0]) == 1000.0
    assert int(out["n_filings"].iloc[0]) == 1


def test_yayimlanmamis_ceyrek_SESSIZCE_atlanmaz(monkeypatch):
    """Eksik çeyrek kapsamı kısaltır — Tur-51'in 20 aylık deliğinin aynı sınıfı."""
    cagrilan: list[str] = []

    def _sahte(q: str) -> pd.DataFrame:
        cagrilan.append(q)
        if q == "2021q2":
            raise OSError("HTTP 404")
        return pd.DataFrame({"symbol": ["AAA"],
                             "filing_date": [pd.Timestamp("2021-01-05")],
                             "buy_value": [1.0], "n_filings": [1]})

    monkeypatch.setattr(B, "_fetch_quarter", _sahte)
    out = B.load_buys("2021-01-01", "2021-06-30")
    assert len(cagrilan) == 2 and len(out) == 1
    # Log bir sözleşme DEĞİLDİR — eksik kapsam makine-okunur taşınmalı.
    assert out.attrs["eksik_ceyrekler"] == ["2021q2"]


def test_BETA_kontrolu_semada_ZORUNLU():
    """Ham ileri getiri bir kenar iddiası değildir (§6) — fazlalık kolonu şemada olmalı."""
    assert "mkt_ret" in B.SCHEMA_COLS and "excess" in B.SCHEMA_COLS
