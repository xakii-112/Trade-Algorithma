"""Tur-77: lot maliyeti, vade, eksik edinim ve pozisyon uyumluluğu."""
import csv
import sys

import pytest

from nihai import tax_report as tr

YONTEMLER = ("position", "fifo", "hifo")
YENI_ALANLAR = {"lot_method", "n_overlap_dispositions", "unmatched",
               "n_option_position_fallback", "option_position_fallback_reason"}


def _islem(giris, cikis, qty=10, ep=100, xp=110, **ek):
    return {"symbol": "A", "asset_type": "stock", "side": "LONG",
            "entry_time": giris, "exit_time": cikis, "quantity": qty,
            "entry_price": ep, "exit_price": xp, "realized_pnl": (xp - ep) * qty, **ek}


def _csv(tmp_path, satirlar):
    p = tmp_path / "trades.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_islem("", "")))
        w.writeheader()
        w.writerows(satirlar)
    return p


def test_kapali_form_ve_toplam_degismezi(tmp_path):
    p = _csv(tmp_path, [_islem("2024-01-02", "2024-03-01"),
                        _islem("2024-02-01", "2024-04-01", ep=120, xp=130)])
    raporlar = {m: tr.build(p, lot_method=m) for m in YONTEMLER}
    for m in ("position", "fifo"):
        assert [r["gain_loss"] for r in raporlar[m]["rows"]] == [100, 100]
        assert [r["date_acquired"] for r in raporlar[m]["rows"]] == ["2024-01-02", "2024-02-01"]
    assert [r["gain_loss"] for r in raporlar["hifo"]["rows"]] == [-100, 300]
    assert [r["date_acquired"] for r in raporlar["hifo"]["rows"]] == ["2024-02-01", "2024-01-02"]
    for r in raporlar.values():
        assert r["total"] == r["short_term_total"] == 200
        assert r["long_term_total"] == 0
        assert r["n_overlap_dispositions"] == 1
        assert r["unmatched"] == []


def test_kismi_lot_iki_satira_bolunur(tmp_path):
    # Tek quantity kolonu iki olayda da kullanılır; açık edinimler ve ayrı satış
    # ile 15 + 5 edinim, 10 satış defteri kurulur. Eksik olay uydurulmaz.
    p = _csv(tmp_path, [_islem("2024-01-01", "", 15, 10, 15),
                        _islem("2024-02-01", "", 5, 20, 15),
                        _islem("", "2024-03-01", 10, 0, 15)])
    r = tr.build(p, lot_method="hifo")
    assert [x["description"] for x in r["rows"]] == ["5 sh. A", "5 sh. A"]
    assert [x["cost_basis"] for x in r["rows"]] == [100, 50]
    assert [x["proceeds"] for x in r["rows"]] == [75, 75]
    assert [x["gain_loss"] for x in r["rows"]] == [-25, 25]
    assert r["n"] == 2 and r["total"] == 0 and r["unmatched"] == []


@pytest.mark.parametrize("method", YONTEMLER)
@pytest.mark.parametrize("cikis,gun,part", [("2024-01-01", 365, "I (kısa vade)"),
                                           ("2024-01-02", 366, "II (uzun vade)")])
def test_vade_siniri(tmp_path, method, cikis, gun, part):
    p = _csv(tmp_path, [_islem("2023-01-01", cikis)])
    s = tr.build(p, lot_method=method)["rows"][0]
    assert s["holding_days"] == gun and s["part"] == part


def test_hifo_eslesen_lotun_vadesi_ve_kalan_miktari(tmp_path):
    p = _csv(tmp_path, [_islem("2023-01-01", "2024-01-02", 10, 10, 15),
                        _islem("2023-12-01", "2024-02-01", 5, 20, 30)])
    hifo = tr.build(p, lot_method="hifo")
    assert [(r["date_acquired"], r["holding_days"], r["gain_loss"])
            for r in hifo["rows"]] == [("2023-12-01", 32, -25), ("2023-01-01", 366, 25),
                                      ("2023-01-01", 396, 100)]
    assert hifo["short_term_total"] == -25 and hifo["long_term_total"] == 125
    fifo = tr.build(p, lot_method="fifo")
    assert fifo["short_term_total"] == 50 and fifo["long_term_total"] == 50
    assert hifo["total"] == fifo["total"] == 100
    assert hifo["unmatched"] == []


def test_position_bit_denklik_ve_sabit_beklenen_sozluk(tmp_path):
    p = _csv(tmp_path, [_islem("2024-01-02", "2024-03-01")])
    varsayilan = tr.build(p)
    acik = tr.build(p, lot_method="position")
    assert varsayilan == acik
    eski = {k: v for k, v in varsayilan.items() if k not in YENI_ALANLAR}
    assert eski == {
        "rows": [{"description": "10 sh. A", "date_acquired": "2024-01-02",
                  "date_sold": "2024-03-01", "proceeds": 1100.0, "cost_basis": 1000.0,
                  "adjustment_code": "", "adjustment_amount": 0.0, "gain_loss": 100.0,
                  "part": "I (kısa vade)", "symbol": "A", "side": "LONG", "holding_days": 59,
                  "virtual": True, "wash_sale_candidate": False}],
        "section_1256": [], "n": 1, "n_1256": 0, "n_wash_candidates": 0,
        "short_term_total": 100.0, "long_term_total": 0.0, "total": 100.0,
        "total_1256": 0, "reason": None,
    }


@pytest.mark.parametrize("side", ["SHORT", "SELL", "-1"])
def test_short_etkilenmez(tmp_path, side):
    p = _csv(tmp_path, [_islem("2021-01-01", "2024-03-01", ep=200, xp=180, side=side),
                        _islem("2021-02-01", "2024-04-01", ep=220, xp=210, side=side)])
    eski = tr.build(p)
    for method in ("fifo", "hifo"):
        r = tr.build(p, lot_method=method)
        assert r["rows"] == eski["rows"]
        assert r["short_term_total"] == 300 and r["long_term_total"] == 0
        assert r["n_overlap_dispositions"] == 0 and r["unmatched"] == []


def test_opsiyon_kontratlari_birlesmez_1256_kovasi_korunur(tmp_path):
    satirlar = [_islem("2024-01-01", "2024-03-01", 1, 5, 7, asset_type="option", symbol=sym)
                for sym in ("SPX", "A")]
    satirlar += [_islem("2024-02-01", "2024-04-01", 1, 10, 12, asset_type="option", symbol=sym)
                 for sym in ("SPX", "A")]
    p = _csv(tmp_path, satirlar + [_islem("2024-01-01", "2024-03-01", 1, 100, 110)])
    eski = tr.build(p)
    for method in ("fifo", "hifo"):
        r = tr.build(p, lot_method=method)
        assert r["section_1256"] == eski["section_1256"]
        assert r["rows"] == eski["rows"]
        assert r["total_1256"] == 400 and r["total"] == 410
        assert r["n_option_position_fallback"] == 4
        assert r["option_position_fallback_reason"] == "KONTRAT_KIMLIGI_YOK"
        assert r["n_overlap_dispositions"] == 0
    assert eski["n_option_position_fallback"] == 0


@pytest.mark.parametrize("method", ["fifo", "hifo"])
def test_fazla_satis_maliyeti_uydurulmaz(tmp_path, method):
    # Yalnız-satış satırı (giriş boş): 10 satış, açıkta 5 → 5 eşlenemez, maliyet uydurulmaz.
    p = _csv(tmp_path, [_islem("2024-01-01", "", 5, 10, 15),
                        _islem("", "2024-03-01", 10, 20, 15)])
    r = tr.build(p, lot_method=method)
    assert r["n"] == 1 and r["rows"][0]["cost_basis"] == 50
    assert r["rows"][0]["proceeds"] == 75 and r["total"] == 25
    assert r["unmatched"] == [{"symbol": "A", "time": "2024-03-01T00:00:00", "quantity": 5}]


@pytest.mark.parametrize("method", ["fifo", "hifo"])
def test_cikisi_girisinden_once_satir_hayalet_lot_birakmaz(tmp_path, method):
    """Tur-77 denetimi: çıkışı girişinden ÖNCE olan tutarsız satır olay akışına girseydi girişi
    kapanmayan hayalet lot olur ve SONRAKİ satışa yanlış maliyet verirdi (burada 99 $)."""
    p = _csv(tmp_path, [_islem("2024-04-01", "2024-03-01", 10, 99, 15),     # tutarsız satır
                        _islem("2024-05-01", "2024-06-01", 10, 10, 15)])
    r = tr.build(p, lot_method=method)
    assert [s["cost_basis"] for s in r["rows"]] == [100]                  # 99 $'lık hayalet lot YOK
    assert r["unmatched"] == [{"symbol": "A", "time": "2024-03-01T00:00:00", "quantity": 10.0,
                               "reason": "CIKIS_GIRISTEN_ONCE"}]
    assert r["n_overlap_dispositions"] == 0
    assert tr.build(p)["unmatched"] == []                                 # pozisyon kipi eskisi gibi


def test_eksik_maliyet_sifir_kabul_edilmez(tmp_path):
    s = _islem("2024-01-01", "2024-02-01")
    s["entry_price"] = ""
    r = tr.build(_csv(tmp_path, [s]), lot_method="hifo")
    assert r["rows"] == [] and r["total"] == 0
    assert r["unmatched"][0]["quantity"] == 10


def test_hifo_esit_maliyette_en_eski_lot_ve_determinizm(tmp_path):
    p = _csv(tmp_path, [_islem("2024-02-01", "2024-03-01"),
                        _islem("2024-01-01", "2024-04-01")])
    r = tr.build(p, lot_method="hifo")
    assert r == tr.build(p, lot_method="hifo")
    assert [s["date_acquired"] for s in r["rows"]] == ["2024-01-01", "2024-02-01"]
    assert [s["date_sold"] for s in r["rows"]] == ["2024-03-01", "2024-04-01"]


def test_esit_zamanda_edinimler_once_utc_hizasi(tmp_path):
    p = _csv(tmp_path, [_islem("2024-01-01", "2024-02-01T12:00:00Z", ep=10, xp=15),
                        _islem("2024-02-01T15:00:00+03:00", "2024-03-01", ep=20, xp=30)])
    r = tr.build(p, lot_method="hifo")
    assert [s["gain_loss"] for s in r["rows"]] == [-50, 200]
    assert r["rows"][0]["holding_days"] == 0
    assert r["n_overlap_dispositions"] == 1 and r["unmatched"] == []


@pytest.mark.parametrize("method", YONTEMLER)
def test_eszamanli_satislar_ortak_acik_lot_durumunu_olcer(tmp_path, method):
    p = _csv(tmp_path, [_islem("2024-01-01", "2024-03-01", ep=10, xp=15),
                        _islem("2024-02-01", "2024-03-01", ep=20, xp=30)])
    r = tr.build(p, lot_method=method)
    assert r["n_overlap_dispositions"] == 2
    assert r["n"] == 2 and r["total"] == 150 and r["unmatched"] == []


@pytest.mark.parametrize("method", ["fifo", "hifo"])
def test_kesirli_miktarlar_hayali_acik_lot_birakmaz(tmp_path, method):
    p = _csv(tmp_path, [_islem("2024-01-01", "", 0.1, 10, 15),
                        _islem("2024-02-01", "", 0.2, 10, 15),
                        _islem("", "2024-03-01", 0.3, 0, 15),
                        _islem("2024-04-01", "2024-05-01", 0.1, 10, 15)])
    r = tr.build(p, lot_method=method)
    assert r["n"] == 3 and r["total"] == 2 and r["unmatched"] == []
    assert r["n_overlap_dispositions"] == 1


def test_sembol_ve_varlik_turu_ayri_havuzlardir(tmp_path):
    p = _csv(tmp_path, [_islem("2024-01-01", "2024-03-01", ep=10, xp=15),
                        _islem("2024-02-01", "2024-04-01", ep=20, xp=30, asset_type="fund"),
                        _islem("2024-02-01", "2024-04-01", ep=40, xp=50, symbol="B")])
    r = tr.build(p, lot_method="hifo")
    assert [s["cost_basis"] for s in r["rows"]] == [100, 200, 400]
    assert r["n_overlap_dispositions"] == 0


@pytest.mark.parametrize("diger_giris,aday", [("2024-02-01", True), ("2024-01-31", False),
                                           ("2024-04-01", True), ("2024-04-02", False)])
def test_wash_eslesen_kayip_ve_baska_lotun_30_gun_siniri(tmp_path, diger_giris, aday):
    p = _csv(tmp_path, [_islem("2023-01-01", "2024-03-02", ep=120, xp=110, realized_pnl=100),
                        _islem(diger_giris, "2024-05-01", ep=100, xp=130)])
    r = tr.build(p, lot_method="hifo")
    assert r["rows"][0]["gain_loss"] == -100
    assert r["rows"][0]["wash_sale_candidate"] is aday
    assert all(s["adjustment_code"] == "" and s["adjustment_amount"] == 0 for s in r["rows"])


def test_wash_ayni_tarihteki_farkli_lot_sayilir_kendisi_sayilmaz(tmp_path):
    s = _islem("2024-01-01", "2024-01-20", ep=120, xp=110)
    p = _csv(tmp_path, [s])
    assert not tr.build(p, lot_method="hifo")["rows"][0]["wash_sale_candidate"]
    p = _csv(tmp_path, [s, {**s, "exit_time": "2024-02-01"}])
    assert tr.build(p, lot_method="hifo")["rows"][0]["wash_sale_candidate"]


@pytest.mark.parametrize("method", ["", "FIFO", "lifo", None])
def test_gecersiz_yontem_api_ihlalidir(tmp_path, method):
    with pytest.raises(ValueError, match="lot_method"):
        tr.build(tmp_path / "yok.csv", lot_method=method)


@pytest.mark.parametrize("method", YONTEMLER)
def test_bos_eksik_ve_okunamayan_dosya(tmp_path, method):
    p = tmp_path / "yok.csv"
    assert tr.build(p, lot_method=method)["reason"] == "DOSYA_YOK"
    p = _csv(tmp_path, [])
    r = tr.build(p, lot_method=method)
    assert r["reason"] == "BOS" and r["rows"] == [] and r["n"] == 0
    assert r["lot_method"] == method and r["unmatched"] == []
    p.write_text("", encoding="utf-8")
    assert tr.build(p, lot_method=method)["reason"].startswith("OKUNAMADI")
    p.write_text("symbol\nA\n", encoding="utf-8")
    assert tr.build(p, lot_method=method)["reason"].startswith("KOLON_EKSIK")


def test_cli_yontemi_aktarir_ve_ortusmeyi_gosterir(tmp_path, monkeypatch, capsys):
    p = _csv(tmp_path, [_islem("2024-01-02", "2024-03-01"),
                        _islem("2024-02-01", "2024-04-01", ep=120, xp=130)])
    out = tmp_path / "rapor.csv"
    monkeypatch.setattr(sys, "argv", ["tax_report", "--trades", str(p), "--out", str(out),
                                     "--lot-method", "hifo"])
    monkeypatch.setattr(tr, "ensure_utf8_stdout", lambda: None)
    tr._cli()
    yazi = capsys.readouterr().out
    assert "Lot yöntemi        : hifo" in yazi and "Örtüşen satış      : 1" in yazi
    with out.open(encoding="utf-8", newline="") as f:
        assert [float(r["gain_loss"]) for r in csv.DictReader(f)] == [-100, 300]
