"""pano_i18n.py — panonun İngilizce sözlüğünün KAPSAMA denetimi (Tur-108).

Pano kaynağı Türkçedir ve öyle kalır (onlarca test kaynak metnine bakar). İngilizce arayüz bir
ÇALIŞMA ANI katmanıdır: `nihai/web/pano_i18n.js` DOM metin düğümlerini, öznitelikleri ve canvas
yazılarını `nihai/web/pano_en.json` sözlüğünden çevirir. Sözlüğün anahtarı ekranda görünen Türkçe
metnin boşlukları tekilleştirilmiş hâlidir; değişken taşıyan metinde yer tutucular `{0}`, `{1}`…
(JS şablonundaki `${…}` ve `'…' + x + '…'` birleştirmeleri) kalıp olur.

Bu araç pano kaynaklarını (web_dashboard.html, lab_dashboard.html, lab_dashboard.js) tarar, ekranda
görünebilecek metinleri çıkarır ve sözlükte karşılığı OLMAYANLARI basar. Yeni bir pano metni
ekleyen tur bunu koşar ve eksikleri `pano_en.json`a yazar.

Kullanım:
    python -m tools.pano_i18n            # sözlükte olmayan Türkçe metinler (çıkış 1 = eksik var)
    python -m tools.pano_i18n --hepsi    # Türkçe-özel harf taşımayan adaylar da (gürültülüdür)

Kapı: `tests/test_tur108_pano_ingilizce.py` — HTML'deki her Türkçe metin düğümü/özniteliği
sözlükte olmalı; JS sabitlerinde kapsama oranı eşiğin altına düşmemeli.

Sınır: çıkarım sezgiseldir. Arka uçtan gelen metinler (API'nin döndürdüğü gerekçe/not) ve nesne
anahtarından üretilen başlıklar kaynak taramasıyla görülmez; onlar sözlüğe elle eklenir.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "nihai" / "web"
KAYNAKLAR = ("web_dashboard.html", "lab_dashboard.html", "lab_dashboard.js")
SOZLUK = WEB / "pano_en.json"

_HARF = "A-Za-zÇĞİÖŞÜçğıöşüâîû"
TR_OZEL = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_KELIME = re.compile(rf"[{_HARF}]{{2,}}")
_OZNITELIK = ("title", "placeholder", "aria-label", "alt", "label")
_YT = "\x00"  # yer tutucu (sonra {0},{1}… diye numaralanır)


def norm(s: str) -> str:
    """Sözlük anahtarı biçimi: boşluk tekil, uçlar kırpık (motorun `norm`u ile aynı)."""
    return re.sub(r"\s+", " ", s).strip()


def _numarala(s: str) -> str:
    parca = s.split(_YT)
    return "".join(p + (f"{{{i}}}" if i < len(parca) - 1 else "") for i, p in enumerate(parca))


# ---------------------------------------------------------------- JS dizgi/şablon tarayıcı
_REGEX_ONCESI = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_KELIME = {"return", "typeof", "case", "in", "of", "do", "else", "void", "delete", "throw", "new"}
_KACIS = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


class _JS:
    """JS kaynağındaki dizgi ve şablon sabitlerini (yorum/regex atlayarak) toplar."""

    def __init__(self, src: str, satir0: int = 1):
        self.s, self.n, self.satir0 = src, len(src), satir0
        self.sabit: list[tuple[str, int, int]] = []  # (metin, baş, son)

    def kod(self, i: int, sablon_ici: bool = False) -> int:
        s, n = self.s, self.n
        derinlik, onceki, kelime = 0, "", ""
        while i < n:
            c = s[i]
            if c in " \t\r\n":
                i += 1
            elif s.startswith("//", i):
                j = s.find("\n", i)
                i = n if j < 0 else j
            elif s.startswith("/*", i):
                j = s.find("*/", i + 2)
                i = n if j < 0 else j + 2
            elif c in "'\"":
                i = self.dizgi(i, c)
                onceki, kelime = "s", ""
            elif c == "`":
                i = self.sablon(i)
                onceki, kelime = "s", ""
            elif c == "/" and (onceki == "" or onceki in _REGEX_ONCESI or kelime in _REGEX_KELIME):
                i = self.regex(i)
                onceki, kelime = "s", ""
            elif c.isalnum() or c in "_$":
                j = i
                while j < n and (s[j].isalnum() or s[j] in "_$"):
                    j += 1
                onceki, kelime, i = "w", s[i:j], j
            else:
                if c == "{":
                    derinlik += 1
                elif c == "}":
                    if sablon_ici and derinlik == 0:
                        return i + 1
                    derinlik -= 1
                onceki, kelime = c, ""
                i += 1
        return i

    def dizgi(self, i: int, q: str) -> int:
        s, n, bas, out = self.s, self.n, i, []
        i += 1
        while i < n and s[i] != q and s[i] != "\n":
            if s[i] == "\\" and i + 1 < n:
                d = s[i + 1]
                if d == "u" and re.fullmatch(r"[0-9a-fA-F]{4}", s[i + 2:i + 6]):
                    out.append(chr(int(s[i + 2:i + 6], 16)))
                    i += 6
                    continue
                out.append("" if d == "\n" else _KACIS.get(d, d))
                i += 2
                continue
            out.append(s[i])
            i += 1
        i = min(i + 1, n)
        self.sabit.append(("".join(out), bas, i))
        return i

    def sablon(self, i: int) -> int:
        s, n, bas, out = self.s, self.n, i, []
        i += 1
        while i < n and s[i] != "`":
            if s[i] == "\\" and i + 1 < n:
                out.append(_KACIS.get(s[i + 1], s[i + 1]))
                i += 2
            elif s.startswith("${", i):
                out.append(_YT)
                i = self.kod(i + 2, sablon_ici=True)
            else:
                out.append(s[i])
                i += 1
        i = min(i + 1, n)
        self.sabit.append(("".join(out), bas, i))
        return i

    def regex(self, i: int) -> int:
        s, n, j, sinif = self.s, self.n, i + 1, False
        while j < n and s[j] != "\n":
            if s[j] == "\\":
                j += 2
                continue
            if s[j] == "[":
                sinif = True
            elif s[j] == "]":
                sinif = False
            elif s[j] == "/" and not sinif:
                j += 1
                while j < n and s[j].isalpha():
                    j += 1
                return j
            j += 1
        return i + 1  # regex değilmiş (bölme)

    def calis(self) -> list[tuple[str, int]]:
        """(metin, satır) — `+` ile değişkene bitişen uçlara yer tutucu eklenmiş."""
        self.kod(0)
        s, out = self.s, []
        for metin, bas, son in self.sabit:
            j = bas - 1
            while j >= 0 and s[j] in " \t\r\n":
                j -= 1
            once = j >= 0 and s[j] == "+" and not (j > 0 and s[j - 1] == "+")
            k = son
            while k < len(s) and s[k] in " \t\r\n":
                k += 1
            sonra = s.startswith("+", k) and not s.startswith(("++", "+="), k)
            out.append(((_YT if once else "") + metin + (_YT if sonra else ""),
                        self.satir0 + s.count("\n", 0, bas)))
        return out


_ETIKET = re.compile(r"<[^<>]*>")
_OZN = re.compile(r"""\b(?:title|placeholder|aria-label|alt|label)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_KOD_GIBI = re.compile(r"[<>{}=;]|^\s*[/#.]|\(\)|=>|px\b|rgba?\(|var\(--|^https?:|\.(js|json|csv|html|png|py)\b")


def _parcalar(metin: str) -> list[str]:
    """HTML taşıyan JS sabitini metin düğümlerine + özniteliklere böl."""
    if "<" not in metin or ">" not in metin:
        return [metin]
    ozn = [m.group(1) if m.group(1) is not None else m.group(2) for m in _OZN.finditer(metin)]
    # Sabit bir etiketin ORTASINDA başlıyor/bitiyorsa ('…">Metin', '<img alt="…" src="') o artık atılır.
    govde = re.sub(r"<[^<>]*$", "", re.sub(r"""^[^<>]*["'=][^<>]*>""", "", metin))
    return ozn + _ETIKET.split(govde)


def _gosterim_mi(k: str) -> bool:
    """JS sabiti ekranda görünen metne benziyor mu (tanımlayıcı/CSS/yol gürültüsünü eler)."""
    cekirdek = re.sub(r"\{\d+\}", " ", k)
    if not _KELIME.search(cekirdek):
        return False
    if TR_OZEL.search(cekirdek):
        return True
    if _KOD_GIBI.search(cekirdek):
        return False
    b = cekirdek.split()
    if all(re.fullmatch(r"[a-z0-9_:\-.]+", x) for x in b) and (
            len(b) == 1 or any("-" in x or "_" in x or any(ch.isdigit() for ch in x) for x in b)):
        return False  # tanımlayıcı / sınıf listesi
    return not (len(b) == 1 and re.fullmatch(r"[a-z][A-Za-z0-9_]*", b[0]))


class _Sayfa(HTMLParser):
    def __init__(self, ekle):
        super().__init__(convert_charrefs=True)
        self.ekle, self.yigin, self.betik = ekle, [], []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.yigin.append(tag)
        satir = self.getpos()[0]
        for ad, deger in attrs:
            if not deger:
                continue
            if ad in _OZNITELIK or (ad == "value" and tag in ("input", "button")):
                self.ekle(deger, satir, "html")
            elif ad.startswith("on"):
                for metin, st in _JS(deger, satir).calis():
                    for p in _parcalar(metin):
                        self.ekle(p, st, "js")

    def handle_endtag(self, tag):
        if self.yigin and self.yigin[-1] == tag:
            self.yigin.pop()

    def handle_data(self, data):
        if not self.yigin:
            self.ekle(data, self.getpos()[0], "html")
        elif self.yigin[-1] == "script":
            self.betik.append((data, self.getpos()[0]))


def topla(kok: Path = WEB) -> dict[str, dict]:
    """Pano kaynaklarındaki aday metinler: anahtar → {"yer": [dosya:satır…], "tur": {"html","js"}}."""
    bulunan: dict[str, dict] = {}
    for ad in KAYNAKLAR:
        def ekle(ham: str, satir: int, tur: str, ad=ad) -> None:
            k = norm(_numarala(ham if tur == "html" else html.unescape(ham)))
            if not k or re.fullmatch(r"(\{\d+\}\W*)+", k):
                return
            if not (_KELIME.search(k) if tur == "html" else _gosterim_mi(k)):
                return
            d = bulunan.setdefault(k, {"yer": [], "tur": set()})
            d["tur"].add(tur)
            if len(d["yer"]) < 3:
                d["yer"].append(f"{ad}:{satir}")

        metin = (kok / ad).read_text(encoding="utf-8")
        if ad.endswith(".js"):
            betikler = [(metin, 1)]
        else:
            sayfa = _Sayfa(ekle)
            sayfa.feed(metin)
            betikler = sayfa.betik
        for betik, satir in betikler:
            for sabit, st in _JS(betik, satir).calis():
                for p in _parcalar(sabit):
                    ekle(p, st, "js")
    return bulunan


def sozluk() -> dict[str, str]:
    return json.loads(SOZLUK.read_text(encoding="utf-8"))


def eksikler(hepsi: bool = False) -> dict[str, dict]:
    """Sözlükte karşılığı olmayan adaylar (varsayılan: yalnız Türkçe-özel harf taşıyanlar)."""
    s = sozluk()
    return {k: v for k, v in topla().items() if k not in s and (hepsi or TR_OZEL.search(k))}


def main() -> int:
    from nihai.console import ensure_utf8_stdout
    ensure_utf8_stdout()          # Tur-24: çıktı dosyaya/pipe'a yönlenince cp1254'e düşer
    ap = argparse.ArgumentParser(description="Pano İngilizce sözlüğünde eksik metinleri listele")
    ap.add_argument("--hepsi", action="store_true", help="Türkçe-özel harf taşımayan adayları da bas")
    a = ap.parse_args()
    e = eksikler(hepsi=a.hepsi)
    for k, v in e.items():
        print(f"{v['yer'][0]:<28} [{'/'.join(sorted(v['tur']))}] {k}")
    print(f"[pano_i18n] aday {len(topla())} · sözlük {len(sozluk())} · eksik {len(e)}")
    return 1 if e else 0


if __name__ == "__main__":
    raise SystemExit(main())
