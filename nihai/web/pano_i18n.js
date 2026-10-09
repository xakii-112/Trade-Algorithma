// [Tur-108] PANO DİLİ: TR (kaynak) ↔ EN (çalışma anı sözlük katmanı).
// Kaynak HTML/JS Türkçedir ve DEĞİŞMEZ; İngilizce kipte bu betik ekrana giden metni (DOM metin
// düğümleri, title/placeholder/aria-label, canvas yazıları, confirm/alert) pano_en.json sözlüğünden
// çevirir. Sunucu sözlüğü bu dosyanın ÖNÜNE `window.__PANO_EN=…` olarak ekler (/i18n.js): betik
// <head>'de eşzamanlı yüklendiği için İngilizce kipte Türkçe metin hiç boyanmaz.
// Seçim: başlıktaki TR | EN düğmesi (localStorage 'pano.dil') ya da adres çubuğunda ?lang=en|tr.
// Kip değişince sayfa yeniden yüklenir (grafik kütüphaneleri ölçtükleri metni önbelleğe alır).
(function () {
    'use strict';
    const ANAHTAR = 'pano.dil';
    const istek = new URLSearchParams(location.search).get('lang');
    let dil = istek === 'en' || istek === 'tr' ? istek : 'tr';
    try {
        if (istek === dil) localStorage.setItem(ANAHTAR, dil);
        else dil = localStorage.getItem(ANAHTAR) === 'en' ? 'en' : 'tr';
    } catch (e) { /* depolama kapalıysa yalnız ?lang= çalışır */ }
    window.panoDil = dil;

    document.addEventListener('DOMContentLoaded', function () {
        const d = document.createElement('button');
        d.type = 'button';
        d.id = 'pano-dil';
        d.className = 'status-badge mono';
        d.setAttribute('translate', 'no');
        d.title = dil === 'en' ? 'Türkçe arayüze geç' : 'Switch to English';
        d.style.cssText = 'cursor:pointer;font:inherit;color:var(--text-muted)';
        const etiket = (k) => `<span style="${k === dil ? 'color:var(--neon-blue);font-weight:800' : ''}">${k.toUpperCase()}</span>`;
        d.innerHTML = etiket('tr') + ' | ' + etiket('en');
        d.onclick = function () {
            const yeni = dil === 'en' ? 'tr' : 'en';
            try { localStorage.setItem(ANAHTAR, yeni); } catch (e) { /* aşağıdaki ?lang= taşır */ }
            const u = new URL(location.href);
            u.searchParams.set('lang', yeni);
            location.replace(u);
        };
        (document.querySelector('.system-status') || document.body).appendChild(d);
    });
    if (dil !== 'en') return;

    // ---- sözlük: tam eşleşme + kalıp ({0},{1}… = değişken parça) ----
    const SOZLUK = window.__PANO_EN || {};
    const HARF = /[A-Za-zÇĞİÖŞÜçğıöşüâîû]/;
    const kacis = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const tam = new Map(), kalip = [];
    // Kenardaki ayraç ("· metin", "metin:") ve sayı ("12 metin", "metin 490") anahtarın parçası sayılmaz.
    const UC = /^([·|•—–]\s+)?(.*?)(\s+[·|•—–:]|[:;.,])?$/s;
    const SAYI = /^([\d.,%+\-−$×~≈≤≥]+\s+)?(.*?)(\s+[\d.,%+\-−$×~≈≤≥]+)?$/s;
    function ekle(tr, en) {
        if (!/\{\d+\}/.test(tr)) { if (!tam.has(tr)) tam.set(tr, en); return; }
        const sabit = tr.split(/\{\d+\}/);
        if (!HARF.test(sabit.join(''))) return;
        // Sabit kısmı kısa, genel kalıp ("{0} gün", "{0} önce") yalnız YALIN bir parçayı yakalar; yoksa
        // "a, b: c gün" gibi bileşik bir metnin tamamını yutar ve içindeki tam karşılıkları ezer.
        const g = sabit.join('').trim().length < 8 ? '([^,:;·|()]*?)' : '(.*?)';
        let kaynak = sabit.map(kacis).join(g);
        // Uçtaki yer tutucu boş kalabilir ("… kayıt {3}" → "… kayıt"): yanındaki boşluk isteğe bağlıdır.
        if (kaynak.startsWith(g + ' ')) kaynak = '(?:' + g + ' )?' + kaynak.slice(g.length + 1);
        if (kaynak.endsWith(' ' + g)) kaynak = kaynak.slice(0, -g.length - 1) + '(?: ' + g + ')?';
        kalip.push([new RegExp('^' + kaynak + '$', 's'), en, sabit.join('').length]);
    }
    for (const [tr, en] of Object.entries(SOZLUK)) {
        ekle(tr, en);
        const m = UC.exec(tr), n = UC.exec(en);   // "· çekimser {0}" anahtarı "çekimser 30" parçasını da çevirir
        if ((m[1] || m[3]) && m[2] && n[2] && !(m[2] in SOZLUK)) ekle(m[2], n[2]);
    }
    kalip.sort((a, b) => b[2] - a[2]);   // en özgül (sabit kısmı en uzun) kalıp önce
    // Bileşik metin (iki cümle, "a · b", "Etiket: değer") kalıba uymazsa BÜYÜKTEN küçüğe ayraçlarla
    // bölünür ve her parça ayrı çevrilir; çevrilemeyen parça olduğu gibi kalır.
    const AYRAC = [/((?<=[.!?;])\s+)/, /(\s[·|•]\s)/, /(\s[—–→↔]\s)/, /(,\s)/, /(:\s)/, /(\s\(|\)\s?)/, /(\s\/\s)/];
    const onbellek = new Map();

    function cekirdek(c, yalnizTam) {
        const t = tam.get(c);
        if (t !== undefined) return t;
        if (yalnizTam || c.length > 900) return c;   // uzun serbest metin: yalnız tam eşleşme
        for (const [re, en] of kalip) {
            const m = re.exec(c);
            if (m) return en.replace(/\{(\d+)\}/g, (_, i) => cevir(m[+i + 1] || '')).replace(/\s+/g, ' ').trim();
        }
        for (const kirp of [UC, SAYI]) {
            const m = kirp.exec(c);
            if (!(m[1] || m[3]) || !m[2]) continue;
            const r = cevir(m[2]);
            if (r !== m[2]) return (m[1] || '') + r + (m[3] || '');
        }
        for (const re of AYRAC) {
            const p = c.split(re);
            if (p.length > 1) return p.map((x, i) => (i % 2 ? x : cevir(x))).join('');
        }
        return c;
    }

    function cevir(s, yalnizTam) {
        if (!s || !HARF.test(s)) return s;
        const anahtar = yalnizTam ? '\x01' + s : s;
        let r = onbellek.get(anahtar);
        if (r !== undefined) return r;
        const m = /^(\s*)([\s\S]*?)(\s*)$/.exec(s);
        const c = m[2].replace(/\s+/g, ' ');
        const en = cekirdek(c, yalnizTam);
        r = en === c ? s : m[1] + en + m[3];
        if (onbellek.size > 20000) onbellek.clear();
        onbellek.set(anahtar, r);
        return r;
    }
    window.panoCevir = cevir;

    // ---- DOM: metin düğümleri + öznitelikler ----
    const OZNITELIK = ['title', 'placeholder', 'aria-label', 'alt', 'label'];
    const ATLA = { SCRIPT: 1, STYLE: 1, TEXTAREA: 1, NOSCRIPT: 1 };
    const SERBEST = 'pre,.log-line,.pre-box,#ev-lessons,#ev-findings';

    function metin(n) {
        const v = n.nodeValue, p = n.parentElement;
        if (n.__en === v || !p || ATLA[p.nodeName] || p.closest('[translate="no"]')) return;
        // Arka uçtan gelen serbest metin (log satırı, brif/özet kutusu, belge alıntısı): yalnız TAM
        // eşleşme — sözlükte olmayan düz yazıyı parça parça çevirmek yarım, okunmaz bir karışım üretir.
        const r = cevir(v, !!p.closest(SERBEST));
        if (r !== v) { n.__en = r; n.nodeValue = r; }
    }

    function oznitelik(el) {
        for (const a of OZNITELIK) {
            const v = el.getAttribute(a);
            if (!v || el['__en_' + a] === v) continue;
            const r = cevir(v);
            if (r !== v) { el['__en_' + a] = r; el.setAttribute(a, r); }
        }
    }

    function gez(kok) {
        if (kok.nodeType === 3) return metin(kok);
        if (kok.nodeType !== 1 || ATLA[kok.nodeName]) return;
        oznitelik(kok);
        const w = document.createTreeWalker(kok, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
            acceptNode: (n) => (n.nodeType === 1 && ATLA[n.nodeName] ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
        });
        for (let n = w.nextNode(); n; n = w.nextNode()) (n.nodeType === 3 ? metin : oznitelik)(n);
    }

    document.documentElement.lang = 'en';
    gez(document.documentElement);
    new MutationObserver(function (kayitlar) {
        for (const k of kayitlar) {
            if (k.type === 'characterData') metin(k.target);
            else if (k.type === 'attributes') oznitelik(k.target);
            else k.addedNodes.forEach(gez);
        }
    }).observe(document.documentElement, {
        subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: OZNITELIK,
    });

    // ---- DOM dışı yüzeyler ----
    // Canvas (Chart.js / lightweight-charts / Emir Akışı): yazı çizimden VE ölçümden önce çevrilir
    // (ikisi aynı metni görmezse eksen/lejant yerleşimi kayar).
    const c2d = window.CanvasRenderingContext2D && CanvasRenderingContext2D.prototype;
    for (const ad of c2d ? ['fillText', 'strokeText', 'measureText'] : []) {
        const asil = c2d[ad];
        c2d[ad] = function (t, ...kalan) { return asil.call(this, typeof t === 'string' ? cevir(t) : t, ...kalan); };
    }
    for (const ad of ['alert', 'confirm', 'prompt']) {
        const asil = window[ad];
        window[ad] = function (m, ...kalan) { return asil.call(window, cevir(String(m)), ...kalan); };
    }
    // Sayı/tarih biçimi: kaynak 'tr-TR' ister (1.234,5 · "Eki"); İngilizce kipte en-GB (1,234.5 · "Oct", 24 saat).
    for (const [proto, ad] of [[Number.prototype, 'toLocaleString'], [Date.prototype, 'toLocaleString'],
        [Date.prototype, 'toLocaleDateString'], [Date.prototype, 'toLocaleTimeString']]) {
        const asil = proto[ad];
        proto[ad] = function (yerel, ...kalan) { return asil.call(this, yerel === 'tr-TR' ? 'en-GB' : yerel, ...kalan); };
    }
})();
