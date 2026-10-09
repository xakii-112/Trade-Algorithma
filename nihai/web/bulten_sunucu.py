"""Bülten — mevcut panodan BAĞIMSIZ ikinci arayüz.

`web_dashboard.py` / `web_dashboard.html` dosyalarına dokunmaz: mevcut FastAPI uygulamasını
olduğu gibi içe aktarır, üstüne yalnız `GET /bulten` sayfasını ekler. Aynı `/api/*` uçlarını
kullanır; mevcut pano `/` adresinde aynen çalışmaya devam eder.

Çalıştırma (bağımsız, mevcut `python -m nihai.web.web_dashboard` moduyla aynı anlam):
    python -m nihai.web.bulten_sunucu                → http://127.0.0.1:8765/bulten
    python -m nihai.web.bulten_sunucu --port 8766    (ana sunucu 8765'i tutuyorsa)
    python -m nihai.web.bulten_sunucu --tahmin       (canlı tahmin servisini de başlat)

Ana sunucuda (`main --serve-web`) da görünmesi için tek satır yeterlidir:
`import nihai.web.bulten_sunucu` — istenmediği için eklenmedi.
"""
import argparse
import json
from pathlib import Path

import uvicorn
from fastapi.responses import HTMLResponse

from nihai.config import WEB_HOST, WEB_PORT, WEB_TOKEN
from nihai.console import ensure_utf8_stdout
from nihai.web.web_dashboard import app

_HTML = Path(__file__).with_name("bulten.html")
_NOCACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}


@app.get("/bulten", response_class=HTMLResponse, include_in_schema=False)
def bulten_sayfasi():
    html = _HTML.read_text(encoding="utf-8")
    # Mevcut panoyla aynı same-origin token enjeksiyonu; `<` kaçışı token'ın </script> ile
    # etiketi kapatmasını engeller.
    token = json.dumps(WEB_TOKEN).replace("<", "\\u003c")
    html = html.replace("</head>", f"<script>window.__WEB_TOKEN__ = {token};</script>\n</head>", 1)
    return HTMLResponse(html, headers=_NOCACHE)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bülten panosunu bağımsız başlat.")
    ap.add_argument("--host", default=WEB_HOST)
    ap.add_argument("--port", type=int, default=WEB_PORT)
    ap.add_argument("--tahmin", action="store_true", help="canlı tahmin servisini de başlat")
    args = ap.parse_args()
    if args.tahmin:
        from nihai.execution.live_predict_service import get_service
        get_service().start()
    print(f"Bülten: http://{args.host}:{args.port}/bulten   (klasik pano: /)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    ensure_utf8_stdout()   # Tur-24: yönlendirilmiş stdout'ta cp1254 çökmesini önle
    main()
