"""
telegram_bot.py — Tek-yönlü bildirim (sinyal/uyarı operatöre).

PDF Referansları:
- Madde 47 : Kritik olaylar anlık operatöre iletilir (push/Telegram/SMS)
- Madde 36 : Heartbeat (sessiz çöküş tespiti) — Telegram'a periyodik nabız

Token boşsa modül sessizce no-op çalışır.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime

from loguru import logger

from nihai import net
from nihai.atomic_io import tail_lines
from nihai.config import DATA_DIR, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# --- Paylaşılan HTTP istemcisi (katalog #27) ----------------------------------------
# Modül düzeyi `httpx.get/post` HER ÇAĞRIDA yeni bir `Client` kurar ve bunun bedeli ağ
# DEĞİL, SSL bağlamı + CA demeti yüklemesidir (ölçüldü: p50 224,66 ms). `CommandListener`
# 7/24 ~26 sn'de bir `getUpdates` çağırır → günde ~3.300 kez ödenen bir vergiydi.
#
# İstemci artık `nihai.net`te: aynı kusur `data/nasdaq_calendar` ve `execution/screen`
# içinde de vardı ve orada DÖNGÜ içindeydi. Üç modül tek istemciyi (tek SSL bağlamı, tek
# bağlantı havuzu) paylaşır; `_http` geriye dönük ad olarak korunur.
_http = net.shared_client
_EVENT_LOCK = threading.Lock()


def _event(kind: str, **fields) -> None:
    try:
        path = DATA_DIR / "telegram_status.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with _EVENT_LOCK, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": kind, "at": time.time(), **fields}, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning(f"[telegram] durum kaydedilemedi: {type(exc).__name__}")


def status(*, probe: bool = False) -> dict:
    if probe and is_enabled():
        try:
            base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
            me = _http().get(base + "getMe", timeout=10).json()
            webhook = _http().get(base + "getWebhookInfo", timeout=10).json()
            info = webhook.get("result") or {}
            _event("probe", ok=bool(me.get("ok") and webhook.get("ok")),
                   username=(me.get("result") or {}).get("username"),
                   webhook_active=bool(info.get("url")), pending_updates=info.get("pending_update_count"))
        except Exception as exc:
            _event("probe", ok=False, error=type(exc).__name__)
    path = DATA_DIR / "telegram_status.jsonl"
    events = []
    for line in tail_lines(path, 500) if path.exists() else []:
        try:
            entry = json.loads(line)
            if isinstance(entry, dict):
                events.append(entry)
        except ValueError:
            continue
    latest = {e["kind"]: e for e in events if isinstance(e, dict) and "kind" in e}
    poll = latest.get("poll")
    return {"enabled": is_enabled(), "probe": latest.get("probe"), "last_send": latest.get("send"),
            "poll": poll, "poll_fresh": bool(poll and poll.get("ok") and time.time() - poll["at"] < 90),
            "commands": [e for e in events if e.get("kind") == "command"][-20:],
            "templates": {"test": "Nihai Algoritma web paneli: test bildirimi.",
                          "signal": "SİNYAL: SEMBOL — eylem, fiyat, hedef, stop, R:R, kalite",
                          "alert": "Seviye — açıklama — zaman"}}


def is_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def _post(text: str) -> bool:
    try:
        response = _http().post(_API, json={
            "chat_id": TELEGRAM_CHAT_ID, "text": text,
            "parse_mode": "Markdown", "disable_web_page_preview": True,
        }, timeout=10)
        payload = response.json()
        ok = response.status_code == 200 and bool(payload.get("ok"))
        _event("send", ok=ok, status_code=response.status_code)
        return ok
    except Exception as e:
        _event("send", ok=False, error=type(e).__name__)
        logger.warning(f"[telegram] gönderim hatası: {type(e).__name__}")
        return False


def send(text: str) -> None:
    """Fire-and-forget: ağ çağrısı arka plan daemon thread'inde — ana tarama
    döngüsü Telegram yavaş/erişilemez olsa bile bloke olmaz (timeout 10 sn idi)."""
    if not is_enabled():
        return
    threading.Thread(target=_post, args=(text,), daemon=True).start()


def _post_photo(image: bytes, caption: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        response = _http().post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"},
            files={"photo": ("chart.png", image, "image/png")},
            timeout=20,
        )
        ok = response.status_code == 200 and bool(response.json().get("ok"))
        _event("send", ok=ok,
               status_code=response.status_code, photo=True)
        return ok
    except Exception as e:
        _event("send", ok=False, error=type(e).__name__, photo=True)
        logger.warning(f"[telegram] foto gönderim hatası: {type(e).__name__}")
        return False


def send_photo(image: bytes, caption: str = "") -> None:
    """Fire-and-forget PNG gönderimi (B2 /grafik). Token yoksa veya görsel boşsa no-op."""
    if not is_enabled() or not image:
        return
    threading.Thread(target=_post_photo, args=(image, caption), daemon=True).start()


def notify_signal(symbol: str, action: str, price: float, rr: float, quality: float,
                  reasons: str = "", stop: float | None = None,
                  target: float | None = None, hedef_nominal: bool = False) -> None:
    lines = [f"📡 *SİNYAL: {symbol}*",
             f"Eylem: `{action}` @ `{price:.2f}`"]
    if stop is not None and target is not None:
        # Tur-106 (BULGU-106-28): quant / trailing kipinde pozisyon sabit hedefte KAPANMAZ (çıkış izleyen stop);
        # bu sistemde uyarı = emir olduğundan nominal hedef kâr-al seviyesi gibi okunmamalı.
        etiket = "Hedef (nominal · çıkış izleyen stopla)" if hedef_nominal else "Hedef"
        lines.append(f"🎯 {etiket}: `{target:.2f}`  |  🛑 Stop: `{stop:.2f}`")
    lines.append(f"R:R: `{rr:.2f}` | Kalite: `{quality:.2f}`")
    if reasons:
        lines.append(reasons)
    lines.append(f"_{datetime.now():%H:%M:%S}_")
    send("\n".join(lines))


def notify_alert(level: str, text: str) -> None:
    icon = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "🚨", "CRIT": "🔴"}.get(level, "•")
    send(f"{icon} *{level}*\n{text}\n_{datetime.now():%H:%M:%S}_")


def heartbeat() -> None:
    """Madde 36 — seans-farkındalıklı heartbeat çağırılır."""
    send(f"💚 heartbeat | {datetime.now():%Y-%m-%d %H:%M:%S}")


# Telegram mesaj sınırı 4096 karakter; karne uzarsa sunucu 400 döner ve mesaj TAMAMEN
# kaybolur. Kırpma sessiz olmasın diye kesildiği söylenir.
_TG_MAX = 3900


def ai_status_text() -> str:
    """AI dürüstlük karnesi metni (KARAR-DIŞI) — `nihai/ai` katmanının Telegram bağı (Tur-37).

    Bildirim katmanı bugüne kadar sistemin PERFORMANSINI (sinyal/uyarı/heartbeat) taşıyordu
    ama kendi TAHMİN KALİTESİ hakkında tek satır söylemiyordu: operatör telefondan "model
    kalibre mi" sorusunu SORAMIYORDU. `/beyin` komutu (execution/main.py) bunu sorar.

    İçerik `nihai.ai.observatory.ai_status_text` tarafından üretilir ve SALT OKUNURDUR —
    hiçbir sınav yeniden koşulmaz, hiçbir model fit edilmez, hiçbir karar etkilenmez.
    AI katmanı yok/hatalıysa dürüstçe söyler (sessiz boş mesaj YOK)."""
    try:
        from nihai.ai.observatory import ai_status_text as _text
        out = (_text() or "").strip()
    except Exception as e:      # noqa: BLE001 — bildirim yolu AI katmanına bağımlı olamaz
        logger.debug(f"[telegram] AI karnesi üretilemedi: {e}")
        return "🧠 AI durumu okunamadı (katman yüklenemedi)."
    if not out:
        return "🧠 AI durumu ölçülmedi (henüz artefakt yok)."
    return out if len(out) <= _TG_MAX else out[:_TG_MAX] + "\n… (kırpıldı)"


# ---------------------------------------------------------------------------
# Komut dinleyici (iki-yönlü) — operatör telefondan watchlist/durum yönetir.
# Saf ayrıştırma (parse_command) ağdan ayrıştırıldı → test edilebilir.
# ---------------------------------------------------------------------------
def parse_command(text: str) -> tuple[str | None, str | None]:
    """'/add nvda' → ('add', 'NVDA'). Bot adı son eki ('/add@bot') temizlenir.
    Komut değilse (None, None). Argüman sembol gibi normalize edilir (upper)."""
    if not text:
        return None, None
    text = text.strip()
    if not text.startswith("/"):
        return None, None
    parts = text[1:].split()
    if not parts:
        return None, None
    cmd = parts[0].split("@", 1)[0].lower()      # '/add@MyBot' → 'add'
    arg = parts[1].strip().upper() if len(parts) > 1 else None
    return cmd, arg


def _get_updates(offset: int, timeout: int = 25) -> list[dict]:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    try:
        r = _http().get(url, params={"offset": offset, "timeout": timeout},
                        timeout=timeout + 10)
        payload = r.json()
        ok = r.status_code == 200 and bool(payload.get("ok"))
        _event("poll", ok=ok, status_code=r.status_code)
        return payload.get("result", []) if ok else []
    except Exception as e:
        _event("poll", ok=False, error=type(e).__name__)
        logger.debug(f"[telegram] getUpdates hatası: {type(e).__name__}")
        return []


class CommandListener:
    """getUpdates long-polling thread'i. Yalnızca yetkili chat_id'den komut kabul
    eder. handlers: {komut: çağrılabilir(arg)->str}. Dönen str operatöre yanıtlanır."""

    def __init__(self, handlers: dict, *, chat_id: str | None = None):
        self.handlers = handlers
        self.chat_id = str(chat_id or TELEGRAM_CHAT_ID)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset = 0

    def dispatch(self, text: str) -> str | None:
        """Saf dispatch (test edilebilir): metni komuta çevirip handler'ı çağırır."""
        cmd, arg = parse_command(text)
        if cmd is None:
            return None
        handler = self.handlers.get(cmd)
        if handler is None:
            return f"Bilinmeyen komut: /{cmd}. /help yaz."
        try:
            return handler(arg)
        except Exception as e:
            return f"Hata: {e}"

    def _loop(self) -> None:
        logger.info(f"[telegram] komut dinleyici başladı ({len(self.handlers)} komut)")
        while not self._stop.is_set():
            for upd in _get_updates(self._offset):
                self._offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                if str(msg.get("chat", {}).get("id")) != self.chat_id:
                    continue                              # yetkisiz kaynak
                cmd, arg = parse_command(msg.get("text", ""))
                if cmd:
                    _event("command", command=cmd, argument=(arg or "")[:40])
                reply = self.dispatch(msg.get("text", ""))
                if reply:
                    send(reply)
            self._stop.wait(1.0)

    def start(self) -> None:
        if not is_enabled() or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="tg-commands", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
