"""
prob_sizing.py — Kalibre P(win) → pozisyon-boyutu güven çarpanı (saf fonksiyon).

Meta-labeling kapısı (meta_gate.probability) bir sinyalin kazanma olasılığını [0,1]
döndürür. İkili kapı (eşik altını TAMAMEN ele) yerine, olasılığı RİSKE oransal bir
çarpana çevirip pozisyonu kademeli boyutlamak istenir: yüksek-olasılık → tam risk,
düşük-olasılık → kısılmış risk. Çıktı [floor, cap] arasına kırpılır ve doğrudan
RiskManager.size_*(confidence_multiplier=...) kancasına verilir (zaten [0,1]'e kırpar).

⚠️ DÜRÜSTLÜK NOTU (EDGE_FINDINGS opt-in disiplini):
Bu eşleme matematiksel olarak doğrudur ama yalnızca model olasılığı GERÇEKTEN kalibre
ve isabetliyse para kazandırır. Mevcut canlı veride (predictions_log) confidence ile
isabet TERS korele çıktı (yüksek güven → düşük isabet). O yüzden bunu boyutlamaya
uygulamak (PROB_SIZING) varsayılan KAPALI olmalı; ancak train_meta.py'ın OOS kapı
simülasyonu modelin gerçek bir kenar gösterdiğini kanıtladıktan sonra açılmalıdır.

Saf ve bağımlılıksız (numpy bile gerekmez) → birim-test edilebilir, hot-path'te ucuz.
"""
from __future__ import annotations


def prob_to_size_mult(
    prob: float | None,
    *,
    p_lo: float = 0.45,
    p_hi: float = 0.60,
    floor: float = 0.25,
    cap: float = 1.0,
) -> float:
    """Kalibre kazanma olasılığını [floor, cap] güven çarpanına eşle (parça-lineer ramp).

    prob <= p_lo            → floor       (en zayıf sinyaller minimum riskle)
    prob >= p_hi            → cap          (en güçlü sinyaller tam riskle)
    p_lo < prob < p_hi      → floor..cap arası lineer interpolasyon

    prob None (model pasif/hata) → cap döndürülür: olasılık bilgisi yoksa boyutlama
    NÖTR kalır (mevcut davranışı bozmaz; kapı pasifken sistem aynen çalışır).

    Parametre güvenliği: p_lo >= p_hi gibi dejenere aralıkta basit eşik davranışına
    düşülür (prob >= p_hi → cap, aksi → floor); 0/sıfıra-bölme üretmez.
    """
    if prob is None:
        return cap
    # Olasılığı [0,1]'e kırp (model nadiren sınır dışı verebilir).
    p = 0.0 if prob < 0.0 else 1.0 if prob > 1.0 else float(prob)
    if p <= p_lo:
        return floor
    if p >= p_hi:
        return cap
    span = p_hi - p_lo
    if span <= 0.0:  # dejenere aralık → eşik davranışı (yukarıdaki iki dal zaten kapsar)
        return cap if p >= p_hi else floor
    frac = (p - p_lo) / span
    return floor + frac * (cap - floor)
