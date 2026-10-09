"""
dolt_options.py — DoltHub `post-no-preference/options` SQL API istemcisi.

NEDEN: projede tarihsel opsiyon verisi HİÇ yoktu. `data_provider.get_option_chain`
yalnız CANLI yfinance zincirini verir; `iv_history.py` ise IV Rank'i günde bir satır
biriktirerek ~252 günde oluşturmaya çalışır (docstring'i itiraf ediyor) → Madde 16
pratikte ölü, backtest edilemez. `backtest.py` opsiyonu `gamma=0.02` sabiti ve
`theta=-prem/dte` ile fiyatlar; opsiyon bacağı ölçüm değil kurgudur.

Bu modül iki tabloyu getirir (2019-02-09 → bugün, 2321 sembol):
  volatility_history : iv_current/iv_year_high/iv_year_low + hv_* → hazır IV Rank
  option_chain       : date, expiration, strike, call_put, bid, ask, vol(IV),
                       delta, gamma, theta, vega, rho → GERÇEK bid/ask + greeks

PIT / SURVIVORSHIP: Dolt **sürümlü** bir veritabanıdır. `as_of="HEAD~50"` (commit
hash veya `TIMESTAMP('2024-01-02')` de olur) verilirse sorgu o commit'teki tabloyu
okur → gerçek point-in-time anlık görüntü. ACI DERS #1'in istediği okuma budur.
Ölçüldü: AAPL HEAD'de 1254 satır, `AS OF 'HEAD~50'` ile 1229.

BAĞIMLILIK: yalnız `requests`. dolt CLI / doltpy GEREKMEZ.

SESSİZ KIRPMA UYARISI: API tek yanıtta en fazla 1000 satır döndürür ve sayfalama
token'ı YOKTUR — kırpıldığında yalnız `query_execution_status="RowLimit"` der. Bu
projede en pahalı hata sınıfı sessizce eksilen veridir, o yüzden `query()`
LIMIT/OFFSET ile kendi sayfalamasını yapar ve sayfalayamadığı durumda SUSMAZ.

SORGU BİÇİMİ KURALI (ölçüldü 2026-08-13, sunucu bunu affetmiyor):
`option_chain`/`volatility_history` PK'si (date, act_symbol, …) ile başlar. Sunucu
tarafında PK ÖNEKİ DIŞINDA bir sıralama ya da çok-sembollü `IN (...)` filtresi tam
tablo taraması tetikler ve sorgu `context deadline exceeded` ile ÖLÜR:
    ORDER BY act_symbol, date  → 51.522 ms   (aynı 53 satır)
    ORDER BY date              →     418 ms
    ORDER BY yok               →     323 ms
    IN ('AAPL','MSFT','SPY','QQQ') + ORDER BY date → ZAMAN AŞIMI
Bu yüzden: sunucuya TEK sembol sorulur, sıralama en fazla `date` ile yapılır, geri
kalan gruplama/sıralama İSTEMCİDE yapılır. Yeni sorgu eklerken bu kurala uyun.

ÖLÇEK SINIRI (dürüst kayıt): burası ücretsiz ve HIZ SINIRLI bir uç noktadır. Arka
arkaya hızlı istek `http status: 403` getirir; tablo 2321 sembol × ~1900 gün ≈ 4,4M
satırdır ve `WHERE act_symbol=X` PK önekini (date) kullanamadığı için her sorgu tam
tarama olur (~15 sn). Onlarca sembol için uygundur, BİNLERCE sembolü buradan çekmeye
çalışmayın — o iş için `dolt clone` (CLI) doğru araçtır. Deadline'a takılan sorgu
KISMİ satır döndürür; bu modül onları KULLANMAZ (eksik veri sessizce kabul edilirse
survivorship'ten beter bir hata sınıfı doğar). Toplu çekimde `DOLT_POLITE_DELAY_SEC`
artırılmalı ve `backfill_iv_history --skip-existing` ile koşu devam ettirilebilir.
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from loguru import logger

from nihai.config import DATA_DIR

try:  # opsiyonel bağımlılık TÜM kapıyı düşürmemeli (Tur-26 dersi)
    import requests
except ImportError:  # pragma: no cover - ortamda requests yoksa
    requests = None  # type: ignore[assignment]

DOLT_REPO = os.getenv("DOLT_OPTIONS_REPO", "post-no-preference/options")
DOLT_BRANCH = os.getenv("DOLT_OPTIONS_BRANCH", "master")  # DİKKAT: 'main' YOK
DOLT_API = "https://www.dolthub.com/api/v1alpha1"
# Toplu CSV 217 MB'dır (option_chain daha da büyük). DATA_DIR OneDrive altındaysa bu
# dosyalar senkronize edilir; bu projede OneDrive kilidi daha önce state.json'ı
# bozmuştu (Tur-28/atomic_io). Salt-okunur bir ÖNBELLEK olduğu için OneDrive dışına
# alınabilir: DOLT_CACHE_DIR=D:\dolt_cache gibi.
DOLT_CACHE_DIR = Path(os.getenv("DOLT_CACHE_DIR", str(DATA_DIR / "dolt_cache")))

API_ROW_LIMIT = 1000          # sunucu tarafı sabit; aşılırsa status="RowLimit"
PAGE_SIZE = 1000              # kendi sayfalamamızın adımı
MAX_PAGES = 500               # emniyet freni (500k satır)
REQUEST_TIMEOUT = 90
RETRY_SLEEP_SEC = 3.0
POLITE_DELAY_SEC = float(os.getenv("DOLT_POLITE_DELAY_SEC", "2.0"))  # semboller arası

# Bu tablolarda tarih kolonu 'date', sembol kolonu 'act_symbol'
NUMERIC_HINT = (
    "iv_", "hv_", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho", "strike",
)


class DoltQueryError(RuntimeError):
    """Sorgu sunucuda hata döndürdü (ağ hatası DEĞİL)."""


# Sunucunun GEÇİCİ hataları: yanıt HTTP 200 gelir ama status=Error olur. Bunlar
# yeniden denenmelidir — kalıcı SQL hatasından (yazım/kolon) ayırt etmek şart, yoksa
# bozuk sorgu için 4 kez beklenir. Ölçüldü: aynı sorgu bir denemede 418 ms, sonrakinde
# deadline; DoltHub'ın ücretsiz uç noktası yük altında dalgalanıyor.
_RETRYABLE_MSG = ("context deadline exceeded", "connection reset", "too many connections",
                  "timeout", "temporarily unavailable")
# Hız sınırı ayrı ele alınır: 403/429 geldiğinde hızlı yeniden deneme durumu KÖTÜLEŞTİRİR.
_RATE_LIMIT_MSG = ("http status: 403", "http status: 429", "rate limit", "too many requests")
RATE_LIMIT_SLEEP_SEC = float(os.getenv("DOLT_RATE_LIMIT_SLEEP_SEC", "60"))


def _api_get(sql: str, *, retries: int = 3) -> dict:
    """Tek HTTP sorgusu. Ağ hatasında VE sunucunun geçici hatalarında yeniden dener."""
    if requests is None:
        raise DoltQueryError(
            "requests kurulu değil — `pip install requests` (dolt CLI gerekmiyor)."
        )
    url = f"{DOLT_API}/{DOLT_REPO}/{DOLT_BRANCH}?q={quote(sql, safe='')}"
    last_err: str = "?"
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            status = payload.get("query_execution_status")
            if status in ("Success", "RowLimit"):
                return payload
            msg = str(payload.get("query_execution_message", status))
            last_err = msg
            low = msg.lower()
            # Deadline'a takılan sorgu KISMİ satır da döndürür (ölçüldü: bir yıllık
            # sorgu Error + 28 satır). Bunları KULLANMIYORUZ — sessizce eksik veri,
            # bu projenin en pahalı hata sınıfı. Ya tam sonuç ya hata.
            if any(m in low for m in _RATE_LIMIT_MSG):
                logger.warning(
                    f"[dolt] HIZ SINIRI ({msg}) — {RATE_LIMIT_SLEEP_SEC:.0f} sn bekleniyor. "
                    f"Toplu çekimde DOLT_POLITE_DELAY_SEC'i artırın."
                )
                time.sleep(RATE_LIMIT_SLEEP_SEC)
                continue
            if not any(m in low for m in _RETRYABLE_MSG):
                raise DoltQueryError(f"sorgu hatası: {msg}")   # kalıcı → hemen bırak
            logger.debug(f"[dolt] geçici sunucu hatası (deneme {attempt + 1}): {msg}")
        except DoltQueryError:
            raise
        except Exception as e:                                  # ağ/5xx/json
            last_err = str(e)
        if attempt < retries:
            time.sleep(RETRY_SLEEP_SEC * (2 ** attempt))        # 3 / 6 / 12 sn
    raise DoltQueryError(f"DoltHub isteği başarısız ({retries + 1} deneme): {last_err}")


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    """API satırları (hepsi string) → tipli DataFrame."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
    for col in df.columns:
        if any(col.startswith(h) or col == h for h in NUMERIC_HINT):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _is_paginatable(sql: str) -> bool:
    """LIMIT/OFFSET eklenebilir mi? Zaten LIMIT içeren sorguya dokunmayız."""
    tail = sql.strip().rstrip(";").lower()
    return " limit " not in f" {tail} "


def query(sql: str, *, paginate: bool = True) -> pd.DataFrame:
    """Ham SQL çalıştır → DataFrame.

    `paginate=True` ise sorgu LIMIT/OFFSET ile sayfalanarak 1000-satır sunucu
    limiti aşılır. Sayfalanamayan (kullanıcının kendi LIMIT'ini yazdığı) bir sorgu
    limite dayanırsa UYARI basılır — sessizce eksik veri döndürmeyiz.

    PIT için sorgunun içine `AS OF` yazın:
        query("SELECT * FROM volatility_history AS OF 'HEAD~50' WHERE ...")
    """
    if not paginate or not _is_paginatable(sql):
        payload = _api_get(sql)        # hata → yükseltir, buraya yalnız başarı gelir
        rows = payload.get("rows") or []
        if payload.get("query_execution_status") == "RowLimit" or len(rows) >= API_ROW_LIMIT:
            logger.warning(
                f"[dolt] sorgu {API_ROW_LIMIT} satırda KIRPILDI ve sayfalanamadı "
                f"(sorguda kendi LIMIT'iniz var). Veri EKSİK olabilir: {sql[:120]}"
            )
        return _rows_to_df(rows)

    base = sql.strip().rstrip(";")
    frames: list[pd.DataFrame] = []
    for page in range(MAX_PAGES):
        offset = page * PAGE_SIZE
        payload = _api_get(f"{base} LIMIT {PAGE_SIZE} OFFSET {offset}")
        rows = payload.get("rows") or []
        if rows:
            frames.append(_rows_to_df(rows))
        if len(rows) < PAGE_SIZE:
            break                      # son sayfa
    else:
        logger.warning(
            f"[dolt] MAX_PAGES={MAX_PAGES} doldu — sonuç KIRPILMIŞ olabilir: {sql[:120]}"
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _as_of(table: str, as_of: str | None) -> str:
    """`AS OF` ekini üret. as_of: 'HEAD~50' | commit hash | "TIMESTAMP('2024-01-02')"."""
    if not as_of:
        return table
    if as_of.upper().startswith("TIMESTAMP("):
        return f"{table} AS OF {as_of}"
    return f"{table} AS OF '{as_of}'"


def _quote_sym(symbol: str) -> str:
    """Tek sembolü SQL literaline çevir (tırnak enjeksiyonunu kırp)."""
    return "'" + symbol.upper().replace("'", "") + "'"


def volatility_history(symbols: str | list[str], *,
                       start: str | None = None, end: str | None = None,
                       as_of: str | None = None) -> pd.DataFrame:
    """Sembol(ler)in tarihsel IV/HV serisi.

    Kolonlar: date, act_symbol, iv_current, iv_year_high, iv_year_low, hv_current …
    `as_of` verilirse o commit'teki (point-in-time) tablo okunur.

    Semboller sunucuya TEK TEK sorulur (yukarıdaki SORGU BİÇİMİ KURALI): çok-sembollü
    `IN (...)` zaman aşımına giriyor. Yan fayda: bir sembolün hatası partiyi düşürmez.
    """
    syms = [symbols] if isinstance(symbols, str) else list(symbols)
    frames: list[pd.DataFrame] = []
    for i, sym in enumerate(syms):
        where = [f"act_symbol = {_quote_sym(sym)}"]
        if start:
            where.append(f"date >= '{start}'")
        if end:
            where.append(f"date <= '{end}'")
        sql = (
            "SELECT date, act_symbol, iv_current, iv_year_high, iv_year_low, "
            "hv_current, hv_year_high, hv_year_low "
            f"FROM {_as_of('volatility_history', as_of)} "
            f"WHERE {' AND '.join(where)} ORDER BY date"
        )
        try:
            df = query(sql)
        except DoltQueryError as e:
            logger.warning(f"[dolt] {sym} volatility_history alınamadı: {e}")
            continue
        if not df.empty:
            frames.append(df)
        if i + 1 < len(syms):
            time.sleep(POLITE_DELAY_SEC)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["act_symbol", "date"], ignore_index=True)  # istemci tarafı


def option_chain(symbol: str, *, date: str, expiration: str | None = None,
                 as_of: str | None = None) -> pd.DataFrame:
    """Bir günün GERÇEK opsiyon zinciri: bid/ask + IV + tam greeks.

    `backtest.py`'nin sentetik Black-Scholes bacağının yerine geçmek içindir:
    orada gamma sabit 0.02, theta = -prim/dte idi; burada ölçülmüş değerler var.
    Sunucuda ORDER BY YOK (PK öneki dışı sıralama zaman aşımı yapıyor) → istemcide.
    """
    where = [f"act_symbol = {_quote_sym(symbol)}", f"date = '{date}'"]
    if expiration:
        where.append(f"expiration = '{expiration}'")
    sql = (
        "SELECT date, act_symbol, expiration, strike, call_put, bid, ask, vol, "
        "delta, gamma, theta, vega, rho "
        f"FROM {_as_of('option_chain', as_of)} "
        f"WHERE {' AND '.join(where)}"
    )
    df = query(sql)
    if df.empty:
        return df
    return df.sort_values(["expiration", "strike", "call_put"], ignore_index=True)


def iv_rank_series(symbols: str | list[str], *,
                   start: str | None = None, end: str | None = None,
                   as_of: str | None = None) -> pd.DataFrame:
    """volatility_history → tarihsel IV Rank (0-100) serisi.

    Rank = (iv_current − iv_year_low) / (iv_year_high − iv_year_low) × 100.
    52 haftalık pencere veri sağlayıcı tarafından zaten hesaplanmıştır; bu yüzden
    `iv_history.get_iv_rank`'ın 252-gün-birikme kısıtı burada YOKTUR.
    Aralık sıfıra çökmüşse (hi==lo) satır NaN bırakılır — 50.0 gibi bir sayı
    uydurmak, ölçüm yokluğunu ölçüm gibi göstermek olurdu.
    """
    df = volatility_history(symbols, start=start, end=end, as_of=as_of)
    if df.empty:
        return df
    span = df["iv_year_high"] - df["iv_year_low"]
    df["iv_rank"] = ((df["iv_current"] - df["iv_year_low"]) / span * 100.0).where(
        span > 1e-9
    )
    df["iv_rank"] = df["iv_rank"].clip(0, 100)
    return df


def _resolve_ref(ref: str) -> str:
    """Göreli ref'i (`HEAD~50`, `HEAD^`) commit hash'ine çevir.

    CSV uç noktası göreli ref KABUL ETMEZ (ölçüldü: `HEAD~50` → HTTP 500), ama commit
    hash'i kabul eder ve UYDURMA ref'e 404 verir — yani 200 dönmesi, ref'in gerçekten
    uygulandığının kanıtıdır. SQL `AS OF` ise göreli ref'i kabul eder. Bu fark
    sessiz bırakılırsa `--as-of HEAD~50` SQL kipinde PIT, toplu kipte hata olurdu.
    """
    r = ref.strip()
    if not r.upper().startswith("HEAD"):
        return r                                  # zaten hash ya da branch adı
    suffix = r[4:]
    if not suffix:
        n = 0
    elif suffix.startswith("~") and suffix[1:].isdigit():
        n = int(suffix[1:])
    elif set(suffix) == {"^"}:
        n = len(suffix)
    else:
        return r                                  # tanımadık biçim — olduğu gibi bırak
    df = query(f"SELECT commit_hash FROM dolt_log LIMIT {n + 1}", paginate=False)
    if df.empty or len(df) <= n:
        raise DoltQueryError(f"{ref} çözülemedi (dolt_log {len(df)} satır döndü)")
    resolved = str(df["commit_hash"].iloc[n])
    logger.info(f"[dolt] {ref} → {resolved}")
    return resolved


def bulk_csv(table: str, *, ref: str | None = None, max_age_days: int = 7,
             force: bool = False) -> Path | None:
    """Tüm tabloyu TEK istekte CSV olarak indir → yerel dosya yolu döndür.

    SQL API toplu çekim için yanlış araçtır (yukarıdaki ÖLÇEK SINIRI): sembol başına
    tam tarama, 54 sn deadline, 403. Bu uç nokta ise tabloyu akıtır:
        https://www.dolthub.com/csv/<owner>/<repo>/<ref>/<table>
    Ölçüldü: ilk bayt ~1,6 sn, hız sınırı tetiklenmiyor. Onlarca sembol için bile
    tek indirme + yerel filtreleme, sembol-başına sorgudan hem hızlı hem naziktir.

    `ref` PIT içindir (branch adı ya da commit hash). Yeni indirme yapılmadan
    önce yerel kopya `max_age_days`'ten tazeyse yeniden indirilmez.
    Ağ hatasında None döner (fail-soft) — çağıran bayat kopyayla devam edebilir.
    """
    if requests is None:
        logger.warning("[dolt] requests yok — toplu indirme atlandı")
        return None
    ref = _resolve_ref(ref) if ref else DOLT_BRANCH
    safe_ref = ref.replace("/", "_").replace("~", "-")
    dest = DOLT_CACHE_DIR / f"{table}__{safe_ref}.csv"

    if dest.exists() and not force:
        age_days = (time.time() - dest.stat().st_mtime) / 86400.0
        if age_days < max_age_days:
            logger.info(f"[dolt] {dest.name} taze ({age_days:.1f} gün) — indirilmedi")
            return dest

    url = f"https://www.dolthub.com/csv/{DOLT_REPO}/{ref}/{table}"
    DOLT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f".csv.indiriliyor-{os.getpid()}")
    logger.info(f"[dolt] {table} indiriliyor ({url}) …")
    t0 = time.time()
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
            r.raise_for_status()
            written = 0
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    written += len(chunk)
        # Yarım indirmeyi ASLA yayımlama: ancak tamamlandıktan sonra yerine koy
        tmp.replace(dest)
        logger.info(f"[dolt] {table}: {written / (1 << 20):.1f} MB, "
                    f"{time.time() - t0:.0f} sn → {dest.name}")
        return dest
    except Exception as e:
        logger.warning(f"[dolt] {table} toplu indirme başarısız: {e}")
        tmp.unlink(missing_ok=True)
        return dest if dest.exists() else None      # bayat kopya > kopya yok


def _sidecar_path(csv_path: Path) -> Path:
    """Toplu CSV'nin yanındaki parquet dosyasının yolu (aynı dizin, aynı gövde)."""
    return csv_path.with_suffix(".parquet")


def _build_sidecar(csv_path: Path, dest: Path, chunksize: int) -> bool:
    """CSV → parquet yan dosyası. Kurulduysa True.

    ŞEMA PANDAS'IN ÇIKARIMIDIR — pyarrow'un KENDİ CSV okuyucusu bilerek
    kullanılmaz: o `date` gibi kolonlarda farklı bir dtype çıkarır ve çağıranın
    gördüğü çerçeve sessizce değişirdi (ölçüldü: `str` → `object`). Parçalar ilk
    parçanın şemasına cast edilir; edilemezse yan dosya KURULMAZ ve çağıran CSV
    yoluna düşer — yarım/bozuk bir yan dosya, bayat veriden beter bir hata
    sınıfıdır (`bulk_csv`'nin yarım-indirme disiplininin aynısı).

    pyarrow OPSİYONELDİR (requirements'ta bildirilmez): yoksa False döner ve
    davranış eskisiyle birebir aynı kalır (Tur-26: opsiyonel bağımlılık tüm
    kapıyı düşürmemeli).
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False
    tmp = dest.with_suffix(f".parquet.yaziliyor-{os.getpid()}")
    writer = None
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            tbl = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
            writer.write_table(tbl.cast(writer.schema))
        if writer is None:
            return False                      # boş CSV → yan dosya kurma
        writer.close()
        writer = None
        tmp.replace(dest)
        return True
    except Exception as e:
        logger.warning(f"[dolt] parquet yan dosyası kurulamadı ({dest.name}): {e}")
        return False
    finally:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)


def read_bulk(path: str | Path | None, *, cols: list[str] | None = None,
              symbols: Iterable[str] | None = None,
              chunksize: int = 500_000, build_cache: bool = True,
              start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Toplu tabloyu OKU — istenen kolonlar + (varsa) `act_symbol` süzmesiyle.

    Bu fonksiyon ÜÇ çağıranda birebir kopyalanmış olan "parça parça oku, sembolü
    süz, birleştir" döngüsünün tek kaynağıdır (`volatility_history_bulk`,
    `research/edge_lab/build_vrp_dataset`, `research/iv_rank_threshold_test`).

    ÖLÇÜLDÜ (volatility_history, 230 MB / 1.913.770 satır, 5 sembol süzmesi):
    CSV parça taraması **1,61 sn** ↔ parquet yan dosyası **0,06 sn = 26,7×**;
    tam okuma 1,48 → 0,54 sn (2,7×); yan dosya 230 MB → **30 MB**. Kolon
    sırası, dtype'lar ve satır sırası CSV yolundakiyle AYNIDIR (testle kilitli).

    TAZELİK: `bulk_csv` dosyayı 7 günde bir yeniden indirir; yan dosya CSV'den
    ESKİYSE yeniden kurulur. Bayat bir yan dosya sessizce eski veri döndürürdü —
    bu depoda en pahalı hata sınıfı budur.

    Süzme `act_symbol`'ü BÜYÜK HARFE katlayarak yapar (dolt tablosunda kolon
    zaten büyük harftir: 300.000 satırda 0 istisna), yani yalnız küçük harfle
    sorulan bir sembol artık boş dönmek yerine eşleşir.
    """
    empty = pd.DataFrame(columns=list(cols)) if cols else pd.DataFrame()
    if path is None:
        return empty
    path = Path(path)               # çağıranlar `bulk_csv` çıktısını str olarak da geçebilir
    if not path.exists():
        return empty
    syms = None if symbols is None else sorted({s.upper() for s in symbols})

    def select(frame):
        if syms is not None:
            frame = frame[frame["act_symbol"].str.upper().isin(syms)]
        if start is not None:
            frame = frame[frame["date"].astype(str) >= start]
        if end is not None:
            frame = frame[frame["date"].astype(str) <= end]
        return frame

    side = _sidecar_path(path)
    ready = side.exists() and side.stat().st_mtime >= path.stat().st_mtime
    if not ready and build_cache:
        ready = _build_sidecar(path, side, chunksize)
    if ready:
        try:
            import pyarrow.parquet as pq
            names = pq.ParquetFile(side).schema_arrow.names
            want = [c for c in names if cols is None or c in cols]   # DOSYA sırası korunur
            filt = [("act_symbol", "in", syms)] if syms is not None else None
            return select(pq.read_table(side, columns=want, filters=filt).to_pandas()).reset_index(drop=True)
        except Exception as e:
            logger.warning(f"[dolt] parquet yan dosyası okunamadı ({side.name}): {e} — CSV'ye düşülüyor")

    frames: list[pd.DataFrame] = []
    try:
        for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize):
            chunk = select(chunk)
            if not chunk.empty:
                frames.append(chunk)
    except Exception as e:
        logger.warning(f"[dolt] toplu CSV okunamadı {path.name}: {e}")
        return empty
    return pd.concat(frames, ignore_index=True) if frames else empty


def prepare_option_history(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Yerel zincirin fiyat/sözleşme sınırı. Eksik Greek fiyatı geçersiz kılmaz;
    girişte ayrıca aranır. Çelişen mükerrer kotasyonlar, çapraz makas ve saat
    bileşenli fiyat tarihleri atılır. Geçersiz çerçeve/sembol → boş DataFrame."""
    required = {"date", "act_symbol", "expiration", "strike", "call_put", "bid", "ask"}
    if (not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns)
            or not isinstance(symbol, str) or not symbol.strip()):
        return pd.DataFrame()
    data = frame.loc[frame["act_symbol"].str.upper() == symbol.upper()].copy()
    n_raw = len(data)
    for col in ("date", "expiration"):
        dates = pd.to_datetime(data[col], errors="coerce")
        # Gün içi kotasyonu gece yarısına çekmek kapanış kararına gelecek veri sızdırır.
        data[col] = dates.where(dates == dates.dt.normalize()) if col == "date" else dates.dt.normalize()
    data["call_put"] = data["call_put"].str.lower().map({"call": "c", "put": "p", "c": "c", "p": "p"})
    for col in ("strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega"):
        data[col] = pd.to_numeric(data[col], errors="coerce") if col in data else np.nan
    valid = (np.isfinite(data[["strike", "bid", "ask"]]).all(axis=1)
             & data["date"].notna() & data["expiration"].notna() & data["call_put"].notna()
             & (data["strike"] > 0) & (data["bid"] >= 0) & (data["ask"] > 0)
             & (data["ask"] >= data["bid"]) & (data["expiration"] >= data["date"]))
    data = data.loc[valid].drop_duplicates()
    keys = ["date", "expiration", "strike", "call_put"]
    data = data.loc[~data.duplicated(keys, keep=False)].sort_values(keys).reset_index(drop=True)
    data.attrs["rejected_rows"] = n_raw - len(data)
    return data


def local_option_history(symbol: str, *, path: str | Path | None = None,
                         start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Ağsız günlük zincir. 10 GB CSV'yi bir kez süzer; toplu parquet kopyası kurmaz.

    Tarih filtresi fiyat-tarihi PIT sağlar; güncel CSV geçmiş revizyonların ilk
    yayımlanma zamanını kanıtlamaz. Bunun için tarihsel Dolt commit ihracı gerekir.
    """
    if path is None:
        name = f"option_chain__{DOLT_BRANCH.replace('/', '_').replace('~', '-')}.csv"
        path = DOLT_CACHE_DIR / name
        if not path.exists() and "DOLT_CACHE_DIR" not in os.environ:
            path = Path.home() / "dolt_cache" / name  # pano toplu indirmesinin konumu
    data = prepare_option_history(read_bulk(path, symbols=[symbol], build_cache=False,
                                           start=start, end=end), symbol)
    data.attrs["source"] = str(Path(path).resolve())
    return data


def option_snapshot(history: pd.DataFrame, *, as_of: str) -> pd.DataFrame:
    """Günlük KAPANIŞ kararında görülebilen son zincir; gelecek güne uzanmaz.

    Saat damgası kabul edilmez: kaynak intraday zamanlama kanıtı taşımıyor.
    Geçersiz history/as_of → boş DataFrame (istisna yok).
    """
    if not isinstance(history, pd.DataFrame) or not isinstance(as_of, (str, date, np.datetime64)):
        return pd.DataFrame()
    try:
        cutoff = pd.Timestamp(as_of)
    except (TypeError, ValueError, OverflowError):
        return pd.DataFrame()
    if pd.isna(cutoff) or cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        return pd.DataFrame()
    if history.empty:
        return history.copy()
    try:
        eligible = history["date"] <= cutoff
        latest = history.loc[eligible, "date"].max()
        return history.loc[history["date"] == latest].copy()
    except (KeyError, TypeError, ValueError):
        return pd.DataFrame()


def volatility_history_bulk(symbols: str | list[str], *, ref: str | None = None,
                            force: bool = False) -> pd.DataFrame:
    """Toplu CSV'den sembolleri süz — SQL API'ye hiç dokunmaz.

    Okuma `read_bulk`a devredilir (parquet yan dosyası varsa 26,7× hızlı; yoksa
    eskisi gibi parça parça CSV).
    """
    syms = {s.upper() for s in ([symbols] if isinstance(symbols, str) else symbols)}
    path = bulk_csv("volatility_history", ref=ref, force=force)
    cols = ["date", "act_symbol", "iv_current", "iv_year_high", "iv_year_low",
            "hv_current", "hv_year_high", "hv_year_low"]
    out = read_bulk(path, cols=cols, symbols=syms)
    if out.empty:
        return pd.DataFrame()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values(["act_symbol", "date"], ignore_index=True)


def coverage(as_of: str | None = None) -> dict:
    """Veri kapsamı (ilk/son tarih, sembol sayısı). Bağlantı testi olarak da kullanılır."""
    sql = (
        "SELECT MIN(date) AS ilk, MAX(date) AS son, "
        "COUNT(DISTINCT act_symbol) AS sembol "
        f"FROM {_as_of('volatility_history', as_of)}"
    )
    df = query(sql, paginate=False)
    return {} if df.empty else df.iloc[0].to_dict()


if __name__ == "__main__":  # pragma: no cover - canlı ağ gerektirir
    from nihai.console import ensure_utf8_stdout

    ensure_utf8_stdout()
    print("kapsam:", coverage())
    ivr = iv_rank_series("AAPL", start="2026-06-01")
    print(f"AAPL IV rank satır={len(ivr)}")
    if not ivr.empty:
        print(ivr.tail(3)[["date", "iv_current", "iv_rank"]].to_string(index=False))
    # PIT kanıtı. DİKKAT: yalnız `past_n < now_n` demek YETMEZ — sorgu tamamen
    # başarısız olup 0 satır dönerse 0 < 1254 de doğrudur ve kapı, elemesi gereken
    # durumu ONAYLAR (ACI DERS #8'in aynı kalıbı). Bu yüzden önce past_n > 0 aranır.
    now_n = len(volatility_history("AAPL"))
    past_n = len(volatility_history("AAPL", as_of="HEAD~50"))
    print(f"PIT kontrolü: HEAD={now_n} satır, HEAD~50={past_n} satır")
    assert now_n > 0, "HEAD okunamadı — bağlantı/sorgu sorunu, PIT iddiası SINANAMADI"
    assert past_n > 0, "AS OF sorgusu BOŞ döndü — sınanmadı, 'past<now' burada kanıt DEĞİL"
    assert past_n < now_n, "AS OF eski commit'i okumuyor — PIT iddiası GEÇERSİZ"
    print("OK")
