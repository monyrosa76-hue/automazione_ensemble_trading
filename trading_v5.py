#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 TRADING PULLBACK MULTI-TIMEFRAME — v5   (scanner live + backtest comparativo)
===============================================================================
 PERCHÉ LA v5 (cosa non funzionava nella v4)
 Il backtest v4 ha prodotto 1 solo trade in 52 giorni. Il foglio "Imbuto" ha
 mostrato tre cause, due delle quali erano errori del codice:

   1. "Impulso superato" (scartava il 44%): la v4 richiedeva che dopo l'ultimo
      massimo CONFERMATO non ci fosse alcun massimo più alto. Ma in un trend un
      nuovo massimo è la norma: è solo non ancora confermato. Ora l'impulso è
      ancorato al massimo (minimo) più alto del periodo, confermato o no.
   2. Zona 50–61,8% obbligatoria (scartava l'84%): troppo stretta. Ora la
      fascia è 38,2–78,6% e il ritracciamento EFFETTIVO viene registrato per
      ogni trade, con statistiche per fascia: decidono i dati, non l'opinione.
   3. Pinbar M15 obbligatoria (scartava il 98,5%): chiedere che la candela da
      15 minuti sia una pinbar proprio mentre il prezzo è in zona e sul livello
      è una coincidenza quasi impossibile — e serve a poco su trade che durano
      settimane. Ora il trigger è una MODALITÀ configurabile e il backtest
      confronta le tre varianti.

 In più, il limite dei dati: con il piano free si scaricano 5000 candele per
 timeframe, quindi la storia disponibile dipende dalla candela d'INGRESSO:
      M15 -> ~52 giorni | H1 -> ~9 mesi | H4 -> ~3 anni
 Per questo la v5 gestisce due set di timeframe con la stessa logica:
      D1H4H1   : bias D1, livelli H4, ingresso H1   (storia lunga, consigliato)
      H4H1M15  : bias H4, livelli H1, ingresso M15  (la configurazione v4)

 CONTROLLI ANTI-ILLUSIONE (nuovi)
   - Divisione del periodo: primi 70% (IS) / ultimi 30% (OOS). Se il risultato
     regge solo sull'IS, è adattamento al campione.
   - Preset "controllo_casuale": ingressi a caso con lo stesso stop, lo stesso
     target e gli stessi costi. Se le regole non battono il caso, non c'è edge.
   - Preset "v3_con_filtro_weekend": la v3 generava il 20% dei trade a mercato
     chiuso; così il confronto è onesto.
   - Foglio "Dati caricati": candele e date per strumento (nel report v4
     EUR/USD era sparito senza che si capisse il perché).
   - Nessuna size variabile in base ai bonus: rischio fisso finché un bonus non
     dimostra di valere qualcosa (le statistiche per bonus sono nel report).

 USO
   python trading_v5.py backtest      (consigliato per primo)
   python trading_v5.py scan
   Variabili d'ambiente (utili su GitHub Actions / Colab):
     TWELVEDATA_API_KEY, MODE=scan|backtest, TF_SET=D1H4H1|H4H1M15,
     ENTRY_MODE=zone|reversal|pinbar, RISK_EUR, BACKTEST_PRESETS="a,b,c"

 AVVERTENZA: strumento didattico. Nessun parametro è validato: serve il
 backtest, con almeno 100 trade per preset, prima di operare anche in demo.
===============================================================================
"""

import os, sys, time, math, re, json, warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURAZIONE
# =============================================================================
MODE = os.environ.get("MODE", "scan")
for _a in sys.argv[1:]:
    if _a in ("scan", "backtest"):
        MODE = _a

RISK_EUR      = float(os.environ.get("RISK_EUR", 10))
REQUEST_DELAY = float(os.environ.get("REQUEST_DELAY", 8))   # piano free: 8 richieste/minuto
OOS_SPLIT     = 0.70        # primi 70% del periodo = in-sample, ultimi 30% = out-of-sample
MIN_TRADES_OK = 100         # soglia per trarre conclusioni
MIN_TRADES_NOISE = 30       # sotto questa soglia è rumore

# Set di timeframe: ruoli BIAS (trend+impulso+Fibonacci) / LIVELLI (S/R) / INGRESSO (trigger)
TF_SETS = {
    "D1H4H1":  dict(bias="1day", level="4h",  entry="1h",
                    bars=(1500, 5000, 5000), time_stop_hours=30 * 24, entry_min=60),
    "H4H1M15": dict(bias="4h",   level="1h",  entry="15min",
                    bars=(400, 1000, 5000),  time_stop_hours=10 * 24, entry_min=60),
}
TF_SET_DEFAULT    = os.environ.get("TF_SET", "D1H4H1")
ENTRY_MODE_DEFAULT = os.environ.get("ENTRY_MODE", "reversal")

# Parametri della strategia (validi per entrambi i set di timeframe)
PARAMS = dict(
    ema_slope_bars   = 6,      # la EMA50 del timeframe bias deve salire/scendere
    impulse_lookback = 60,     # candele bias in cui cercare l'impulso
    min_impulse_atr  = 3.0,    # impulso minimo in ATR del timeframe bias
    fib_min          = 0.382,  # fascia di ritracciamento ammessa...
    fib_max          = 0.786,  # ...registrata poi trade per trade
    rsi_level_long   = (25, 65),   # RSI del timeframe livelli: né caduta libera né ripartenza
    rsi_level_short  = (35, 75),
    level_lookback   = 60,     # candele livelli in cui cercare supporti/resistenze
    level_tol_atr    = 0.30,   # due pivot sono lo stesso livello se distano < 0,3 ATR
    level_near_atr   = 1.00,   # il livello deve stare entro 1 ATR dal prezzo
    min_touches      = 2,
    trigger_lookback = 2,      # candele d'ingresso in cui cercare il trigger
    pin_wick         = 0.60,
    pin_body         = 0.35,
    pin_min_atr      = 0.50,
    engulf_body      = 0.60,
    rsi_entry_filter = False,  # filtro RSI sulla candela d'ingresso (solo modalità pinbar)
    rsi_entry_long   = (25, 60),
    rsi_entry_short  = (40, 75),
    stop_buffer_atr_level = 0.25,
    min_stop_atr_bias     = 1.00,   # stop mai più stretto di 1 ATR del timeframe bias
    rr_target        = 2.5,
    rr_min           = 1.8,    # spazio minimo fino al massimo/minimo dell'impulso
    max_spread_R     = 0.10,
    one_trade_per_impulse = True,   # un solo ingresso per impulso: evita trade ripetuti nella stessa fascia
    weekend_filter   = True,
    exposure_filter  = True,
    close_before_weekend = False,
)

# Preset del backtest: nome -> (set timeframe, modalità ingresso, override parametri)
PRESET_DEFS = {
    "D1H4H1_zone":      ("D1H4H1",  "zone",     {}),
    "D1H4H1_reversal":  ("D1H4H1",  "reversal", {}),
    "D1H4H1_pinbar":    ("D1H4H1",  "pinbar",   dict(rsi_entry_filter=True, pin_min_atr=0.8)),
    "H4H1M15_zone":     ("H4H1M15", "zone",     {}),
    "H4H1M15_reversal": ("H4H1M15", "reversal", {}),
    "H4H1M15_pinbar":   ("H4H1M15", "pinbar",   dict(rsi_entry_filter=True, pin_min_atr=0.8)),
    "v3_originale":        ("H4H1M15", "v3", dict(weekend_filter=False, exposure_filter=False)),
    "v3_con_filtro_weekend": ("H4H1M15", "v3", dict(weekend_filter=True, exposure_filter=False)),
    "controllo_casuale":   ("D1H4H1", "random", {}),
}
BACKTEST_PRESETS = [p.strip() for p in os.environ.get(
    "BACKTEST_PRESETS",
    "D1H4H1_zone,D1H4H1_reversal,D1H4H1_pinbar,H4H1M15_zone,H4H1M15_reversal,"
    "v3_originale,v3_con_filtro_weekend,controllo_casuale").split(",") if p.strip()]
RANDOM_TRADES_PER_SYMBOL = 40
RANDOM_SEED = 12345

# Esposizione: posizioni nella stessa direzione per chiave (valute: 1)
EXPOSURE_LIMIT = {"EQ_US": 2, "EQ_EU": 1, "EQ_ASIA": 1}

# Spread stimati in % del prezzo  [STIME DA VERIFICARE sulla propria piattaforma]
SPREAD_PCT = {"fx_major": 0.012, "fx_cross": 0.025, "metal": 0.035,
              "index": 0.015, "stock_us": 0.10, "stock_eu": 0.12, "stock_asia": 0.15}

FX_MAJ = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "AUD/USD"]
FX_CRS = ["EUR/GBP", "EUR/CHF", "GBP/CHF", "EUR/CAD", "EUR/AUD", "EUR/JPY", "CAD/JPY", "CHF/JPY"]
INSTRUMENTS = {p: (p, None, "fx_major", "fx") for p in FX_MAJ}
INSTRUMENTS.update({p: (p, None, "fx_cross", "fx") for p in FX_CRS})
INSTRUMENTS.update({
    "Oro (XAU/USD)":       ("XAU/USD", None,   "metal",    "fx"),
    "DAX 40":              ("DAX",     None,   "index",    "europe"),
    "Apple (AAPL)":        ("AAPL",  "NASDAQ", "stock_us", "us"),
    "Microsoft (MSFT)":    ("MSFT",  "NASDAQ", "stock_us", "us"),
    "NVIDIA (NVDA)":       ("NVDA",  "NASDAQ", "stock_us", "us"),
    "Amazon (AMZN)":       ("AMZN",  "NASDAQ", "stock_us", "us"),
    "Alphabet (GOOGL)":    ("GOOGL", "NASDAQ", "stock_us", "us"),
    "JPMorgan (JPM)":      ("JPM",   "NYSE",   "stock_us", "us"),
    "Coca-Cola (KO)":      ("KO",    "NYSE",   "stock_us", "us"),
    "Berkshire B (BRK.B)": ("BRK.B", "NYSE",   "stock_us", "us"),
})
BACKTEST_SYMBOLS = FX_MAJ + FX_CRS + ["Oro (XAU/USD)"]

SESSIONS = {"tokyo": (2, 9), "europe": (9, 17.5), "us": (15.5, 22)}
BASE_URL = "https://api.twelvedata.com/time_series"
INTERVAL_MIN = {"1day": 1440, "4h": 240, "1h": 60, "15min": 15}

try:
    import pytz
    ROME = pytz.timezone("Europe/Rome")
except ImportError:
    from zoneinfo import ZoneInfo
    ROME = ZoneInfo("Europe/Rome")
try:
    from google.colab import drive, userdata   # noqa
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def log(m=""):
    print(m, flush=True)

def get_api_key():
    k = os.environ.get("TWELVEDATA_API_KEY", "")
    if not k and IN_COLAB:
        try:
            k = userdata.get("TWELVEDATA_API_KEY")
        except Exception:
            k = ""
    return k or ""

def get_out_dir():
    d = os.environ.get("OUT_DIR", "")
    if not d:
        if IN_COLAB:
            drive.mount("/content/drive", force_remount=False)
            d = "/content/drive/MyDrive/TRADING_V5"
        else:
            d = os.path.join(os.getcwd(), "TRADING_V5")
    os.makedirs(os.path.join(d, "cache"), exist_ok=True)
    return d

# =============================================================================
# 2. DATI
# =============================================================================
def fetch_td(symbol, exchange, interval, n, api_key):
    params = {"symbol": symbol, "interval": interval, "outputsize": n, "apikey": api_key,
              "format": "JSON", "order": "ASC", "timezone": "UTC"}
    if exchange:
        params["exchange"] = exchange
    for attempt in range(2):
        try:
            j = requests.get(BASE_URL, params=params, timeout=30).json()
        except Exception as e:
            return None, f"RETE: {e}"
        if j.get("status") == "error" or "values" not in j:
            code = j.get("code", "?")
            if code == 429 and attempt == 0:
                log("    [RATE LIMIT] attendo 65s..."); time.sleep(65); continue
            label = {400: "SIMBOLO NON TROVATO", 401: "API KEY NON VALIDA",
                     403: "PIANO NON SUPPORTATO", 404: "DATI NON DISPONIBILI"}.get(code, f"ERR {code}")
            return None, f"{label}: {str(j.get('message', ''))[:80]}"
        df = pd.DataFrame(j["values"])
        df["time"] = pd.to_datetime(df["datetime"], utc=True)
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[["time", "open", "high", "low", "close"]].dropna()
        return df.sort_values("time").drop_duplicates("time").reset_index(drop=True), "OK"
    return None, "RETRY ESAURITO"

def drop_incomplete(df, interval, now_utc):
    if df is None or df.empty:
        return df
    return df[df["time"] + pd.Timedelta(minutes=INTERVAL_MIN[interval]) <= now_utc].reset_index(drop=True)

def ohlc_valid(df, min_rows):
    if df is None or len(df) < min_rows:
        return False
    bad = ((df.high < df.low) | (df.close > df.high) | (df.close < df.low) |
           (df.open > df.high) | (df.open < df.low) | (df.close <= 0))
    return not bool(bad.any())

def cache_path(cache_dir, name, interval):
    return os.path.join(cache_dir, f"{re.sub(r'[^A-Za-z0-9]', '_', name)}_{interval}.csv")

def load_series(name, interval, bars, api_key, cache_dir, now_utc):
    """Scarica e UNISCE alla cache: la storia si allunga a ogni esecuzione."""
    sym, exch, _, _ = INSTRUMENTS[name]
    fn = cache_path(cache_dir, name, interval)
    old = None
    if os.path.exists(fn):
        old = pd.read_csv(fn)
        old["time"] = pd.to_datetime(old["time"], utc=True)
    new, msg = (None, "cache (nessuna API key)") if not api_key else fetch_td(sym, exch, interval, bars, api_key)
    if api_key:
        time.sleep(REQUEST_DELAY)
    parts = [d for d in (old, new) if d is not None and not d.empty]
    if not parts:
        return None, msg
    df = pd.concat(parts, ignore_index=True).sort_values("time")
    df = df.drop_duplicates("time", keep="last").reset_index(drop=True)
    df.to_csv(fn, index=False)
    return drop_incomplete(df, interval, now_utc), msg

# =============================================================================
# 3. INDICATORI
# =============================================================================
def rsi_wilder(close, period=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(100)

def atr_series(df, period=14):
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def add_features(df, role):
    df = df.copy()
    df["atr"] = atr_series(df)
    if role == "bias":
        df["ema50"] = df.close.ewm(span=50, adjust=False).mean()
        df["ema200"] = df.close.ewm(span=200, adjust=False).mean()
    if role in ("level", "entry"):
        df["rsi"] = rsi_wilder(df.close)
    if role == "entry":
        df["ema20"] = df.close.ewm(span=20, adjust=False).mean()
    return df

def fractal_flags(high, low, k=2):
    """True dove la candela è un massimo/minimo locale confermato (k candele per lato)."""
    n = len(high)
    fh = np.zeros(n, bool); fl = np.zeros(n, bool)
    for i in range(k, n - k):
        if high[i] >= high[i - k:i].max() and high[i] >= high[i + 1:i + k + 1].max():
            fh[i] = True
        if low[i] <= low[i - k:i].min() and low[i] <= low[i + 1:i + k + 1].min():
            fl[i] = True
    return fh, fl

# =============================================================================
# 4. PRECALCOLO (una volta per strumento: rende il backtest veloce)
# =============================================================================
def precompute_bias(hb, P):
    """Per ogni candela bias: direzione del trend, estremo e partenza dell'impulso."""
    n = len(hb)
    close, high, low = hb.close.values, hb.high.values, hb.low.values
    e50, e200, atr = hb.ema50.values, hb.ema200.values, hb.atr.values
    sb = P["ema_slope_bars"]; lb = P["impulse_lookback"]
    direction = np.zeros(n, np.int8); ext = np.full(n, np.nan); start = np.full(n, np.nan)
    for k in range(max(200, lb, sb), n):
        up = e50[k] > e200[k] and e50[k] > e50[k - sb] and close[k] > e200[k]
        dn = e50[k] < e200[k] and e50[k] < e50[k - sb] and close[k] < e200[k]
        if not (up or dn):
            continue
        a, b = k - lb + 1, k + 1
        if up:
            # FIX v5: estremo = massimo più alto della finestra (anche non confermato)
            j = a + int(np.argmax(high[a:b]))
            if j <= a:
                continue
            s = float(low[a:j].min())
            if low[j:b].min() <= s:          # struttura rotta: impulso annullato
                continue
            direction[k] = 1; ext[k] = high[j]; start[k] = s
        else:
            j = a + int(np.argmin(low[a:b]))
            if j <= a:
                continue
            s = float(high[a:j].max())
            if high[j:b].max() >= s:
                continue
            direction[k] = -1; ext[k] = low[j]; start[k] = s
    return dict(dir=direction, ext=ext, start=start, atr=atr)

def precompute_levels(hl, P):
    """Per ogni candela livelli: elenco di (livello, numero di tocchi) dai pivot recenti."""
    high, low, atr = hl.high.values, hl.low.values, hl.atr.values
    fh, fl = fractal_flags(high, low)
    n = len(high); lb = P["level_lookback"]
    sup, res = [None] * n, [None] * n
    for k in range(lb, n):
        a = k - lb
        tol = P["level_tol_atr"] * atr[k]
        for store, flags, vals in ((sup, fl, low), (res, fh, high)):
            piv = vals[a:k][flags[a:k]]
            out = []
            if len(piv):
                used = np.zeros(len(piv), bool)
                order = np.argsort(piv)
                for idx in order:
                    if used[idx]:
                        continue
                    sel = np.abs(piv - piv[idx]) <= tol
                    used |= sel
                    out.append((float(piv[sel].mean()), int(sel.sum())))
            store[k] = out
    return dict(sup=sup, res=res, atr=atr, rsi=hl.rsi.values)

def map_bars(entry_times, other_times, other_minutes):
    """Per ogni candela d'ingresso, l'indice dell'ultima candela chiusa dell'altro timeframe."""
    closes = other_times + pd.Timedelta(minutes=other_minutes)
    return np.searchsorted(closes.values, entry_times.values, side="right") - 1

# =============================================================================
# 5. CALENDARIO / ESPOSIZIONE
# =============================================================================
def market_open(name, now_rome):
    _, _, _, mkt = INSTRUMENTS[name]
    wd, h = now_rome.weekday(), now_rome.hour + now_rome.minute / 60
    if mkt == "fx":
        return not (wd == 5 or (wd == 6 and h < 23) or (wd == 4 and h >= 23))
    if wd >= 5:
        return False
    s, e = SESSIONS[mkt]
    return s <= h < e

def weekend_block(ts_rome):
    wd, h = ts_rome.weekday(), ts_rome.hour
    return (wd == 4 and h >= 18) or wd >= 5

def exposure_keys(name, direction):
    sym, _, cls, mkt = INSTRUMENTS.get(name, (name, None, "fx_cross", "fx"))
    s = 1 if direction == "buy" else -1
    if "/" in sym:
        a, b = sym.split("/")
        return {a: s, b: -s}
    if cls == "index":
        return {"EQ_EU" if mkt == "europe" else "EQ_US": s}
    return {"EQ_US" if cls == "stock_us" else ("EQ_EU" if cls == "stock_eu" else "EQ_ASIA"): s}

def exposure_ok(name, direction, open_list):
    counts = {}
    for n, d in open_list:
        for k, v in exposure_keys(n, d).items():
            counts[(k, v)] = counts.get((k, v), 0) + 1
    for k, v in exposure_keys(name, direction).items():
        if counts.get((k, v), 0) >= EXPOSURE_LIMIT.get(k, 1):
            return False, f"già esposto su {k} {'long' if v > 0 else 'short'}"
    return True, ""

# =============================================================================
# 6. TRIGGER D'INGRESSO
# =============================================================================
def trigger(mode, he, i, d, level, tol, P):
    """Verifica il trigger sulla candela d'ingresso i. Ritorna (ok, descrizione, riferimento_stop)."""
    o, h, l, c = he.open.values, he.high.values, he.low.values, he.close.values
    a = he.atr.values[i]
    if mode == "zone":
        return True, "ingresso diretto in zona", (l[i] if d == 1 else h[i])
    lookback = min(P["trigger_lookback"], i)
    for j in range(i, i - lookback - 1, -1):
        rg = h[j] - l[j]
        if rg <= 0:
            continue
        body = abs(c[j] - o[j])
        touched = (l[j] <= level + tol) if d == 1 else (h[j] >= level - tol)
        side_ok = (c[j] > level) if d == 1 else (c[j] < level)
        if not (touched and side_ok):
            continue
        wick = (min(o[j], c[j]) - l[j]) if d == 1 else (h[j] - max(o[j], c[j]))
        is_pin = wick >= P["pin_wick"] * rg and body <= P["pin_body"] * rg and rg >= P["pin_min_atr"] * a
        if mode == "pinbar":
            if is_pin:
                return True, "pinbar", (l[j] if d == 1 else h[j])
            continue
        # mode == "reversal": pinbar, engulfing o rientro sopra/sotto il livello
        if is_pin:
            return True, "pinbar", (l[j] if d == 1 else h[j])
        if j > 0 and body >= P["engulf_body"] * rg:
            prev_opposite = (c[j - 1] < o[j - 1]) if d == 1 else (c[j - 1] > o[j - 1])
            engulf = (c[j] > o[j - 1]) if d == 1 else (c[j] < o[j - 1])
            if prev_opposite and engulf:
                return True, "engulfing", (min(l[j], l[j - 1]) if d == 1 else max(h[j], h[j - 1]))
        pierced = (l[j] <= level) if d == 1 else (h[j] >= level)
        if pierced:
            return True, "rientro sul livello", (l[j] if d == 1 else h[j])
    return False, "", None

def rsi_entry_ok(he, i, d, P):
    if not P["rsi_entry_filter"]:
        return True
    r, rp = he.rsi.values[i], he.rsi.values[i - 1]
    lo, hi = P["rsi_entry_long"] if d == 1 else P["rsi_entry_short"]
    rising = r > rp if d == 1 else r < rp
    return lo <= r <= hi and rising

# =============================================================================
# 7. VALUTAZIONE DI UNA CANDELA D'INGRESSO
# =============================================================================
def evaluate(i, pb, pl, he, kb, kl, mode, P, name):
    """Ritorna dict con ok / gate / parametri operativi. Gate in ordine: G1..G7."""
    r = {"ok": False, "gate": ""}
    d = int(pb["dir"][kb])
    if d == 0:
        r["gate"] = "G1 trend bias non definito"; return r
    ext, start, ab = pb["ext"][kb], pb["start"][kb], pb["atr"][kb]
    if not np.isfinite(ext) or not np.isfinite(start):
        r["gate"] = "G2 impulso non identificabile"; return r
    rng = (ext - start) if d == 1 else (start - ext)
    if rng < P["min_impulse_atr"] * ab:
        r["gate"] = "G2 impulso troppo piccolo"; return r
    price = float(he.close.values[i])
    retr = ((ext - price) / rng) if d == 1 else ((price - ext) / rng)
    r.update({"Direzione": "buy" if d == 1 else "sell", "Ritracciamento %": round(retr * 100, 1),
              "ATR bias": ab})
    if not (P["fib_min"] <= retr <= P["fib_max"]):
        r["gate"] = f"G3 ritracciamento fuori fascia ({retr*100:.0f}%)"; return r
    rsi_l = float(pl["rsi"][kl])
    lo, hi = P["rsi_level_long"] if d == 1 else P["rsi_level_short"]
    r["RSI livelli"] = round(rsi_l, 1)
    if not (lo <= rsi_l <= hi):
        r["gate"] = f"G4 RSI timeframe livelli fuori fascia ({rsi_l:.0f})"; return r
    al = pl["atr"][kl]; tol = P["level_tol_atr"] * al
    cands = (pl["sup"][kl] if d == 1 else pl["res"][kl]) or []
    best, best_n = None, 0
    for lvl, nt in cands:
        near = abs(lvl - price) <= P["level_near_atr"] * al + tol
        side = (lvl <= price + tol) if d == 1 else (lvl >= price - tol)
        if near and side and nt > best_n:
            best, best_n = lvl, nt
    if best is None or best_n < P["min_touches"]:
        r["gate"] = "G5 nessun livello con 2 tocchi vicino al prezzo"; return r
    r.update({"Livello": round(best, 6), "Tocchi livello": best_n})
    ok_trig, trig_name, ref = trigger(mode, he, i, d, best, tol, P)
    if not ok_trig:
        r["gate"] = "G6 nessun trigger d'ingresso"; return r
    if not rsi_entry_ok(he, i, d, P):
        r["gate"] = "G6 RSI candela d'ingresso non conferma"; return r
    r["Trigger"] = trig_name
    entry = price
    if d == 1:
        stop = min(min(ref, best) - P["stop_buffer_atr_level"] * al, entry - P["min_stop_atr_bias"] * ab)
        risk = entry - stop
        tp = min(entry + P["rr_target"] * risk, ext - 0.1 * ab)
    else:
        stop = max(max(ref, best) + P["stop_buffer_atr_level"] * al, entry + P["min_stop_atr_bias"] * ab)
        risk = stop - entry
        tp = max(entry - P["rr_target"] * risk, ext + 0.1 * ab)
    if risk <= 0:
        r["gate"] = "G7 stop non valido"; return r
    rr = abs(tp - entry) / risk
    cls = INSTRUMENTS[name][2]
    spread_R = SPREAD_PCT[cls] / 100 * entry / risk
    r.update({"Entry": round(entry, 6), "SL": round(stop, 6), "TP": round(tp, 6), "R:R": round(rr, 2),
              "Stop %": round(risk / entry * 100, 3), "Spread/Rischio %": round(spread_R * 100, 1),
              "Rischio €": RISK_EUR, "Esposizione €": round(RISK_EUR / (risk / entry), 0),
              "_ext": float(ext)})
    if rr < P["rr_min"]:
        r["gate"] = f"G7 spazio insufficiente fino all'estremo (R:R {rr:.2f})"; return r
    if spread_R > P["max_spread_R"]:
        r["gate"] = f"G7 spread troppo alto rispetto allo stop ({spread_R*100:.0f}%)"; return r
    # Bonus informativi (NON modificano la size): servono solo a essere misurati
    bonus, det = 0, []
    if best_n >= 3:
        bonus += 1; det.append("livello 3+ tocchi")
    if trig_name == "pinbar":
        bonus += 1; det.append("pinbar")
    if abs(entry - float(he.ema20.values[i])) <= 0.5 * float(he.atr.values[i]):
        bonus += 1; det.append("su EMA20 ingresso")
    r.update({"ok": True, "gate": "tutti i gate superati", "Bonus": bonus,
              "Bonus dettaglio": ", ".join(det) or "-",
              "Setup": f"{'Supporto' if d == 1 else 'Resistenza'} {best:.5g} ({best_n} tocchi), "
                       f"ritracc. {retr*100:.0f}%, trigger: {trig_name}"})
    return r

# =============================================================================
# 8. LOGICA v3 (riproduzione per il confronto)
# =============================================================================
def _v3_ema(closes, period):
    arr = np.asarray(closes, float)
    if len(arr) < period:
        return None
    k = 2.0 / (period + 1); e = float(arr[:period].mean())
    for p in arr[period:]:
        e = p * k + e * (1 - k)
    return e

def _v3_trend(df, n=8):
    lows, highs = df.low.values[-n:], df.high.values[-n:]
    hl = sum(lows[i] > lows[i - 1] for i in range(1, len(lows)))
    lh = sum(highs[i] < highs[i - 1] for i in range(1, len(highs)))
    return "Rialzista" if hl >= 5 else "Ribassista" if lh >= 5 else "Laterale"

def evaluate_v3(hb, hl, he, kb, kl, i, name):
    """Logica del notebook v3, bug inclusi (tolleranza Fib 5%, stop a 0,1%)."""
    r = {"ok": False, "gate": "v3"}
    h4 = hb.iloc[max(0, kb - 249):kb + 1]; h1 = hl.iloc[max(0, kl - 99):kl + 1]
    m15 = he.iloc[max(0, i - 29):i + 1]
    if len(h4) < 60 or len(h1) < 60 or len(m15) < 20:
        r["gate"] = "v3 dati insufficienti"; return r
    t4 = _v3_trend(h4); c4 = h4.close.values
    e50 = _v3_ema(c4, 50); e200 = _v3_ema(c4, 200) if len(c4) >= 200 else None
    if e50 is None:
        r["gate"] = "v3 dati insufficienti"; return r
    p = c4[-1]
    al4 = ("long" if p > e50 > (e200 or -np.inf) else "short" if p < e50 < (e200 or np.inf) else "mista") \
        if e200 is not None else ("long" if p > e50 else "short")
    if t4 == "Laterale" and al4 == "mista":
        r["gate"] = "v3 bias H4 non definito"; return r
    bias = t4 if t4 != "Laterale" else ("Rialzista" if "long" in al4 else "Ribassista")
    t1 = _v3_trend(h1)
    if t1 == "Laterale" or (bias == "Rialzista") != (t1 == "Rialzista"):
        r["gate"] = "v3 conflitto o H1 laterale"; return r
    trend = t1
    c1 = h1.close.values; e20_1 = _v3_ema(c1, 20); e50_1 = _v3_ema(c1, 50)
    al1 = "long" if c1[-1] > e20_1 > e50_1 else "short" if c1[-1] < e20_1 < e50_1 else "mista"
    piv = h1.tail(30).low.values if trend == "Rialzista" else h1.tail(30).high.values
    best, bn = None, 0
    for q in piv:
        cl = piv[np.abs(piv - q) / abs(q) <= 0.002]
        if len(cl) > bn:
            bn, best = len(cl), float(cl.mean())
    if best is None or bn < 2:
        r["gate"] = "v3 nessun livello H1"; return r
    b = m15.iloc[-1]
    o, h, l, c = b.open, b.high, b.low, b.close
    rg = h - l
    if rg <= 0:
        r["gate"] = "v3 candela nulla"; return r
    near = abs((l if trend == "Rialzista" else h) - best) / abs(best) <= 0.003
    body = abs(c - o)
    pin = near and (((min(o, c) - l) >= 0.6 * rg) if trend == "Rialzista" else ((h - max(o, c)) >= 0.6 * rg)) \
        and body <= 0.3 * rg
    if not pin:
        r["gate"] = "v3 nessuna pinbar M15"; return r
    rsi15 = float(rsi_wilder(m15.close).iloc[-1])
    rsi_ok = 25 <= rsi15 <= 75 and (rsi15 <= 70 if trend == "Rialzista" else rsi15 >= 30)
    if trend == "Rialzista":
        sl = min(l, best) * 0.999; tp = c + 2.5 * (c - sl)
    else:
        sl = max(h, best) * 1.001; tp = c - 2.5 * (sl - c)
    dirok = lambda a: ("long" in a and trend == "Rialzista") or ("short" in a and trend == "Ribassista")
    e20_15 = _v3_ema(m15.close.values, 20)
    score = int(dirok(al4)) + int(dirok(al1)) + int(rsi_ok) + \
        int(e20_15 is not None and abs(c - e20_15) / e20_15 <= 0.005)
    if score < 3:
        r["gate"] = "v3 score < 3"; return r
    risk = abs(c - sl)
    if risk <= 0:
        r["gate"] = "v3 stop non valido"; return r
    r.update(ok=True, gate="tutti i gate superati", Direzione="buy" if trend == "Rialzista" else "sell",
             Entry=round(c, 6), SL=round(sl, 6), TP=round(tp, 6), **{"R:R": 2.5},
             **{"Stop %": round(risk / c * 100, 3)}, Bonus=score, Trigger="pinbar (v3)",
             **{"Ritracciamento %": np.nan, "Tocchi livello": bn, "_ext": np.nan})
    return r

# =============================================================================
# 9. SIMULAZIONE DEL TRADE
# =============================================================================
def simulate(he, i, sig, P, cls, time_stop_hours):
    d = 1 if sig["Direzione"] == "buy" else -1
    entry, sl, tp = sig["Entry"], sig["SL"], sig["TP"]
    risk = abs(entry - sl)
    cost_R = SPREAD_PCT[cls] / 100 * entry / risk
    deadline = pd.Timestamp(he.time.values[i]) + pd.Timedelta(hours=time_stop_hours)
    H, L, O, C, T = he.high.values, he.low.values, he.open.values, he.close.values, he.time.values
    for j in range(i + 1, len(he)):
        if d == 1:
            if L[j] <= sl:
                return j, (min(O[j], sl) - entry) / risk - cost_R, "SL"
            if H[j] >= tp:
                return j, (tp - entry) / risk - cost_R, "TP"
        else:
            if H[j] >= sl:
                return j, (entry - max(O[j], sl)) / risk - cost_R, "SL"
            if L[j] <= tp:
                return j, (entry - tp) / risk - cost_R, "TP"
        tj = pd.Timestamp(T[j])
        if tj >= deadline:
            return j, d * (C[j] - entry) / risk - cost_R, "TIME"
        if P["close_before_weekend"]:
            tr = tj.tz_localize("UTC").tz_convert(ROME)
            if tr.weekday() == 4 and tr.hour >= 21:
                return j, d * (C[j] - entry) / risk - cost_R, "WEEKEND"
    return len(he) - 1, d * (C[-1] - entry) / risk - cost_R, "APERTO A FINE DATI"

# =============================================================================
# 10. BACKTEST
# =============================================================================
def backtest_symbol(name, series, preset, gate_count, pb=None, pl=None):
    tfset_name, mode, over = PRESET_DEFS[preset]
    TF = TF_SETS[tfset_name]; P = dict(PARAMS, **over)
    hb = add_features(series[TF["bias"]], "bias")
    hl = add_features(series[TF["level"]], "level")
    he = add_features(series[TF["entry"]], "entry")
    if min(len(hb), len(hl), len(he)) < 120:
        return []
    kb_map = map_bars(he.time, hb.time, INTERVAL_MIN[TF["bias"]])
    kl_map = map_bars(he.time, hl.time, INTERVAL_MIN[TF["level"]])
    times_rome = he.time.dt.tz_convert(ROME)
    cls = INSTRUMENTS[name][2]
    trades, busy_until, last_ext = [], -1, None

    if mode == "random":                      # preset di controllo: ingressi a caso
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 31) + RANDOM_SEED)
        n = len(he)
        idxs = rng.choice(np.arange(60, n - 10), size=min(RANDOM_TRADES_PER_SYMBOL, max(1, n - 80)),
                          replace=False)
        for i in sorted(idxs):
            kb = kb_map[i]
            if kb < 200:
                continue
            if P["weekend_filter"] and weekend_block(times_rome.iloc[i]):
                continue
            d = 1 if rng.random() < 0.5 else -1
            entry = float(he.close.values[i]); ab = float(hb.atr.values[kb])
            risk = P["min_stop_atr_bias"] * ab
            sig = {"Direzione": "buy" if d == 1 else "sell", "Entry": entry,
                   "SL": entry - d * risk, "TP": entry + d * P["rr_target"] * risk,
                   "R:R": P["rr_target"], "Stop %": round(risk / entry * 100, 3),
                   "Bonus": 0, "Trigger": "casuale", "Ritracciamento %": np.nan,
                   "Tocchi livello": np.nan}
            j, R, why = simulate(he, i, sig, P, cls, TF["time_stop_hours"])
            trades.append(_trade_row(preset, name, sig, he, i, j, R, why, times_rome))
        return trades

    for i in range(TF["entry_min"], len(he)):
        if i <= busy_until:
            continue
        kb, kl = kb_map[i], kl_map[i]
        if kb < 200 or kl < P["level_lookback"] + 5:
            continue
        if P["weekend_filter"] and weekend_block(times_rome.iloc[i]):
            continue
        if mode == "v3":
            sig = evaluate_v3(hb, hl, he, kb, kl, i, name)
        else:
            sig = evaluate(i, pb, pl, he, kb, kl, mode, P, name)
        if not sig["ok"]:
            g = sig["gate"].split(" (")[0]
            gate_count[g] = gate_count.get(g, 0) + 1
            continue
        if P["one_trade_per_impulse"] and last_ext is not None and np.isfinite(sig.get("_ext", np.nan)) \
                and np.isclose(sig["_ext"], last_ext, rtol=1e-9, atol=0):
            gate_count["scartato: impulso già operato"] = gate_count.get("scartato: impulso già operato", 0) + 1
            continue
        j, R, why = simulate(he, i, sig, P, cls, TF["time_stop_hours"])
        busy_until = j
        last_ext = sig.get("_ext", None)
        trades.append(_trade_row(preset, name, sig, he, i, j, R, why, times_rome))
    return trades

def _trade_row(preset, name, sig, he, i, j, R, why, times_rome):
    return {"Preset": preset, "Strumento": name, "Direzione": sig["Direzione"],
            "Ingresso (Roma)": times_rome.iloc[i].strftime("%Y-%m-%d %H:%M"),
            "Uscita (Roma)": times_rome.iloc[j].strftime("%Y-%m-%d %H:%M"),
            "_t_in": he.time.values[i], "_t_out": he.time.values[j],
            "Entry": sig["Entry"], "SL": sig["SL"], "TP": sig["TP"],
            "R:R": sig.get("R:R"), "Stop %": sig.get("Stop %"),
            "Ritracciamento %": sig.get("Ritracciamento %"), "Trigger": sig.get("Trigger"),
            "Tocchi livello": sig.get("Tocchi livello"), "Bonus": sig.get("Bonus"),
            "Giorni in posizione": round((pd.Timestamp(he.time.values[j]) -
                                          pd.Timestamp(he.time.values[i])).total_seconds() / 86400, 2),
            "Uscita per": why, "R": round(R, 3)}

def apply_exposure(trades, enabled):
    if not enabled:
        return trades
    kept, openl = [], []
    for t in sorted(trades, key=lambda x: x["_t_in"]):
        openl = [o for o in openl if o["_t_out"] > t["_t_in"]]
        ok, _ = exposure_ok(t["Strumento"], t["Direzione"],
                            [(o["Strumento"], o["Direzione"]) for o in openl])
        if ok:
            kept.append(t); openl.append(t)
    return kept

def stats(tr, label=""):
    if not tr:
        return {"Periodo": label, "Trade": 0, "Giudizio": "nessun trade"}
    tr = sorted(tr, key=lambda x: x["_t_out"])
    R = pd.Series([t["R"] for t in tr])
    eq = R.cumsum(); dd = float((eq - eq.cummax()).min())
    streak = mx = 0
    for x in R:
        streak = streak + 1 if x <= 0 else 0; mx = max(mx, streak)
    w, l = R[R > 0], R[R <= 0]
    span_days = (pd.Timestamp(tr[-1]["_t_out"]) - pd.Timestamp(tr[0]["_t_in"])).days or 1
    pf = round(w.sum() / abs(l.sum()), 2) if len(l) and l.sum() != 0 else None
    be = round(abs(l.mean()) / (w.mean() + abs(l.mean())) * 100, 1) if len(w) and len(l) else None
    n = len(R)
    giudizio = ("campione sufficiente" if n >= MIN_TRADES_OK else
                "campione scarso" if n >= MIN_TRADES_NOISE else "rumore: non concludere")
    return {"Periodo": label, "Trade": n, "Dal": f"{pd.Timestamp(tr[0]['_t_in']):%Y-%m-%d}",
            "Al": f"{pd.Timestamp(tr[-1]['_t_out']):%Y-%m-%d}",
            "Trade/mese": round(n / (span_days / 30.4), 1),
            "Vincenti %": round((R > 0).mean() * 100, 1),
            "Win rate di pareggio %": be,
            "R medio (expectancy)": round(R.mean(), 3), "R totale": round(R.sum(), 2),
            "R medio vincenti": round(w.mean(), 2) if len(w) else None,
            "R medio perdenti": round(l.mean(), 2) if len(l) else None,
            "Profit factor": pf, "Max drawdown (R)": round(dd, 2),
            "Max perdite consecutive": mx,
            "Giorni medi in posizione": round(np.mean([t["Giorni in posizione"] for t in tr]), 1),
            "Usciti per time stop %": round(np.mean([t["Uscita per"] == "TIME" for t in tr]) * 100, 1),
            "Giudizio": giudizio}

def bucket_table(df, col, bins=None, labels=None):
    if df.empty or col not in df:
        return pd.DataFrame()
    d = df.copy()
    key = col
    if bins is not None:
        d["_b"] = pd.cut(d[col], bins=bins, labels=labels); key = "_b"
    g = d.groupby(["Preset", key], observed=True).R.agg(["count", "mean", "sum"]).round(3)
    return g.rename(columns={"count": "Trade", "mean": "R medio", "sum": "R totale"})

pre_b, pre_l = {}, {}   # cache dei precalcoli per strumento (riusata tra i preset)

def run_backtest(api_key, out_dir):
    now_utc = pd.Timestamp.now(tz="UTC")
    cache = os.path.join(out_dir, "cache")
    needed = sorted({TF_SETS[PRESET_DEFS[p][0]][role] for p in BACKTEST_PRESETS for role in
                     ("bias", "level", "entry")}, key=lambda x: -INTERVAL_MIN[x])
    log("=" * 70)
    log("  BACKTEST v5 — confronto preset")
    log(f"  Preset: {', '.join(BACKTEST_PRESETS)}")
    log(f"  Timeframe da scaricare: {', '.join(needed)}")
    log("=" * 70)
    data, dataload = {}, []
    for name in BACKTEST_SYMBOLS:
        series, ok = {}, True
        for tf in needed:
            df, msg = load_series(name, tf, 5000, api_key, cache, now_utc)
            row = {"Strumento": name, "Timeframe": tf,
                   "Candele": 0 if df is None else len(df), "Messaggio": msg}
            if df is not None and len(df):
                row.update({"Dal": f"{df.time.iloc[0]:%Y-%m-%d}", "Al": f"{df.time.iloc[-1]:%Y-%m-%d}"})
            dataload.append(row)
            if not ohlc_valid(df, 120):
                ok = False
                log(f"  {name:16s} {tf:6s} NON UTILIZZABILE ({msg})")
                continue
            series[tf] = df
            log(f"  {name:16s} {tf:6s} {len(df):5d} candele  {df.time.iloc[0]:%Y-%m-%d} -> {df.time.iloc[-1]:%Y-%m-%d}")
        if series:
            data[name] = series
    if not data:
        raise SystemExit(
            "ERRORE: nessun dato disponibile.\n"
            "Cause tipiche, in ordine di probabilità:\n"
            "  1) API key Twelvedata assente o non leggibile (secret non impostato o con nome diverso);\n"
            "  2) cache vuota al primo avvio e nessuna chiave con cui scaricare;\n"
            "  3) simboli non coperti dal piano sottoscritto.\n"
            "Controlla il foglio/log 'Dati caricati' e il messaggio di ciascun timeframe.")

    # precalcolo per i preset non-v3 (una volta per strumento e per set di timeframe)
    all_tr, summary, gate_rows = [], [], []
    for preset in BACKTEST_PRESETS:
        tfset_name, mode, over = PRESET_DEFS[preset]
        TF = TF_SETS[tfset_name]; P = dict(PARAMS, **over)
        gate_count = {}
        tr = []
        log(f"\n[{preset}] ...")
        for name, series in data.items():
            if any(TF[r] not in series for r in ("bias", "level", "entry")):
                continue
            pb = pl = None
            if mode not in ("v3", "random"):
                key = (name, tfset_name)
                if key not in pre_b:
                    log(f"    precalcolo {name} ({tfset_name}) ...")
                    pre_b[key] = precompute_bias(add_features(series[TF["bias"]], "bias"), P)
                    pre_l[key] = precompute_levels(add_features(series[TF["level"]], "level"), P)
                pb, pl = pre_b[key], pre_l[key]
            tr += backtest_symbol(name, series, preset, gate_count, pb, pl)
        tr = apply_exposure(tr, P["exposure_filter"])
        for t in tr:
            all_tr.append(t)
        if tr:
            t0 = min(pd.Timestamp(t["_t_in"]) for t in tr); t1 = max(pd.Timestamp(t["_t_out"]) for t in tr)
            cut = t0 + (t1 - t0) * OOS_SPLIT
            tis = [t for t in tr if pd.Timestamp(t["_t_in"]) <= cut]
            toos = [t for t in tr if pd.Timestamp(t["_t_in"]) > cut]
        else:
            tis = toos = []
        for label, subset in (("tutto", tr), (f"in-sample (primi {int(OOS_SPLIT*100)}%)", tis),
                              ("out-of-sample (ultimi 30%)", toos)):
            s = stats(subset, label); s["Preset"] = preset; summary.append(s)
        for g, n in sorted(gate_count.items(), key=lambda x: -x[1]):
            gate_rows.append({"Preset": preset, "Motivo di scarto": g, "Candele d'ingresso": n})
        s0 = [s for s in summary if s["Preset"] == preset and s["Periodo"] == "tutto"][0]
        log(f"    trade {s0['Trade']} | R medio {s0.get('R medio (expectancy)')} | "
            f"R totale {s0.get('R totale')} | {s0['Giudizio']}")

    df_tr = pd.DataFrame(all_tr)
    sm = pd.DataFrame(summary)
    cols = ["Preset", "Periodo", "Trade", "Dal", "Al", "Trade/mese", "Vincenti %", "Win rate di pareggio %",
            "R medio (expectancy)", "R totale", "R medio vincenti", "R medio perdenti",
            "Profit factor", "Max drawdown (R)", "Max perdite consecutive",
            "Giorni medi in posizione", "Usciti per time stop %", "Giudizio"]
    sm = sm[[c for c in cols if c in sm.columns]]
    log("\n" + sm[sm.Periodo == "tutto"].to_string(index=False))

    note = pd.DataFrame({"Come leggere il report": [
        "R = multiplo del rischio: -1R = stop pieno, +2,5R = target pieno. Expectancy = R medio per trade.",
        "'Win rate di pareggio %' = percentuale minima di vincenti per andare in pari con quel rapporto "
        "tra vincite e perdite medie. Se 'Vincenti %' è sotto, il preset perde.",
        f"Giudizio: sotto {MIN_TRADES_NOISE} trade è rumore; da {MIN_TRADES_OK} si può iniziare a concludere.",
        "In-sample / out-of-sample: il periodo è diviso in due. Un preset che va bene solo nella prima "
        "parte è adattato al campione, non valido.",
        "'controllo_casuale' entra a caso con lo stesso stop, lo stesso target e gli stessi costi: è il "
        "metro di paragone. Un preset che non lo batte non ha alcun vantaggio.",
        "'v3_originale' include il difetto noto (forex considerato aperto nel weekend); "
        "'v3_con_filtro_weekend' è il confronto onesto.",
        "I conteggi del foglio 'Motivi di scarto' sono CANDELE, non setup indipendenti.",
        "Fogli 'Per fascia Fibonacci' / 'Per trigger' / 'Per bonus': servono a vedere DOVE sta l'eventuale "
        "vantaggio. Attenzione: più confronti si fanno, più è facile trovare un risultato casuale. "
        "Una differenza va considerata reale solo se regge anche out-of-sample.",
        "Spread: stime in SPREAD_PCT, da verificare sulla piattaforma. Nessuno slippage oltre ai gap.",
        "Candela con stop e target nella stessa barra = conteggiata come stop (prudente).",
        "La cache in cache/ si allunga a ogni esecuzione: rilanciare periodicamente aumenta la storia.",
    ]})
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    path = os.path.join(out_dir, f"Backtest_v5_{ts}.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        sm.to_excel(xw, sheet_name="Riepilogo", index=False)
        pd.DataFrame(dataload).to_excel(xw, sheet_name="Dati caricati", index=False)
        if not df_tr.empty:
            bucket_table(df_tr, "Strumento").to_excel(xw, sheet_name="Per strumento")
            bucket_table(df_tr, "Ritracciamento %", bins=[0, 38.2, 50, 61.8, 78.6, 200],
                         labels=["<38,2%", "38,2-50%", "50-61,8%", "61,8-78,6%", ">78,6%"]
                         ).to_excel(xw, sheet_name="Per fascia Fibonacci")
            bucket_table(df_tr, "Trigger").to_excel(xw, sheet_name="Per trigger")
            bucket_table(df_tr, "Bonus").to_excel(xw, sheet_name="Per bonus")
            bucket_table(df_tr, "Uscita per").to_excel(xw, sheet_name="Per tipo di uscita")
            df_tr.drop(columns=["_t_in", "_t_out"]).to_excel(xw, sheet_name="Trade", index=False)
        pd.DataFrame(gate_rows).to_excel(xw, sheet_name="Motivi di scarto", index=False)
        note.to_excel(xw, sheet_name="Come leggere", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(
                    70, max(10, max(len(str(c.value or "")) for c in col[:60]) + 2))
    log(f"\nFile: {path}")
    log("Ricorda: risultati storici e in parte in-sample. Non sono una previsione.")

# =============================================================================
# 11. SCANNER
# =============================================================================
SCAN_COLS = ["Strumento", "Esito", "Direzione", "Entry", "SL", "TP", "R:R", "Stop %", "Rischio €",
             "Esposizione €", "Ritracciamento %", "Trigger", "Livello", "Tocchi livello",
             "RSI livelli", "Bonus", "Bonus dettaglio", "Setup", "Time stop (Roma)",
             "Spread/Rischio %", "Set timeframe", "Modalità ingresso", "Data e ora analisi"]

def run_scan(api_key, out_dir):
    tfset_name = TF_SET_DEFAULT; mode = ENTRY_MODE_DEFAULT
    TF = TF_SETS[tfset_name]
    P = dict(PARAMS, **(dict(rsi_entry_filter=True, pin_min_atr=0.8) if mode == "pinbar" else {}))
    now_utc = pd.Timestamp.now(tz="UTC"); now_rome = now_utc.tz_convert(ROME)
    cache = os.path.join(out_dir, "cache")
    log("=" * 70)
    log(f"  SCANNER v5 — {now_rome:%Y-%m-%d %H:%M} (Roma)")
    log(f"  Set timeframe: {tfset_name} (bias {TF['bias']} / livelli {TF['level']} / ingresso {TF['entry']})")
    log(f"  Modalità ingresso: {mode}")
    log("=" * 70)
    if weekend_block(now_rome):
        log("\n(!) Finestra weekend: nessun nuovo ingresso, analisi solo informativa.")
    open_pos, op_path = [], os.path.join(out_dir, "posizioni_aperte.csv")
    if os.path.exists(op_path):
        try:
            op = pd.read_csv(op_path)
            open_pos = [(str(a).strip(), str(b).strip().lower()) for a, b in zip(op["Strumento"], op["Direzione"])]
            log(f"Posizioni aperte lette: {len(open_pos)}")
        except Exception as e:
            log(f"(!) posizioni_aperte.csv illeggibile: {e}")

    rows = []
    for idx, name in enumerate(INSTRUMENTS, 1):
        base = {"Strumento": name, "Data e ora analisi": f"{now_rome:%Y-%m-%d %H:%M}",
                "Set timeframe": tfset_name, "Modalità ingresso": mode}
        log(f"  [{idx:02d}/{len(INSTRUMENTS)}] {name:<22}")
        if not market_open(name, now_rome):
            rows.append({**base, "Esito": "MERCATO CHIUSO"}); continue
        series, err = {}, None
        for role, bars in zip(("bias", "level", "entry"), TF["bars"]):
            df, msg = load_series(name, TF[role], bars, api_key, cache, now_utc)
            if not ohlc_valid(df, 230 if role == "bias" else 120):
                err = f"DATI {TF[role]} NON UTILIZZABILI ({msg})"; break
            series[role] = df
        if err:
            rows.append({**base, "Esito": err}); continue
        hb = add_features(series["bias"], "bias")
        hl = add_features(series["level"], "level")
        he = add_features(series["entry"], "entry")
        pre_b[name] = precompute_bias(hb, P); pre_l[name] = precompute_levels(hl, P)
        i = len(he) - 1
        kb = int(map_bars(he.time, hb.time, INTERVAL_MIN[TF["bias"]])[i])
        kl = int(map_bars(he.time, hl.time, INTERVAL_MIN[TF["level"]])[i])
        res = evaluate(i, pre_b[name], pre_l[name], he, kb, kl, mode, P, name)
        row = {**base, **{k: v for k, v in res.items() if not k.startswith("_") and k not in ("ok", "gate")}}
        row["Esito"] = res["gate"]
        row["_ok"] = res["ok"]
        if res["ok"]:
            row["Time stop (Roma)"] = f"{now_rome + timedelta(hours=TF['time_stop_hours']):%Y-%m-%d %H:%M}"
        rows.append(row)
        log(f"        -> {row['Esito']}")

    sig = sorted([r for r in rows if r.get("_ok")], key=lambda r: -(r.get("Bonus") or 0))
    accepted = list(open_pos)
    for r in sig:
        if weekend_block(now_rome):
            r["Esito"] = "SCARTATO: finestra weekend"; r["_ok"] = False; continue
        ok, why = exposure_ok(r["Strumento"], r["Direzione"], accepted)
        if not ok:
            r["Esito"] = f"SCARTATO: {why}"; r["_ok"] = False; continue
        accepted.append((r["Strumento"], r["Direzione"]))

    df = pd.DataFrame(rows)
    for c in SCAN_COLS:
        if c not in df.columns:
            df[c] = ""
    ok_df = df[df.get("_ok", False) == True][SCAN_COLS]
    legenda = pd.DataFrame({"Voce": ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "Filtri finali",
                                     "Bonus", "Esposizione €", "Time stop", "AVVERTENZA"],
                            "Significato": [
        "Trend sul timeframe bias: EMA50/EMA200 allineate, EMA50 inclinata, prezzo dal lato giusto",
        f"Impulso ancorato all'estremo del periodo, ampiezza >= {PARAMS['min_impulse_atr']} ATR, struttura intatta",
        f"Ritracciamento tra {PARAMS['fib_min']*100:.1f}% e {PARAMS['fib_max']*100:.1f}% (valore effettivo in colonna)",
        "RSI del timeframe livelli nella fascia ammessa",
        "Supporto/resistenza con almeno 2 tocchi entro 1 ATR dal prezzo",
        "Trigger secondo la modalità scelta (zone / reversal / pinbar)",
        "Stop >= 1 ATR bias, target prima dell'estremo dell'impulso, R:R minimo, spread accettabile",
        "Niente ingressi ven 18:00–dom; max 1 posizione per valuta nella stessa direzione",
        "Solo informativi: NON cambiano la size finché il backtest non dimostra che valgono qualcosa",
        "Controvalore della posizione (non il margine) per rischiare esattamente 'Rischio €'",
        "Chiusura forzata se né TP né SL vengono raggiunti entro la scadenza",
        "Parametri non validati: eseguire prima MODE=backtest e verificare anche l'out-of-sample"]})
    # strumenti che stanno avvicinandosi alla fascia: utili da tenere d'occhio quando non ci sono segnali
    watch = df.copy()
    watch["_r"] = pd.to_numeric(watch["Ritracciamento %"], errors="coerce")
    watch = watch[(watch.get("_ok", False) != True) & watch._r.between(25, 95)] \
        .sort_values("_r")[["Strumento", "Direzione", "Ritracciamento %", "Esito", "RSI livelli",
                            "Livello", "Set timeframe"]]
    ts = now_rome.strftime("%Y%m%d_%H%M")
    path = os.path.join(out_dir, f"Scan_v5_{ts}.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        ok_df.to_excel(xw, sheet_name="Segnali operativi", index=False)
        watch.to_excel(xw, sheet_name="Da tenere d'occhio", index=False)
        df[SCAN_COLS].to_excel(xw, sheet_name="Tutti gli strumenti", index=False)
        legenda.to_excel(xw, sheet_name="Legenda", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(
                    70, max(10, max(len(str(c.value or "")) for c in col[:60]) + 2))
    log(f"\nSegnali operativi: {len(ok_df)} | Da tenere d'occhio: {len(watch)} | File: {path}")
    if len(ok_df) == 0:
        log("Nessun segnale: con regole selettive è normale. Il foglio \"Da tenere d'occhio\" mostra "
            "gli strumenti che stanno rientrando nella fascia.")

# =============================================================================
def main():
    api_key = get_api_key(); out_dir = get_out_dir()
    cache_files = [f for f in os.listdir(os.path.join(out_dir, "cache")) if f.endswith(".csv")]
    log(f"Chiave API: {'presente (' + str(len(api_key)) + ' caratteri)' if api_key else 'ASSENTE'} | "
        f"file in cache: {len(cache_files)} | modalità: {MODE} | cartella output: {out_dir}")
    if not api_key:
        if MODE == "scan" or not cache_files:
            raise SystemExit(
                "ERRORE: API key Twelvedata assente.\n"
                "Su GitHub Actions: Settings > Secrets and variables > Actions > scheda 'Secrets'.\n"
                "Il nome del secret deve coincidere con quello indicato nel file del workflow.\n"
                "Attenzione: un valore inserito nella scheda 'Variables' NON viene letto come secret.")
        log("(!) API key assente: proseguo con i soli dati già in cache.")
    (run_scan if MODE == "scan" else run_backtest)(api_key, out_dir)

if __name__ == "__main__":
    main()
