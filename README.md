# Crypto Backtesting Tool

Web app lokal untuk **backtesting strategi trading crypto** di Binance Futures. Dibangun dengan [`backtesting.py`](https://github.com/kernc/backtesting.py) sebagai engine, data historis dari Binance via CCXT, dan antarmuka Next.js dengan chart interaktif.

**Tujuan utama:** mem-backtest strategi Stochastic state-machine yang sudah berjalan sebagai live bot, tanpa perlu menjalankan bot yang sesungguhnya.

> **Disclaimer:** Ini adalah personal tool untuk keperluan riset. Bukan untuk live/paper trading, bukan multi-user SaaS, dan tidak dirancang untuk deployment publik.

---

## Fitur

- **Backtest multi-strategi** — Stochastic State Machine, RSI, SMA Cross, Supertrend
- **Data otomatis dari Binance** — fetch OHLCV via CCXT, semua pair Futures USDT-M
- **Equity curve & trade log** — chart interaktif dengan overlay indikator
- **Parameter real-time** — semua param strategi dirender otomatis di UI, tidak perlu edit kode
- **Risk management lengkap** — SL, TP, trailing tiered milestone, auto-reverse on SL

---

## Tech Stack

| Layer | Teknologi |
|---|---|
| Backend | Python 3.11+, FastAPI, `backtesting.py`, `ccxt`, `pandas` |
| Frontend | Next.js 14 (App Router), TypeScript, Tailwind CSS, `lightweight-charts` |
| Package manager | `uv` (backend), `npm` (frontend) |

---

## Struktur Proyek

```
backtesting-trade/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routers (backtest, data, strategies)
│   │   ├── core/
│   │   │   ├── data_source.py       # Fetch OHLCV dari Binance via CCXT
│   │   │   ├── indicators.py        # Shared indicator helpers
│   │   │   ├── runner.py            # Backtesting.py runner + hasil serializer
│   │   │   └── strategies/
│   │   │       ├── stoch_state_machine.py   # Strategi utama
│   │   │       ├── rsi.py
│   │   │       ├── sma_cross.py
│   │   │       └── supertrend.py
│   │   └── main.py
│   ├── tests/
│   └── pyproject.toml
├── frontend/
│   ├── app/                # Next.js App Router
│   ├── components/         # DataPicker, StrategyPicker, Chart, StatsTable
│   └── lib/api.ts          # REST client ke backend
├── STRATEGY.md             # Spesifikasi lengkap strategi Stochastic
└── CLAUDE.md               # Panduan untuk kontributor / AI assistant
```

---

## Cara Menjalankan

### Prasyarat

- Python 3.11+
- Node.js 18+
- [`uv`](https://github.com/astral-sh/uv) — package manager Python

### 1. Backend

```bash
cd backend
uv sync                                        # install dependencies
cp .env.example .env                           # (opsional) tambahkan BINANCE_API_KEY / SECRET
uvicorn app.main:app --reload                  # jalankan di port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                                    # jalankan di port 3000
```

Buka browser di `http://localhost:3000`.

---

## Strategi

### Stochastic State Machine (Utama)

Direplikasi dari bot live yang berjalan di Binance Futures. Dokumentasi lengkap ada di [STRATEGY.md](STRATEGY.md).

**Ringkasan:**
- **Indikator:** Stochastic Oscillator (%K dan %D) dua tahap
- **State machine:** 5 state — `IDLE → LONG_ARMED → IN_LONG` / `IDLE → SHORT_ARMED → IN_SHORT`
- **Entry LONG:** cross up %K/%D di zona oversold (< 20), konfirmasi breakout keluar zona
- **Entry SHORT:** cross down %K/%D di zona overbought (> 80), konfirmasi breakdown keluar zona
- **Risk:** SL%, TP%, trailing tiered milestone, auto-reverse saat SL hit

**Parameter yang bisa dikonfigurasi:**

| Parameter | Default | Keterangan |
|---|---|---|
| `stoch_k` | 14 | Lookback periode %K |
| `stoch_d` | 3 | Smoothing %D |
| `stoch_smooth` | 3 | Smoothing %K |
| `sl_pct` | 2.0 % | Jarak Stop Loss |
| `tp_pct` | 3.0 % | Jarak Take Profit |
| `leverage` | 5x | Leverage (sizing notional) |
| `trailing_enabled` | false | Aktifkan trailing stop |
| `trailing_trigger_pct` | 1.0 % | Profit minimum untuk aktifkan trailing |
| `trailing_offset_pct` | 0.5 % | Step antar milestone trailing |
| `auto_reverse_enabled` | false | Flip sisi otomatis saat SL hit |

### Strategi Lain

| Strategi | Deskripsi |
|---|---|
| SMA Cross | Golden/death cross dua SMA |
| RSI | Overbought/oversold via RSI |
| Supertrend | Trend-following dengan ATR band |

---

## Menambah Strategi Baru

1. Buat file baru di `backend/app/core/strategies/namaStrategi.py`, inherit dari `backtesting.Strategy`
2. Definisikan `params` dict dengan schema `{field: {type, default, min, max}}`
3. Register di `backend/app/core/strategies/__init__.py`
4. Frontend akan auto-render param di UI — tidak perlu sentuh kode frontend

---

## API Endpoint

| Method | Path | Deskripsi |
|---|---|---|
| `GET` | `/api/strategies` | List semua strategi + param schema |
| `GET` | `/api/data/symbols` | List pair Binance Futures |
| `GET` | `/api/data/timeframes` | List timeframe tersedia |
| `POST` | `/api/backtest/run` | Jalankan backtest, return stats + trades + equity curve |

Contoh request backtest:

```bash
curl -X POST http://localhost:8000/api/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "15m",
    "start_date": "2024-01-01",
    "end_date": "2024-06-30",
    "strategy": "StochStateMachine",
    "params": {"sl_pct": 2.0, "tp_pct": 3.0, "leverage": 5}
  }'
```

Response memiliki keys: `stats`, `trades`, `equity_curve`, `indicators`.

---

## Lisensi

MIT — gunakan bebas untuk keperluan pribadi dan riset.
