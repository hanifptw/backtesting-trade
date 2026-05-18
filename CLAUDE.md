# CLAUDE.md

Panduan untuk Claude Code (dan kontributor lain) yang bekerja di repo ini.

## Apa Proyek Ini

Web app lokal untuk **backtesting strategi trading crypto** di Binance Futures. Engine pakai [`backtesting.py`](https://github.com/kernc/backtesting.py), data dari Binance via CCXT, UI Next.js dengan chart `lightweight-charts`.

**Tujuan utama:** mem-backtest strategi Stochastic state-machine milik user (lihat [STRATEGY.md](STRATEGY.md)) yang sudah live sebagai bot, tanpa perlu menjalankan bot beneran.

**Bukan untuk:** live/paper trading, multi-user SaaS, deployment publik. Murni personal tool yang jalan di laptop.

## Stack & Layout

- **Backend:** Python 3.11+, FastAPI, `backtesting.py`, `ccxt`, `pandas`. Folder: `backend/`. Run: `cd backend && uvicorn app.main:app --reload` (port 8000).
- **Frontend:** Next.js 14 (App Router) + TypeScript + Tailwind + `lightweight-charts`. Folder: `frontend/`. Run: `cd frontend && npm run dev` (port 3000).
- **Komunikasi:** REST JSON, CORS dibuka untuk `localhost:3000`. Sync request (tidak ada job queue di MVP).

## Sumber Kebenaran Strategi

[STRATEGY.md](STRATEGY.md) adalah **source-of-truth** untuk logika strategi Stochastic state-machine. File itu meng-dokumentasi:

- Indikator (rumus %K, %D)
- State machine 5-state (IDLE / LONG_ARMED / SHORT_ARMED / IN_LONG / IN_SHORT) + transisi
- Position sizing (`trade_amount × leverage`)
- Risk management (SL, TP, trailing tiered milestone)
- Auto-reverse on SL
- Settings lengkap dengan default & range

**Saat memodifikasi `backend/app/core/strategies/stoch_state_machine.py`:**

1. Logika state machine harus tetap match dengan STRATEGY.md §4. Kalau ada perubahan, update STRATEGY.md dulu lalu sinkronkan kode.
2. Param defaults & ranges harus match dengan STRATEGY.md §10 (Settings Table).
3. Hal-hal yang **disederhanakan** untuk konteks backtest (vs bot live) dicatat di docstring kelas strategi:
   - Tidak ada `STOP_MARKET algoOrder` Binance — pakai `sl=`, `tp=` native backtesting.py
   - Tidak ada mark-price retry / -2021 — engine fill langsung di candle
   - Tidak ada quantize ke `LOT_SIZE.stepSize` — fractional sizing
   - Tidak ada watermark / in-progress bar handling — semua bar di backtest sudah closed
   - Tidak ada reconciliation TTL (`armed_ttl`) — single backtest run
4. **Trailing & auto-reverse** diimplementasi di backtesting.py side menggunakan:
   - Trailing: update `trade.sl` (writable property) di `next()` saat milestone tercapai.
   - Auto-reverse: cek `self.closed_trades[-1]` di awal `next()`, kalau exit price ≈ SL price → flip side via `self.buy()` / `self.sell()`.

## Konvensi

- **Naming kolom OHLCV:** backtesting.py butuh `Open,High,Low,Close,Volume` (title case). CCXT return lowercase — selalu rename di `data_source.py`. Jangan duplikasi rename di tempat lain.
- **Timezone:** Semua timestamp UTC. Frontend convert ke local time hanya di display.
- **Param schema format:** `{field_name: {type: "int"|"float"|"bool", default: ..., min: ..., max: ...}}`. Tambahkan field baru = update kelas strategy + schema dict, frontend auto-render lewat `StrategyPicker`.
- **Strategy registry:** tambah preset baru = bikin file di `backend/app/core/strategies/`, register di `__init__.py`. Tidak perlu sentuh API endpoint atau frontend.
- **Indicator registration:** selalu pakai `self.I(func, ...)` di `init()` agar indikator di-precompute, plot-ready, dan keluar di response API (untuk overlay di chart).

## Yang Harus Dihindari

- **Jangan** rebuild engine backtest dari nol. Pakai `backtesting.py`. Kalau ada limitasi, dokumentasikan dan kerja-around, jangan bikin engine baru.
- **Jangan** tambah auth, multi-user, atau database persistence di MVP. Semua in-memory. Kalau butuh state, tunggu user explicit request.
- **Jangan** ubah `STRATEGY.md` tanpa diskusi — itu spesifikasi bot live user, harus tetap akurat.
- **Jangan** ekspos endpoint backtest ke public network. CORS hardcoded untuk `localhost` only.

## Verifikasi Cepat

Setelah perubahan di engine atau strategy:

1. Smoke test via curl: `curl -X POST http://localhost:8000/api/backtest/run -d @sample_request.json`
2. Cek output JSON punya keys: `stats`, `trades`, `equity_curve`, `indicators`
3. Untuk strategi Stoch: jalankan dengan default params di BTC/USDT 15m 6 bulan terakhir. Sanity:
   - Setiap entry LONG harus terjadi setelah cross_up dimana kedua %K terakhir < `oversold_level`
   - SL/TP harus dipasang di harga `entry × (1 ∓ sl_pct/100)` dan `entry × (1 ± tp_pct/100)`
   - Toggle `trailing_enabled` → max drawdown turun

## Plan Implementasi

Plan lengkap MVP ada di `/Users/hanifptw/.claude/plans/hey-clalude-saya-ingin-temporal-hollerith.md`. Phase 0 (file ini + STRATEGY.md) sudah selesai. Phase 1 selanjutnya: backend skeleton (FastAPI + CCXT fetch endpoint).
