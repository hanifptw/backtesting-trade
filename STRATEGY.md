# Strategy Reference

Dokumen ini meng-export strategi yang dipakai bot ini supaya bisa direplikasi di project lain. Semua angka yang bertanda **(configurable)** dapat diubah saat runtime dan ditandai dengan field di `settings` table.

---

## 1. Ringkasan

- **Pasar:** Binance Futures USDT-M Perpetual.
- **Universe:** Top-N coin berdasarkan market cap dari CoinGecko, di-intersect dengan daftar perpetual USDT yang aktif di Binance. Stablecoin & wrapped token di-exclude.
- **Indikator:** Stochastic Oscillator dua-tahap (%K dan %D) — diaplikasikan ke close-time bar yang sudah closed.
- **Logika sinyal:** State machine 5 status (IDLE → ARMED → IN_POSITION → IDLE).
- **Sisi:** Long & Short keduanya didukung; bisa di-toggle independen.
- **Risk:** Fixed-margin sizing (`trade_amount × leverage`), SL persen, TP persen, optional tiered trailing.
- **Eksekusi:** Market entry + STOP_MARKET SL + TAKE_PROFIT_MARKET TP (closePosition=True, workingType=MARK_PRICE).

---

## 2. Universe Selection

| Parameter | Default | Catatan |
|---|---|---|
| Source | CoinGecko `/coins/markets` (`order=market_cap_desc`) | |
| Top-N | **20** | Hard-coded (`_TOP_N`); ubah konstanta untuk N lain |
| Refresh interval | **6 jam** | APScheduler `IntervalTrigger(hours=6)` |
| Filter | Eksklusi stablecoin + wrapped token, intersect dengan Binance USDT-M perpetual `status=TRADING` | |
| Cache TTL | **6 jam** | In-memory CoinGecko cache |

Output: `[(binance_symbol, base_asset, mcap_rank)]`, mis. `("BTCUSDT", "BTC", 1)`.

---

## 3. Stochastic Indicator

### Rumus

```
lowest_n  = rolling_min(low,  window=K_PERIOD)
highest_n = rolling_max(high, window=K_PERIOD)
raw_k     = 100 * (close - lowest_n) / (highest_n - lowest_n)
%K        = SMA(raw_k, window=SMOOTH)
%D        = SMA(%K,    window=D_PERIOD)
```

### Parameter (configurable)

| Field | Default | Range typikal | Field DB |
|---|---|---|---|
| K period | **14** | 5–21 | `settings.stoch_k` |
| D period | **3**  | 2–5  | `settings.stoch_d` |
| Smooth  | **3**  | 1–5  | `settings.stoch_smooth` |

Diaplikasikan pada DataFrame OHLCV yang bar terakhirnya **sudah closed** (drop in-progress bar dari `klines` endpoint).

---

## 4. Signal State Machine

State per simbol disimpan persistent (mis. di tabel `signal_states`). Diproses setiap kali bar baru tertutup.

### Konstanta zone

```
OVERSOLD   = 20.0   # K/D di bawah ini = "in OS zone"
OVERBOUGHT = 80.0   # K/D di atas ini = "in OB zone"
```

> Catatan: di project ini level 20/80 di-hardcode di `state.py`. Kalau mau diparameterkan, expose sebagai `oversold_level` / `overbought_level` setting.

### State

```
IDLE          — tidak ada sinyal aktif
LONG_ARMED    — sinyal long ter-armed, menunggu konfirmasi breakout
SHORT_ARMED   — sinyal short ter-armed, menunggu konfirmasi breakdown
IN_LONG       — posisi long sedang berjalan
IN_SHORT      — posisi short sedang berjalan
```

### Transisi

**Definisi cross:**
- `cross_up(prev, curr)`   → `prev.K ≤ prev.D` DAN `curr.K > curr.D`
- `cross_down(prev, curr)` → `prev.K ≥ prev.D` DAN `curr.K < curr.D`

#### Dari IDLE

| Kondisi (cek pada dua bar terakhir yang closed) | Transisi |
|---|---|
| `cross_up` DAN `prev.K < 20` DAN `curr.K < 20` | → **LONG_ARMED**, simpan `armed_extreme_k = min(prev.K, curr.K)` |
| `cross_down` DAN `prev.K > 80` DAN `curr.K > 80` | → **SHORT_ARMED**, simpan `armed_extreme_k = max(prev.K, curr.K)` |
| Selain itu | tetap **IDLE** |

#### Dari LONG_ARMED

| Kondisi | Transisi | Aksi |
|---|---|---|
| `curr.K ≥ 20` (breakout dari OS zone) | → **IN_LONG** | **ENTER_LONG** (open posisi) |
| `curr.K < armed_extreme_k` (invalidasi: K turun di bawah ekstrem yang tercatat) | → **IDLE** | reset |
| Selain itu | tetap **LONG_ARMED** | update `armed_extreme_k = min(armed_extreme_k, curr.K)` |

#### Dari SHORT_ARMED

| Kondisi | Transisi | Aksi |
|---|---|---|
| `curr.K ≤ 80` (breakdown dari OB zone) | → **IN_SHORT** | **ENTER_SHORT** |
| `curr.K > armed_extreme_k` (invalidasi: K naik di atas ekstrem) | → **IDLE** | reset |
| Selain itu | tetap **SHORT_ARMED** | update `armed_extreme_k = max(armed_extreme_k, curr.K)` |

#### Dari IN_LONG

| Kondisi | Transisi | Aksi |
|---|---|---|
| `cross_down(prev, curr)` | → **IDLE** | **EXIT_LONG (TP signal)** |
| Selain itu | tetap **IN_LONG** | — |

#### Dari IN_SHORT

| Kondisi | Transisi | Aksi |
|---|---|---|
| `cross_up(prev, curr)` | → **IDLE** | **EXIT_SHORT (TP signal)** |
| Selain itu | tetap **IN_SHORT** | — |

### Catatan penting

- **"Cross"** butuh dua bar berurutan (prev + curr). Cold-start (state baru, prev=None) → HOLD.
- Sinyal Exit (cross balik %K/%D) **bukan satu-satunya cara keluar**. Posisi juga bisa keluar via:
  - SL order di Binance (STOP_MARKET reduceOnly)
  - TP order di Binance (TAKE_PROFIT_MARKET reduceOnly) — di project ini TP order dipasang sekaligus sehingga exit lebih sering via order, bukan via cross.
  - Manual close.
- **Cleanup armed yang nyangkut:** jika state ARMED tidak transisi dalam `armed_ttl = 4 jam`, reset ke IDLE (lihat `reconcile_states`).

---

## 5. Position Sizing

```
notional = trade_amount × leverage
qty      = notional / entry_price
```

Lalu `qty` di-quantize ke `LOT_SIZE.stepSize` exchange info (floor).

| Field | Default | Catatan |
|---|---|---|
| `trade_amount` | **100 USDT** (configurable) | Margin per trade |
| `leverage`     | **5x** (configurable)        | Cap Binance per simbol biasanya 20–125 |
| `max_positions`| **5** (configurable)         | Cap posisi terbuka simultan |
| `equity_pct`   | **2 %** (configurable, tidak dipakai default tapi disimpan) | Sizing alternatif berbasis equity |

**Notional efektif** per trade = `trade_amount × leverage`. Misal `100 × 5 = 500 USDT` notional.

---

## 6. Risk Management

### Stop Loss (statis)

```
LONG  : sl_price = entry × (1 − sl_pct/100)
SHORT : sl_price = entry × (1 + sl_pct/100)
```

| Field | Default | Range typikal |
|---|---|---|
| `sl_pct` | **2.0 %** (configurable) | 0.5–5 % |

SL order: `STOP_MARKET`, `closePosition=true`, `workingType=MARK_PRICE`, `reduceOnly=true`.

**Mark-aware retry**: kalau SL ditolak Binance dengan `-2021` ("would immediately trigger"), refresh mark price dan retry sekali dengan SL yang di-clamp ke sisi aman mark + buffer `max(sl_pct × 0.2, 0.1) %`. Quantize **away-from-mark** (floor untuk LONG, ceil untuk SHORT).

### Take Profit (statis, optional)

```
LONG  : tp_price = entry × (1 + tp_pct/100)
SHORT : tp_price = entry × (1 − tp_pct/100)
```

| Field | Default | Range typikal |
|---|---|---|
| `tp_pct` | **3.0 %** (configurable) | 1–10 % |

TP order: `TAKE_PROFIT_MARKET`, `closePosition=true`, `workingType=MARK_PRICE`. TP non-kritis: kalau gagal, posisi tetap aman karena ada SL.

### Tiered Trailing Stop (optional)

Diaktifkan dengan `trailing_enabled=true`. Setiap 30 detik, untuk setiap posisi:

```
LONG:  profit_pct = (mark − entry) / entry × 100
SHORT: profit_pct = (entry − mark) / entry × 100

if profit_pct < trailing_trigger_pct:
    biarkan SL awal
else:
    milestone_idx = floor((profit_pct − trigger) / step)
    sl_offset_pct = milestone_idx × step      # M1=0, M2=step, M3=2·step, ...
    desired_sl    = entry × (1 ± sl_offset_pct/100)
    if desired_sl lebih bagus dari SL saat ini DAN improvement ≥ 0.1 %:
        place SL baru DULU, lalu cancel SL lama (never unprotected)
```

| Field | Default | Catatan |
|---|---|---|
| `trailing_enabled`        | **false** (configurable) | |
| `trailing_trigger_pct`    | **1.0 %** (configurable) | Profit minimum untuk aktifkan trailing |
| `trailing_offset_pct`     | **0.5 %** (configurable) | Step antar milestone |
| `_MIN_BUMP_PCT`           | **0.1 %** (hard-coded)   | Hindari API spam |

Tier examples (trigger=1 %, step=0.5 %):
- profit 1.0–1.49 % → M1, SL di entry (breakeven)
- profit 1.5–1.99 % → M2, SL di entry ± 0.5 %
- profit 2.0–2.49 % → M3, SL di entry ± 1.0 %

---

## 7. Filter Eksekusi (Risk Gates)

Berlaku **sebelum** entry diteruskan ke executor:

1. **Autotrade enabled?** Kalau `autotrade_enabled=false`, drop sinyal entry (state machine tetap jalan).
2. **Side enabled?** Drop kalau `long_enabled=false` & sinyal LONG; idem SHORT.
3. **No double-up:** Cek `positions` table — kalau simbol sudah ada posisi OPEN, skip.
4. **Cap max_positions:** Hitung posisi OPEN; tolak kalau ≥ cap. (Auto-reverse dapat +1 slack.)
5. **Sizing > 0:** Cek `qty > 0` setelah quantize ke stepSize.

---

## 8. Auto-Reverse on SL (optional)

| Field | Default | Catatan |
|---|---|---|
| `auto_reverse_enabled` | **false** (configurable) | Aktifkan flip otomatis |
| `auto_reverse_max`     | **1** (configurable, cap 10 di runtime) | Maksimum flip berturut-turut |

**Trigger:** saat `sync_positions` job mendeteksi posisi tertutup karena SL hit di Binance.

**Aksi:** publish `EntrySignal(side=opposite, reverse_chain=prev_chain+1)`. Auto-reverse entry bypass autotrade/side toggle (asumsinya original entry sudah authorized) tapi tetap respek `max_positions` (dengan +1 slack).

**Chain reset:** kalau posisi tidak ditutup karena SL (TP atau manual), `reverse_chain` reset ke 0 lewat repository.

---

## 9. Operasional Timings

| Job | Interval | Tujuan |
|---|---|---|
| Universe refresh | 6 jam (configurable di scheduler) | Re-fetch top-N |
| Kline poll + signal | **60 s** | Idempotent: hanya proses bar dengan `close_time` baru |
| Trailing tick | **30 s** | Cek milestone untuk semua posisi |
| Position sync | **2 menit** | Deteksi SL hit / liquidation di luar bot |
| Weekly AI report | Cron, default Sun 23:00 | Eval AI atas trade history |

---

## 10. Settings Table — Complete Reference

Satu row singleton (`id=1`). Semua field di bawah dapat diubah runtime tanpa restart bot.

| Field | Tipe | Default | Range | Catatan |
|---|---|---|---|---|
| `timeframe` | str | `15m` | `1m,3m,5m,15m,30m,1h,2h,4h,1d` | Klines interval |
| `mode` | str | `testnet` | `testnet \| live` | Swap base URL & API key |
| `autotrade_enabled` | bool | **false** | — | Master switch eksekusi |
| `long_enabled` | bool | true | — | |
| `short_enabled` | bool | true | — | |
| `leverage` | int | 5 | 1–125 | Per simbol |
| `trade_amount` | Decimal | 100.0 USDT | ≥ 10 | Margin per trade |
| `equity_pct` | Decimal | 2.0 % | 0.1–100 | Sizing alternatif (tidak dipakai default) |
| `max_positions` | int | 5 | 1–20 | Cap simultan |
| `sl_pct` | Decimal | 2.0 % | 0.1–100 | SL distance |
| `tp_pct` | Decimal | 3.0 % | 0.1–100 | TP distance |
| `trailing_enabled` | bool | false | — | |
| `trailing_trigger_pct` | Decimal | 1.0 % | 0.1–100 | Aktifkan trailing |
| `trailing_offset_pct` | Decimal | 0.5 % | 0.1–100 | Step milestone |
| `auto_reverse_enabled` | bool | false | — | |
| `auto_reverse_max` | int | 1 | 0–10 | Maks chain |
| `stoch_k` | int | 14 | ≥ 1 | Lookback periode K |
| `stoch_d` | int | 3 | ≥ 1 | Smoothing %D |
| `stoch_smooth` | int | 3 | ≥ 1 | Smoothing %K |

---

## 11. Portable Pseudocode

```python
# === per-tick (every 60s) ===
for symbol in universe:
    df = fetch_klines(symbol, settings.timeframe, limit=200)
    df_closed = df[:-1]                          # buang in-progress bar
    if df_closed.empty: continue
    if df_closed.last.close_time <= watermark[symbol]: continue

    df_ind = add_stochastic(df_closed,
                            k=settings.stoch_k,
                            d=settings.stoch_d,
                            smooth=settings.stoch_smooth)
    valid = df_ind.dropna_stoch()
    if len(valid) < 2: continue
    prev_bar = Bar(K=valid[-2].K, D=valid[-2].D)
    curr_bar = Bar(K=valid[-1].K, D=valid[-1].D)

    transition = step(state[symbol], prev_bar, curr_bar, armed_extreme_k[symbol])
    save_state(symbol, transition.new_state, transition.armed_extreme_k)
    watermark[symbol] = df_closed.last.close_time

    if transition.decision == ENTER_LONG:  emit EntrySignal(LONG,  symbol, curr_bar.close)
    if transition.decision == ENTER_SHORT: emit EntrySignal(SHORT, symbol, curr_bar.close)
    if transition.decision == EXIT_LONG:   emit ExitSignal(LONG,   symbol)
    if transition.decision == EXIT_SHORT:  emit ExitSignal(SHORT,  symbol)

# === entry handler ===
def on_entry(ev):
    if not settings.autotrade_enabled: return
    if ev.side == LONG  and not settings.long_enabled:  return
    if ev.side == SHORT and not settings.short_enabled: return
    if has_open_position(ev.symbol): return
    if count_open_positions() >= settings.max_positions: return

    qty = quantize_qty(ev.symbol, settings.trade_amount * settings.leverage / ev.price)
    if qty <= 0: return

    set_leverage(ev.symbol, settings.leverage)
    market_resp = place_market(ev.symbol, BUY if ev.side==LONG else SELL, qty)
    fill = parse_fill_price(market_resp, fallback=ev.price)

    sl_price = fill * (1 - settings.sl_pct/100) if ev.side==LONG else fill * (1 + settings.sl_pct/100)
    sl_price = quantize_price(ev.symbol, sl_price)
    try:
        sl_resp = place_stop_market_reduce_only(
            ev.symbol,
            side=SELL if ev.side==LONG else BUY,
            stop_price=sl_price,
            close_position=True,
            working_type=MARK_PRICE,
        )
    except BinanceAPIException as e:
        if e.code == -2021:
            mark = mark_price(ev.symbol)
            safety_pct = max(settings.sl_pct * 0.2, 0.1)
            sl_price = mark * (1 - safety_pct/100) if ev.side==LONG else mark * (1 + safety_pct/100)
            sl_price = quantize_away_from_mark(ev.symbol, sl_price, mark)
            sl_resp = place_stop_market_reduce_only(...)
        else:
            close_position_market(ev.symbol)   # rollback
            return

    tp_price = fill * (1 + settings.tp_pct/100) if ev.side==LONG else fill * (1 - settings.tp_pct/100)
    tp_price = quantize_price(ev.symbol, tp_price)
    try:
        tp_resp = place_take_profit_market_reduce_only(ev.symbol, ..., tp_price)
    except Exception:
        tp_resp = None    # non-fatal

    persist_position(ev, qty, fill, sl_price, sl_resp.id, tp_price, tp_resp.id if tp_resp else None)

# === exit handler ===
def on_exit(ev):
    pos = open_position(ev.symbol)
    if pos is None: return
    close_resp = place_market_reduce_only(ev.symbol, opposite_side(pos), pos.qty)
    cancel(pos.sl_order_id)
    cancel(pos.tp_order_id)
    record_trade(pos, close_resp)

# === trailing tick (every 30s) ===
for pos in open_positions if settings.trailing_enabled:
    mark = mark_price(pos.symbol)
    profit_pct = (mark - pos.entry)/pos.entry*100 if pos.side==LONG else (pos.entry - mark)/pos.entry*100
    if profit_pct < settings.trailing_trigger_pct: continue
    milestone = floor((profit_pct - settings.trailing_trigger_pct) / settings.trailing_offset_pct)
    offset = milestone * settings.trailing_offset_pct
    desired = pos.entry * (1 + offset/100) if pos.side==LONG else pos.entry * (1 - offset/100)
    desired = quantize_price(pos.symbol, desired)
    if not_better_than_current(desired, pos.sl_price, pos.side): continue
    if abs(desired - pos.sl_price)/pos.sl_price*100 < 0.1: continue
    new_resp = place_stop_market_reduce_only(...)   # place FIRST
    cancel(pos.sl_order_id)                         # then cancel old
    update_position_sl(pos, desired, new_resp.id)

# === sync (every 2 min) ===
for pos in db_open_positions:
    if pos.symbol not in binance_open_positions:
        exit_px = best_effort_fill_price(pos) or pos.sl_price or pos.entry
        reason  = classify(exit_px, pos.sl_price, pos.tp_price)
        close_in_db(pos, exit_px, reason)
        cancel_stragglers(pos.sl_order_id, pos.tp_order_id)
        if reason == SL and settings.auto_reverse_enabled and pos.reverse_chain < settings.auto_reverse_max:
            emit EntrySignal(opposite(pos.side), pos.symbol, exit_px, reverse_chain=pos.reverse_chain+1)
```

---

## 12. Catatan Implementasi yang Sering Terlupa

- **`python-binance` route `STOP_MARKET` / `TAKE_PROFIT_MARKET` ke endpoint `algoOrder`.** Response pakai `algoId` + `algoStatus` (bukan `orderId` + `status`). Validasi response harus terima keduanya, dan cancel-nya kirim `algoId=` (bukan `orderId=`).
- **`avgPrice` testnet bisa `0.00000`.** Fallback ke harga sinyal atau query `userTrades` untuk fill price beneran.
- **Quantize harga away-from-mark** untuk SL/TP supaya tidak balik ke arah mark dan trigger -2021. Floor untuk side di bawah mark, ceil untuk side di atas mark.
- **Place new SL DULU, baru cancel SL lama** saat trailing — agar tidak ada window posisi tanpa proteksi.
- **Skip in-progress bar.** Binance `klines` selalu return bar terakhir yang masih berjalan; drop sebelum hitung indikator dan transisi state.
- **Watermark per simbol.** Simpan `last_bar_close_ms` agar tidak re-process bar yang sama setelah restart atau ganti TF (clear saat TF berubah).
- **Reset state machine kalau ganti TF.** K/D/ARMED lama tidak relevan untuk TF baru.
- **State `IN_LONG`/`IN_SHORT` tanpa posisi terbuka** harus di-reconcile (reset ke IDLE) supaya sinyal baru tidak terblokir kalau crash terjadi di tengah entry.
