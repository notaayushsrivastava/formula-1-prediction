# Market prices (paper trading only)

**No real money.** Real-money online betting is illegal in India, and pitwall-f1 is an
educational project. These files feed pretend stakes so we can measure, honestly,
whether the engine's odds know anything the market does not.

## Typing prices in
1. After a prediction is saved: `pitwall-f1 market template 2026 Abu` writes
   `data/markets/<year>-<round>.csv` with one line per driver.
2. Copy prices from a **public** odds page (no login, no scraping), with the UTC time you
   saw them. Only prices seen **before** a prediction was made are ever used against it.
3. Columns: `timestamp_utc, market (win|podium|points), driver (code, name or f1db id),
   price, price_kind (decimal|fractional|american|probability), source, kind
   (bookmaker|exchange|prediction_market)`.
4. `pitwall-f1 market check 2026 Abu --moment quali` shows the margin and engine vs market.

## Polymarket (optional, read-only, no key)
Find the event: `pitwall-f1 market find "Abu Dhabi Grand Prix"`, then add a line to
`polymarket.csv`: `year,round,event_slug`. pitwall reads public price history only.
