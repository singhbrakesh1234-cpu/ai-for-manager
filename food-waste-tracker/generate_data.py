"""
Synthetic household pantry simulator – AI Food Waste & Expiry Tracker
400 households x 15 grocery categories x 84 days (12 weeks).
Each purchase is a 'lot' with a storage place and expiry date; households eat from
their pantry day by day (first-expiring first) and whatever is left at expiry is wasted.
ALL DATA IS SYNTHETIC.
"""
import os
import numpy as np
import pandas as pd

SEED, N_HH, N_DAYS = 7, 400, 84
OUT = "data"
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- catalogue
# name, unit, kg per unit, price (Rs), shelf life by storage, best storage, common worse storage, daily units per person
CAT = [
    ("Milk",          "1 L pack",    1.00,  62, {"fridge": 3,  "pantry": 1},   "fridge", "fridge", 0.22),
    ("Curd",          "400 g tub",   0.40,  45, {"fridge": 6,  "pantry": 2},   "fridge", "fridge", 0.08),
    ("Paneer",        "200 g pack",  0.20,  95, {"fridge": 4,  "freezer": 30}, "fridge", "fridge", 0.04),
    ("Bread",         "400 g loaf",  0.40,  45, {"pantry": 4,  "fridge": 6},   "fridge", "pantry", 0.07),
    ("Eggs",          "6 eggs",      0.36,  48, {"fridge": 18, "pantry": 7},   "fridge", "pantry", 0.08),
    ("Bananas",       "6 bananas",   0.70,  50, {"pantry": 4,  "fridge": 6},   "pantry", "pantry", 0.06),
    ("Apples",        "1 kg",        1.00, 180, {"fridge": 20, "pantry": 8},   "fridge", "pantry", 0.03),
    ("Tomatoes",      "1 kg",        1.00,  40, {"pantry": 5,  "fridge": 9},   "fridge", "pantry", 0.06),
    ("Leafy greens",  "250 g bunch", 0.25,  30, {"fridge": 3,  "pantry": 1},   "fridge", "fridge", 0.07),
    ("Coriander",     "100 g bunch", 0.10,  15, {"fridge": 5,  "pantry": 2},   "fridge", "pantry", 0.06),
    ("Onions",        "1 kg",        1.00,  35, {"pantry": 30},                "pantry", "pantry", 0.07),
    ("Chicken",       "500 g",       0.50, 160, {"fridge": 2,  "freezer": 60}, "freezer", "fridge", 0.04),
    ("Cheese",        "200 g",       0.20, 120, {"fridge": 25},                "fridge", "fridge", 0.02),
    ("Snacks",        "1 pack",      0.20,  50, {"pantry": 90},                "pantry", "pantry", 0.05),
    ("Rice & atta",   "5 kg bag",    5.00, 300, {"pantry": 180},               "pantry", "pantry", 0.006),
]
CATS = pd.DataFrame(CAT, columns=["category", "unit", "kg_per_unit", "price", "shelf", "best_storage", "worse_storage", "rate"])
CATS["shelf_life_best"] = [s[b] for s, b in zip(CATS.shelf, CATS.best_storage)]
NC = len(CATS)
NONVEG = {"Chicken", "Eggs"}

# ---------------------------------------------------------------- households
size = rng.choice([1, 2, 3, 4, 5, 6], N_HH, p=[.15, .22, .20, .25, .12, .06])
hh = pd.DataFrame({
    "household_id": [f"H{i:03d}" for i in range(1, N_HH + 1)],
    "household_size": size,
    "household_type": np.select([size == 1, size == 2, size <= 4], ["Single", "Couple", "Family"], "Joint family"),
    "vegetarian": rng.random(N_HH) < 0.45,
    "cook_days_per_week": rng.integers(3, 8, N_HH),
    "shops_per_week": rng.choice([1, 2, 3], N_HH, p=[.35, .45, .20]),
    "overbuy": np.clip(rng.lognormal(0.04, 0.16, N_HH), 0.85, 1.8),      # buys more than needed
    "pantry_awareness": rng.uniform(0.2, 0.9, N_HH),                       # checks what is at home
    "stores_correctly": rng.uniform(0.45, 0.95, N_HH),                     # uses best storage
    "eat_out_rate": rng.uniform(0.03, 0.30, N_HH),                         # plans change
})
pref = rng.uniform(0.5, 1.5, (N_HH, NC)) * (rng.random((N_HH, NC)) < 0.85)
for i, v in enumerate(hh.vegetarian):
    if v:
        for j, c in enumerate(CATS.category):
            if c in NONVEG: pref[i, j] = 0
pref[:, CATS.category.tolist().index("Onions")] = np.maximum(pref[:, CATS.category.tolist().index("Onions")], 0.6)

# ---------------------------------------------------------------- daily demand (units of each category)
dow = np.arange(N_DAYS) % 7
weekend = np.isin(dow, [5, 6]).astype(float)
festival = ((np.arange(N_DAYS) // 7) == 8).astype(float)             # festival week (week 9)
demand = np.zeros((N_HH, NC, N_DAYS))
for i in range(N_HH):
    eat_out = rng.random(N_DAYS) < hh.eat_out_rate[i]
    cook = hh.cook_days_per_week[i] / 7
    base = hh.household_size[i] * CATS.rate.values * pref[i]
    for j in range(NC):
        mult = (1 + 0.2 * weekend + 0.3 * festival) * np.where(eat_out, 0.35, 1.0) * (0.6 + 0.6 * cook)
        demand[i, j] = base[j] * mult * rng.gamma(6, 1 / 6, N_DAYS)

# ---------------------------------------------------------------- simulate pantry
lots, consumed_log = [], np.zeros((N_HH, NC, N_DAYS))
open_lots_end = []
lot_id = 0
SHOP_DOWS = [''] * N_HH
for i in range(N_HH):
    k = hh.shops_per_week[i]
    shop_dows = sorted(rng.choice(7, k, replace=False))
    SHOP_DOWS[i] = " ".join(map(str, shop_dows))
    shop_days = [d for d in range(N_DAYS) if d % 7 in shop_dows]
    for j in range(NC):
        if pref[i, j] == 0: continue
        cat = CATS.iloc[j]
        pantry = []                               # [lot_index, remaining, expiry_day]
        hist_use = []
        # very short-life items (milk) are bought on a daily top-up; others on shopping days
        my_days = list(range(0, N_DAYS, 2)) if cat.shelf_life_best <= 3 else shop_days
        for d in range(N_DAYS):
            if d in my_days:
                nxt = next((s for s in my_days if s > d), d + 7)
                gap = min(nxt - d, cat.shelf_life_best + 1)
                exp_need = hh.household_size[i] * cat.rate * pref[i, j] * gap
                on_hand = sum(p[1] for p in pantry)
                promo = rng.random() < 0.12
                want = exp_need * hh.overbuy[i] * rng.uniform(0.8, 1.25) - hh.pantry_awareness[i] * on_hand
                if promo: want *= rng.uniform(1.5, 2.2)
                # minimum pack: if anything is needed and stock is low, buy at least 1
                qty = int(np.ceil(want)) if want > 0.25 else 0
                if cat.category in ("Rice & atta", "Snacks"):
                    qty = 1 if on_hand < hh.household_size[i] * cat.rate * pref[i, j] * 7 else 0
                if qty > 0:
                    good = rng.random() < hh.stores_correctly[i]
                    storage = cat.best_storage if good else cat.worse_storage
                    if storage not in cat.shelf: storage = cat.best_storage
                    life = max(1, int(round(cat.shelf[storage] * rng.uniform(0.85, 1.15))))
                    past = np.mean(hist_use[-28:]) if len(hist_use) >= 7 else hh.household_size[i] * cat.rate
                    lots.append(dict(lot_id=lot_id, household_id=hh.household_id[i], category=cat.category, day=d,
                                     week=d // 7, storage=storage, shelf_life_days=life, quantity_units=qty,
                                     unit_price=cat.price, bought_on_promo=int(promo), on_hand_units=round(on_hand, 2),
                                     est_daily_use=round(max(past, 0.01), 4), expiry_day=d + life,
                                     wasted_units=0.0))
                    pantry.append([len(lots) - 1, float(qty), d + life])
                    lot_id += 1
            # eat: first-expiring first
            need = demand[i, j, d]
            pantry.sort(key=lambda p: p[2])
            eaten = 0.0
            for p in pantry:
                if need <= 0: break
                take = min(p[1], need); p[1] -= take; need -= take; eaten += take
            consumed_log[i, j, d] = eaten
            hist_use.append(demand[i, j, d])
            # waste whatever expires today
            keep = []
            for p in pantry:
                if p[2] <= d and p[1] > 1e-6:
                    lots[p[0]]["wasted_units"] += p[1]
                elif p[1] > 1e-6:
                    keep.append(p)
            pantry = keep
        for p in pantry:
            open_lots_end.append(dict(lot_id=p[0], remaining_units=p[1]))

lots = pd.DataFrame(lots)
lots = lots.merge(hh[["household_id", "household_size", "cook_days_per_week", "eat_out_rate", "vegetarian"]], on="household_id")
lots = lots.merge(CATS[["category", "kg_per_unit"]], on="category")
lots["value"] = lots.quantity_units * lots.unit_price
lots["wasted_fraction"] = (lots.wasted_units / lots.quantity_units).round(4)
lots["wasted_value"] = (lots.wasted_units * lots.unit_price).round(2)
lots["wasted_kg"] = (lots.wasted_units * lots.kg_per_unit).round(3)
lots["complete"] = (lots.expiry_day <= N_DAYS - 1).astype(int)

weekly = pd.DataFrame([dict(household_id=hh.household_id[i], category=CATS.category[j], week=w,
                            demand_units=round(demand[i, j, w * 7:(w + 1) * 7].sum(), 3))
                       for i in range(N_HH) for j in range(NC) if pref[i, j] > 0 for w in range(N_DAYS // 7)])

# current pantry ("today" = day 83) for the app
op = pd.DataFrame(open_lots_end).merge(lots, on="lot_id")
op["days_left"] = op.expiry_day - (N_DAYS - 1)
current = op[op.days_left > 0][["household_id", "category", "storage", "shelf_life_days", "quantity_units", "remaining_units",
                                "unit_price", "bought_on_promo", "on_hand_units", "est_daily_use", "day", "days_left"]].copy()
current["remaining_units"] = current.remaining_units.round(2)

# daily demand saved compactly for the back-test (weeks 9-12 only)
dd = [dict(household_id=hh.household_id[i], category=CATS.category[j], day=d, demand_units=round(demand[i, j, d], 4))
      for i in range(N_HH) for j in range(NC) if pref[i, j] > 0 for d in range(56, N_DAYS)]
daily = pd.DataFrame(dd)

# realistic data-quality issues in the raw purchase log
raw = lots.copy()
dup = raw.sample(frac=0.01, random_state=1)
raw = pd.concat([raw, dup]).reset_index(drop=True)
raw.loc[raw.sample(frac=0.02, random_state=2).index, "storage"] = np.nan                  # not recorded
raw.loc[raw.sample(frac=0.004, random_state=3).index, "quantity_units"] = 0               # scanner error

os.makedirs(OUT, exist_ok=True)
hh["shop_days"] = SHOP_DOWS
hh.to_csv(f"{OUT}/households.csv", index=False)
pd.DataFrame(pref, columns=CATS.category).assign(household_id=hh.household_id).to_csv(f"{OUT}/preferences.csv", index=False)
CATS.drop(columns=["shelf"]).to_csv(f"{OUT}/categories.csv", index=False)
raw.to_csv(f"{OUT}/purchases_raw.csv", index=False)
weekly.to_csv(f"{OUT}/weekly_demand.csv", index=False)
daily.to_csv(f"{OUT}/daily_demand_backtest.csv", index=False)
current.to_csv(f"{OUT}/current_pantry.csv", index=False)
print(f"households={len(hh)} purchases={len(raw):,} weekly_rows={len(weekly):,} current_items={len(current):,}")
