import json
import math
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==================================================================
# Settings – rename the app here
# ==================================================================
APP_NAME = "FreshKeep AI"
TAGLINE = "AI Food Waste & Expiry Tracker"
TODAY = pd.Timestamp("2026-10-26")          # the simulated "today" (day 84 of the data)

st.set_page_config(page_title=f"{APP_NAME} – {TAGLINE}", page_icon="🥬", layout="wide")

try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path.cwd()

LEAF, MINT, TOMATO, AMBER, INK, MUTED = "#1F5C45", "#E6F2EC", "#E4572E", "#F2A541", "#17302A", "#5E716B"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {{ font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ visibility: hidden; }}
.block-container {{ padding-top: 1.4rem; max-width: 1250px; }}
h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; }}
.hero {{ background: {LEAF}; color: white; border-radius: 22px; padding: 28px 32px; display: flex; justify-content: space-between;
        align-items: center; gap: 24px; flex-wrap: wrap; margin-bottom: 8px; }}
.hero h1 {{ color: white; font-size: 2.3rem; font-weight: 800; margin: 0; }}
.hero .tag {{ color: #BFE3D0; font-size: 1.02rem; font-weight: 600; margin-top: 2px; }}
.hero .sub {{ color: #E6F2EC; font-size: 0.98rem; margin-top: 10px; max-width: 640px; }}
.hero .pill {{ background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25); border-radius: 999px; padding: 8px 16px;
              font-weight: 600; font-size: 0.92rem; white-space: nowrap; }}
.kpi {{ background: white; border: 1px solid #E3ECE7; border-radius: 18px; padding: 18px 20px; height: 100%; min-height: 132px;
       box-shadow: 0 1px 2px rgba(23,48,42,0.04); }}
.kpi .label {{ color: {MUTED}; font-size: 0.84rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }}
.kpi .value {{ color: {INK}; font-size: 2rem; font-weight: 800; line-height: 1.2; margin-top: 4px; }}
.kpi .note {{ color: {MUTED}; font-size: 0.86rem; margin-top: 2px; }}
.kpi.alert .value {{ color: {TOMATO}; }}
.kpi.good .value {{ color: {LEAF}; }}
.item {{ background: white; border: 1px solid #E3ECE7; border-radius: 18px; padding: 16px 18px; margin-bottom: 14px; }}
.item .top {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
.item .name {{ font-weight: 700; color: {INK}; font-size: 1.05rem; }}
.item .meta {{ color: {MUTED}; font-size: 0.86rem; margin-top: 6px; }}
.item .tip {{ color: {INK}; font-size: 0.9rem; margin-top: 10px; background: {MINT}; border-radius: 10px; padding: 8px 10px; }}
.chip {{ border-radius: 999px; padding: 3px 10px; font-size: 0.78rem; font-weight: 700; white-space: nowrap; }}
.chip.red {{ background: #FDE5DE; color: #B23A1B; }}
.chip.amber {{ background: #FDF0DA; color: #8A5A00; }}
.chip.green {{ background: {MINT}; color: {LEAF}; }}
.bar {{ height: 8px; background: #EEF3F0; border-radius: 99px; margin-top: 10px; overflow: hidden; }}
.bar > div {{ height: 100%; border-radius: 99px; }}
.recipe {{ background: white; border: 1px solid #E3ECE7; border-radius: 18px; padding: 18px 20px; margin-bottom: 14px; }}
.recipe .title {{ font-weight: 800; font-size: 1.1rem; color: {INK}; }}
.recipe .uses {{ margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }}
.recipe .steps {{ color: {MUTED}; font-size: 0.9rem; margin-top: 10px; }}
.section-note {{ color: {MUTED}; font-size: 0.95rem; margin-top: -6px; margin-bottom: 14px; }}
.big {{ font-size: 2.6rem; font-weight: 800; color: {LEAF}; line-height: 1.1; }}
div[data-baseweb="tab-list"] {{ gap: 6px; }}
button[data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# ==================================================================
# Data and models
# ==================================================================
NEEDED = ["waste_risk_model.pkl", "waste_risk_meta.pkl", "consumption_model.pkl", "consumption_features.pkl", "kpis.json",
          "waste_model_comparison.csv", "data/households.csv", "data/categories.csv", "data/current_pantry_features.csv",
          "data/household_waste_history.csv", "data/weekly_demand.csv", "data/household_backtest.csv"]
missing = [f for f in NEEDED if not (BASE / f).exists()]
if missing:
    st.error("Missing project files: " + ", ".join(missing) + ". Run FoodWaste_Expiry_Tracker.ipynb first.")
    st.stop()


@st.cache_resource
def load_models():
    return (joblib.load(BASE / "waste_risk_model.pkl"), joblib.load(BASE / "waste_risk_meta.pkl"),
            joblib.load(BASE / "consumption_model.pkl"), joblib.load(BASE / "consumption_features.pkl"))


@st.cache_data
def load_data():
    d = BASE / "data"
    return (pd.read_csv(d / "households.csv"), pd.read_csv(d / "categories.csv"), pd.read_csv(d / "current_pantry_features.csv"),
            pd.read_csv(d / "household_waste_history.csv"), pd.read_csv(d / "weekly_demand.csv"),
            pd.read_csv(d / "household_backtest.csv"), json.load(open(BASE / "kpis.json")),
            pd.read_csv(BASE / "waste_model_comparison.csv"))


risk_model, meta, cons_model, cons_feats = load_models()
households, cats, pantry_all, history, weekly, hh_bt, K, model_cmp = load_data()
FEATURES, ALERT_TH = meta["features"], meta["threshold"]
CAT = cats.set_index("category")

LIFE = {"Milk": {"fridge": 3, "pantry": 1}, "Curd": {"fridge": 6, "pantry": 2}, "Paneer": {"fridge": 4, "freezer": 30},
        "Bread": {"pantry": 4, "fridge": 6}, "Eggs": {"fridge": 18, "pantry": 7}, "Bananas": {"pantry": 4, "fridge": 6},
        "Apples": {"fridge": 20, "pantry": 8}, "Tomatoes": {"pantry": 5, "fridge": 9}, "Leafy greens": {"fridge": 3, "pantry": 1},
        "Coriander": {"fridge": 5, "pantry": 2}, "Onions": {"pantry": 30}, "Chicken": {"fridge": 2, "freezer": 60},
        "Cheese": {"fridge": 25}, "Snacks": {"pantry": 90}}
EMOJI = {"Milk": "🥛", "Curd": "🥣", "Paneer": "🧀", "Bread": "🍞", "Eggs": "🥚", "Bananas": "🍌", "Apples": "🍎", "Tomatoes": "🍅",
         "Leafy greens": "🥬", "Coriander": "🌿", "Onions": "🧅", "Chicken": "🍗", "Cheese": "🧀", "Snacks": "🍪"}
TIPS = {"Milk": "Boil it, or turn it into paneer or kheer before it turns.", "Curd": "Make lassi, raita or curd rice today.",
        "Paneer": "Freeze it in cubes – it keeps for weeks.", "Bread": "Toast it or make bread upma; keep loaves in the fridge.",
        "Eggs": "Keep them in the fridge, not on the counter.", "Bananas": "Very ripe? Banana shake, or peel and freeze.",
        "Apples": "Store in the fridge crisper – they last twice as long.", "Tomatoes": "Cook a quick tomato chutney; move the rest to the fridge.",
        "Leafy greens": "Cook palak dal today – greens wilt fast.", "Coriander": "Wrap in a damp cloth in the fridge, or blend into chutney.",
        "Onions": "Keep in a dry, airy basket.", "Chicken": "Cook today or move it to the freezer.", "Cheese": "Wrap tightly after opening.",
        "Snacks": "Close the pack with a clip."}
RECIPES = [
    ("Bread upma", ["Bread", "Onions", "Tomatoes", "Coriander"], 15, "Cube the bread, sauté onion and tomato with mustard seeds and curry leaves, toss the bread in, finish with coriander."),
    ("French toast", ["Bread", "Eggs", "Milk"], 10, "Whisk eggs with milk and a pinch of salt or sugar, dip the slices, pan-fry until golden."),
    ("Banana milkshake", ["Bananas", "Milk"], 5, "Blend ripe bananas with chilled milk and a little honey."),
    ("Palak dal", ["Leafy greens", "Onions", "Tomatoes"], 30, "Cook dal, add chopped spinach, finish with a tempering of garlic, cumin and onion."),
    ("Paneer bhurji", ["Paneer", "Onions", "Tomatoes", "Coriander"], 15, "Crumble paneer into sautéed onion and tomato with turmeric and garam masala."),
    ("Masala omelette", ["Eggs", "Onions", "Tomatoes", "Coriander"], 10, "Beat eggs with chopped onion, tomato and coriander, cook in a hot pan."),
    ("Curd rice", ["Curd", "Milk"], 10, "Mix cooked rice with curd and a splash of milk, temper with mustard seeds and curry leaves."),
    ("Sweet lassi", ["Curd", "Milk"], 5, "Blend curd with a little milk, sugar and cardamom."),
    ("Rice kheer", ["Milk"], 40, "Simmer milk with a handful of rice until creamy, add sugar, cardamom and nuts."),
    ("Tomato chutney", ["Tomatoes", "Onions", "Coriander"], 20, "Cook tomatoes and onion until soft, blend with chilli and salt, temper with mustard seeds."),
    ("Fruit chaat", ["Apples", "Bananas"], 10, "Chop the fruit, toss with chaat masala and lemon juice."),
    ("Chicken curry", ["Chicken", "Onions", "Tomatoes", "Coriander"], 45, "Brown onion, cook tomato and spices into a masala, simmer the chicken until done."),
    ("Cheese toast", ["Bread", "Cheese", "Tomatoes"], 10, "Top bread with tomato slices and cheese, toast until bubbling."),
    ("Raita", ["Curd", "Onions", "Coriander"], 5, "Whisk curd with chopped onion, coriander, roasted cumin and salt."),
    ("Green chutney", ["Coriander", "Leafy greens"], 10, "Blend coriander and spinach with chilli, lemon and salt."),
    ("Apple shake", ["Apples", "Milk"], 5, "Blend chopped apples with cold milk and a pinch of cinnamon."),
]


def score_pantry(df):
    df = df.copy()
    df["risk"] = risk_model.predict_proba(df[FEATURES])[:, 1]
    df["value_left"] = df.remaining_units * df.unit_price
    df["at_risk_value"] = df.value_left * df.risk
    df["need_days"] = df.remaining_units / df.est_daily_use.clip(lower=0.01)
    df["expiry_date"] = [TODAY + pd.Timedelta(days=int(x)) for x in df.days_left]
    return df


@st.cache_data
def household_options():
    sc = score_pantry(pantry_all)
    hot = sc[sc.risk >= ALERT_TH].groupby("household_id").size()
    h = households.set_index("household_id")
    labels = {}
    for hid in h.index:
        r = h.loc[hid]
        diet = "veg" if r.vegetarian else "non-veg"
        labels[hid] = f"{r.household_type} of {r.household_size} · {diet} · {hid}" if r.household_size > 1 else f"Single · {diet} · {hid}"
    bt = hh_bt.set_index("household_id")
    gain = (bt.wasted_value_habit - bt.wasted_value_ai)
    pool = [x for x in hot[hot >= 5].index if h.loc[x, "household_size"] in (3, 4) and x in gain.index]
    target = gain.quantile(0.6)
    default = min(pool, key=lambda x: abs(gain[x] - target)) if pool else hot.idxmax()
    return labels, default


labels, default_hh = household_options()

# ==================================================================
# Sidebar – choose a demo household
# ==================================================================
with st.sidebar:
    st.markdown(f"### 🥬 {APP_NAME}")
    st.caption(TAGLINE)
    ids = list(labels)
    hid = st.selectbox("Demo household", ids, index=ids.index(default_hh), format_func=lambda x: labels[x])
    H = households.set_index("household_id").loc[hid]
    st.markdown(f"""
**Household profile**
- 👥 {H.household_size} people · {'vegetarian' if H.vegetarian else 'non-vegetarian'}
- 🛒 Shops {H.shops_per_week}× a week
- 🍳 Cooks at home {H.cook_days_per_week} days a week
""")
    st.caption(f"Today: {TODAY:%d %b %Y}. All households and data are simulated.")

if "used" not in st.session_state: st.session_state.used = set()
if "added" not in st.session_state: st.session_state.added = []

pantry = pantry_all[pantry_all.household_id == hid].copy()
pantry["item_key"] = [f"{hid}-{i}" for i in pantry.index]
if st.session_state.added:
    extra = pd.DataFrame([a for a in st.session_state.added if a["household_id"] == hid])
    if len(extra): pantry = pd.concat([pantry, extra], ignore_index=True)
pantry = pantry[~pantry.item_key.isin(st.session_state.used)]
pantry = score_pantry(pantry) if len(pantry) else pantry
past_rate = float(pantry_all.loc[pantry_all.household_id == hid, "hh_past_waste_rate"].iloc[0]) if (pantry_all.household_id == hid).any() else meta["global_waste_rate"]


def status(days, risk):
    if days <= 1 or risk >= 0.75: return "red", "Use today" if days <= 1 else "High risk"
    if days <= 3 or risk >= ALERT_TH: return "amber", "Use soon"
    return "green", "Fresh"


def fmt_qty(cat, units):
    return f"{units:.1f} × {CAT.loc[cat, 'unit']}" if units % 1 else f"{int(units)} × {CAT.loc[cat, 'unit']}"


def item_card(r):
    colour, label = status(r.days_left, r.risk)
    bar_col = {"red": TOMATO, "amber": AMBER, "green": LEAF}[colour]
    return f"""
<div class="item">
  <div class="top"><div class="name">{EMOJI.get(r.category, '🥫')} {r.category}</div><span class="chip {colour}">{label}</span></div>
  <div class="meta">{fmt_qty(r.category, r.remaining_units)} left · expires {r.expiry_date:%a %d %b} ({int(r.days_left)} day{'s' if r.days_left != 1 else ''}) · {r.storage}</div>
  <div class="meta">AI waste risk <b>{r.risk:.0%}</b> · ₹{r.value_left:,.0f} of food</div>
  <div class="bar"><div style="width:{r.risk * 100:.0f}%; background:{bar_col}"></div></div>
  <div class="tip">💡 {TIPS.get(r.category, '')}</div>
</div>"""


def kpi(label, value, note="", kind=""):
    return f'<div class="kpi {kind}"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>'


# ==================================================================
# Hero
# ==================================================================
soon = pantry[(pantry.days_left <= 2)] if len(pantry) else pantry
risky = pantry[pantry.risk >= ALERT_TH] if len(pantry) else pantry
bt = hh_bt.set_index("household_id").loc[hid] if hid in set(hh_bt.household_id) else None
save_month = (bt.wasted_value_habit - bt.wasted_value_ai) if bt is not None else K["waste_rs_month_habit"] - K["waste_rs_month_ai"]
st.markdown(f"""
<div class="hero">
  <div>
    <h1>🥬 {APP_NAME}</h1>
    <div class="tag">{TAGLINE}</div>
    <div class="sub">Knows what is in your kitchen, warns you before food goes to waste, suggests what to cook,
    and tells you how much to buy next time.</div>
  </div>
  <div style="display:flex; flex-direction:column; gap:10px; align-items:flex-end">
    <span class="pill">🔔 {len(risky)} item{'s' if len(risky) != 1 else ''} need attention</span>
    <span class="pill">💰 Save about ₹{save_month:,.0f} a month</span>
  </div>
</div>
""", unsafe_allow_html=True)

home, pan, cook, shop, impact, about = st.tabs(["🏠 Home", "🧺 My Pantry", "🍳 Use It Up", "🛒 Smart List", "🌱 Impact", "ℹ️ About"])

# ==================================================================
# Home
# ==================================================================
with home:
    c1, c2, c3, c4 = st.columns(4)
    at_risk_rs = pantry.at_risk_value.sum() if len(pantry) else 0
    co2 = ((bt.wasted_kg_habit - bt.wasted_kg_ai) if bt is not None else 3.4) * K["co2_per_kg"]
    c1.markdown(kpi("Use within 2 days", f"{len(soon)}", "items expiring soon", "alert" if len(soon) else ""), unsafe_allow_html=True)
    c2.markdown(kpi("Food at risk", f"₹{at_risk_rs:,.0f}", "expected waste if nothing changes", "alert"), unsafe_allow_html=True)
    c3.markdown(kpi("Monthly saving", f"₹{save_month:,.0f}", "with AI alerts + smart list", "good"), unsafe_allow_html=True)
    c4.markdown(kpi("CO₂ avoided", f"{co2:,.1f} kg", "per month, estimated", "good"), unsafe_allow_html=True)

    st.markdown("#### Use these first")
    st.markdown('<div class="section-note">Ranked by the AI: the chance each item ends up in the bin, weighted by its value.</div>', unsafe_allow_html=True)
    if len(pantry):
        top = pantry.sort_values(["at_risk_value"], ascending=False).head(6)
        cols = st.columns(3)
        for k, (_, r) in enumerate(top.iterrows()):
            cols[k % 3].markdown(item_card(r), unsafe_allow_html=True)
    else:
        st.success("Your pantry is empty – nothing to rescue today.")

    st.markdown("#### Where your money was wasted (last 12 weeks)")
    h = history[history.household_id == hid].groupby("category", as_index=False).wasted_value.sum()
    h = h[h.wasted_value > 0].sort_values("wasted_value", ascending=False)
    if len(h):
        chart = alt.Chart(h).mark_bar(cornerRadiusEnd=6, color=LEAF, size=18).encode(
            x=alt.X("wasted_value:Q", title="₹ wasted"), y=alt.Y("category:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("category:N"), alt.Tooltip("wasted_value:Q", format=",.0f", title="₹ wasted")]).properties(height=34 * len(h))
        st.altair_chart(chart, width="stretch")

# ==================================================================
# My Pantry
# ==================================================================
with pan:
    st.markdown("#### Everything in your kitchen")
    st.markdown('<div class="section-note">Sorted by urgency. Tick items you have used and press the button to remove them.</div>', unsafe_allow_html=True)
    if len(pantry):
        show = pantry.sort_values(["days_left", "risk"], ascending=[True, False])
        table = pd.DataFrame({
            "Item": [f"{EMOJI.get(c, '🥫')} {c}" for c in show.category],
            "Left": [fmt_qty(c, u) for c, u in zip(show.category, show.remaining_units)],
            "Stored in": show.storage.str.title(),
            "Expires": [f"{d:%a %d %b}" for d in show.expiry_date],
            "Days left": show.days_left.astype(int),
            "AI waste risk": (show.risk * 100).round(0),
            "Status": [("🔴 " if s == "red" else "🟠 " if s == "amber" else "🟢 ") + l for s, l in (status(d, r) for d, r in zip(show.days_left, show.risk))],
            "Value (₹)": show.value_left.round(0),
        })
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"AI waste risk": st.column_config.ProgressColumn("AI waste risk", min_value=0, max_value=100, format="%d%%"),
                                    "Value (₹)": st.column_config.NumberColumn(format="₹%d")})
        used = st.multiselect("Mark items as used", options=list(show.item_key),
                              format_func=lambda k: (lambda r: f"{r.category} – {fmt_qty(r.category, r.remaining_units)} (expires {r.expiry_date:%d %b})")(show.set_index('item_key').loc[k]))
        if st.button("✅ Remove used items", disabled=not used):
            st.session_state.used.update(used); st.rerun()

    st.markdown("#### Add something you just bought")
    with st.form("add_item"):
        a1, a2, a3, a4 = st.columns(4)
        new_cat = a1.selectbox("Item", list(LIFE), index=list(LIFE).index("Milk"))
        new_qty = a2.number_input(f"How many packs", min_value=1, max_value=20, value=2)
        new_store = a3.selectbox("Where will you keep it?", ["fridge", "pantry", "freezer"])
        new_promo = a4.checkbox("Bought on offer / bulk deal")
        go = st.form_submit_button("🔍 Check waste risk")
    if go:
        store = new_store if new_store in LIFE[new_cat] else max(LIFE[new_cat], key=LIFE[new_cat].get)
        life = LIFE[new_cat][store]
        w = weekly[(weekly.household_id == hid) & (weekly.category == new_cat)]
        daily_use = (w.demand_units.tail(4).mean() / 7) if len(w) else H.household_size * CAT.loc[new_cat, "rate"]
        on_hand = pantry.loc[pantry.category == new_cat, "remaining_units"].sum() if len(pantry) else 0
        row = {"category": new_cat, "storage": store, "shelf_life_days": life, "quantity_units": new_qty, "unit_price": CAT.loc[new_cat, "price"],
               "bought_on_promo": int(new_promo), "on_hand_units": on_hand, "est_daily_use": max(daily_use, 0.01),
               "household_size": H.household_size, "cook_days": H.cook_days_per_week, "hh_past_waste_rate": past_rate}
        row["cover_days"] = (on_hand + new_qty) / row["est_daily_use"]; row["cover_ratio"] = row["cover_days"] / life
        risk = float(risk_model.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])
        need = (on_hand + new_qty) / row["est_daily_use"]
        colour, label = status(life, risk)
        m1, m2, m3 = st.columns(3)
        m1.markdown(kpi("Expires", f"{(TODAY + pd.Timedelta(days=life)):%d %b}", f"{life} days in the {store}"), unsafe_allow_html=True)
        m2.markdown(kpi("AI waste risk", f"{risk:.0%}", label, "alert" if colour != "green" else "good"), unsafe_allow_html=True)
        m3.markdown(kpi("Time to finish", f"{need:.0f} days", f"at your usual pace, incl. {on_hand:.1f} already at home"), unsafe_allow_html=True)
        better = [s for s, d in LIFE[new_cat].items() if d > life]
        if risk >= ALERT_TH:
            fewer = max(1, math.floor(life * row["est_daily_use"] - on_hand))
            st.warning(f"This is likely to be wasted. You usually finish about **{life * row['est_daily_use']:.1f} packs** in {life} days"
                       + (f" – buying **{fewer}** would be enough." if fewer < new_qty else ".")
                       + (f" Storing it in the **{better[0]}** keeps it for {LIFE[new_cat][better[0]]} days." if better else ""))
        else:
            st.success("Looks fine – you are likely to use this up before it expires.")
        st.session_state.added.append({**row, "household_id": hid, "remaining_units": float(new_qty), "days_left": life,
                                       "day": 84, "item_key": f"{hid}-new-{len(st.session_state.added)}"})

# ==================================================================
# Use It Up – recipes
# ==================================================================
with cook:
    st.markdown("#### Cook these to rescue your food")
    st.markdown('<div class="section-note">Recipes ranked by how much at-risk food they use up.</div>', unsafe_allow_html=True)
    if len(pantry):
        risk_by_cat = pantry.groupby("category").at_risk_value.sum()
        have = set(pantry.category)
        ranked = []
        for name, ing, mins, steps in RECIPES:
            uses = [i for i in ing if i in have]
            if not uses: continue
            ranked.append((sum(risk_by_cat.get(i, 0) for i in uses), len(uses) / len(ing), name, ing, mins, steps, uses))
        ranked.sort(reverse=True)
        cols = st.columns(2)
        for k, (rescue, share, name, ing, mins, steps, uses) in enumerate(ranked[:8]):
            chips = "".join(
                f'<span class="chip {"red" if risk_by_cat.get(i, 0) > 20 else "green"}">{EMOJI.get(i, "")} {i}</span>' if i in uses
                else f'<span class="chip" style="background:#F1F3F2;color:#8A9893">+ {i}</span>' for i in ing)
            cols[k % 2].markdown(f"""
<div class="recipe">
  <div class="top" style="display:flex;justify-content:space-between"><div class="title">🍳 {name}</div><span class="chip green">⏱ {mins} min</span></div>
  <div class="meta" style="color:{MUTED};font-size:.88rem;margin-top:4px">Rescues about ₹{rescue:,.0f} of food · you have {len(uses)} of {len(ing)} ingredients</div>
  <div class="uses">{chips}</div>
  <div class="steps">{steps}</div>
</div>""", unsafe_allow_html=True)
    st.caption("Red = an item the AI expects you to waste. Grey = something you would need to add.")

# ==================================================================
# Smart List – Model 2
# ==================================================================
with shop:
    st.markdown("#### Next week's smart shopping list")
    st.markdown('<div class="section-note">The AI forecasts how much your household will actually use next week, then subtracts what is already at home.</div>', unsafe_allow_html=True)
    w = weekly[weekly.household_id == hid].sort_values(["category", "week"])
    rows = []
    for cat_name, g in w.groupby("category"):
        if cat_name == "Rice & atta" or len(g) < 5: continue
        s = g.demand_units.values
        normal = g.loc[g.week != 8, "demand_units"].mean()
        x = pd.DataFrame([{"lag_1": s[-1], "lag_2": s[-2], "lag_3": s[-3], "avg_4w": s[-4:].mean(), "avg_normal_weeks": normal,
                           "festival_week": 0, "festival_in_last_4w": 1, "household_size": H.household_size,
                           "cat_code": sorted(weekly.category.unique()).index(cat_name)}])[cons_feats]
        f = max(0.0, float(cons_model.predict(x)[0]))
        on_hand = pantry.loc[pantry.category == cat_name, "remaining_units"].sum() if len(pantry) else 0
        buy = max(0, math.ceil(f * 1.10 - on_hand - 0.25))
        hist = history[(history.household_id == hid) & (history.category == cat_name)]
        usual = hist.value.sum() / CAT.loc[cat_name, "price"] / 12
        life = LIFE[cat_name][CAT.loc[cat_name, "best_storage"]]
        rows.append({"Item": f"{EMOJI.get(cat_name, '')} {cat_name}", "Pack": CAT.loc[cat_name, "unit"], "Forecast use": round(f, 1),
                     "At home": round(on_hand, 1), "Buy": buy, "You usually buy": round(usual, 1),
                     "Tip": "Split into 2 trips" if life <= 4 and buy >= 2 else ("Nothing needed" if buy == 0 else ""),
                     "_saving": max(0, usual - buy) * CAT.loc[cat_name, "price"], "_cat": cat_name})
    sl = pd.DataFrame(rows)
    if len(sl):
        s1, s2, s3 = st.columns(3)
        s1.markdown(kpi("Items to buy", f"{(sl.Buy > 0).sum()}", f"{int(sl.Buy.sum())} packs in total"), unsafe_allow_html=True)
        s2.markdown(kpi("Less than usual", f"{max(0, sl['You usually buy'].sum() - sl.Buy.sum()):.0f} packs", "this week"), unsafe_allow_html=True)
        s3.markdown(kpi("Saved this week", f"₹{sl._saving.sum():,.0f}", "vs your usual weekly basket – partly because you already have enough at home", "good"), unsafe_allow_html=True)
        st.dataframe(sl.drop(columns=["_saving", "_cat"]).sort_values("Buy", ascending=False), hide_index=True, width="stretch",
                     column_config={"Buy": st.column_config.NumberColumn("Buy (packs)", format="%d"),
                                    "Forecast use": st.column_config.NumberColumn("Forecast use (packs)")})
        long = sl.melt(id_vars=["_cat"], value_vars=["You usually buy", "Buy"], var_name="Basket", value_name="Packs")
        long["Basket"] = long.Basket.replace({"Buy": "AI smart list"})
        st.altair_chart(alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("_cat:N", title=None), x=alt.X("Packs:Q", title="Packs"), yOffset=alt.YOffset("Basket:N"),
            color=alt.Color("Basket:N", scale=alt.Scale(range=["#A9B3B0", LEAF]), legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("_cat:N", title="Item"), "Basket", "Packs"]).properties(height=40 * len(sl)), width="stretch")

# ==================================================================
# Impact
# ==================================================================
with impact:
    st.markdown("#### What the AI achieves")
    st.markdown(f'<div class="section-note">Back-test: the last 3 weeks of all {K["households"]} simulated households replayed with their usual shopping habit, and again with the AI smart list.</div>', unsafe_allow_html=True)
    i1, i2, i3, i4 = st.columns(4)
    i1.markdown(kpi("Food waste cut", f"{K['waste_cut_pct']}%", f"₹{K['waste_rs_month_habit']} → ₹{K['waste_rs_month_ai']} per home per month", "good"), unsafe_allow_html=True)
    i2.markdown(kpi("Less food binned", f"{K['waste_kg_month_habit'] - K['waste_kg_month_ai']:.1f} kg", f"{K['waste_kg_month_habit']} → {K['waste_kg_month_ai']} kg per home per month", "good"), unsafe_allow_html=True)
    i3.markdown(kpi("Grocery spend", f"−₹{K['spend_rs_month_habit'] - K['spend_rs_month_ai']}", "per home per month", "good"), unsafe_allow_html=True)
    i4.markdown(kpi("Smarter alerts", f"{K['alerts_per_hh_week_ai']}/week", f"{K['useful_alerts_ai_pct']}% useful vs {K['useful_alerts_dates_pct']}% for date reminders"), unsafe_allow_html=True)

    st.markdown("#### If more homes used it")
    homes = st.slider("Households using the app", 1_000, 1_000_000, 100_000, step=1_000, format="%d")
    saved_rs = (K["waste_rs_month_habit"] - K["waste_rs_month_ai"]) * 12 * homes
    saved_t = (K["waste_kg_month_habit"] - K["waste_kg_month_ai"]) * 12 * homes / 1000
    j1, j2, j3 = st.columns(3)
    j1.markdown(kpi("Money saved a year", f"₹{saved_rs / 1e7:,.1f} crore", "by households", "good"), unsafe_allow_html=True)
    j2.markdown(kpi("Food saved a year", f"{saved_t:,.0f} tonnes", "kept out of the bin", "good"), unsafe_allow_html=True)
    j3.markdown(kpi("CO₂e avoided a year", f"{saved_t * K['co2_per_kg']:,.0f} tonnes", f"at {K['co2_per_kg']} kg CO₂e per kg of food", "good"), unsafe_allow_html=True)

    with st.expander("How the AI models perform"):
        st.markdown(f"**Model 1 – Will this item be wasted?** Four models compared on the last two weeks of purchases. "
                    f"Selected: **{K['selected_model']}** (ROC-AUC {K['model_roc_auc']}, catches {K['model_recall']:.0%} of wasted items, "
                    f"{K['model_precision']:.0%} of its alerts are right).")
        st.dataframe(model_cmp.round(3), hide_index=True, width="stretch")
        st.markdown(f"**Model 2 – How much will we use next week?** Random Forest forecast error {K['forecast_wape_ai']}% "
                    f"vs {K['forecast_wape_baseline']}% for a 4-week average.")
        st.caption("Selection rule: keep models within 0.01 ROC-AUC of the best, pick the highest F1-score, prefer the explainable model on a tie.")

# ==================================================================
# About
# ==================================================================
with about:
    st.markdown(f"#### About {APP_NAME}")
    st.write(f"{APP_NAME} is an AI-powered kitchen assistant. It keeps track of what is in the fridge and pantry, "
             "predicts which items are likely to be thrown away, and helps the household use them in time and buy smarter next time.")
    st.markdown("#### Why it matters")
    b1, b2, b3 = st.columns(3)
    b1.markdown(kpi("Household share", "60%", "of the world's food waste happens at home (UNEP, 2024)"), unsafe_allow_html=True)
    b2.markdown(kpi("In Indian homes", "≈55 kg", "of food wasted per person per year (reported)"), unsafe_allow_html=True)
    b3.markdown(kpi("In our data", f"{K['share_spend_wasted_pct']}%", f"of grocery spend wasted – about ₹{K['waste_per_household_month_rs']} a month"), unsafe_allow_html=True)
    st.markdown("#### How it works")
    h1, h2, h3, h4 = st.columns(4)
    for col, t, d in [(h1, "1. Log", "Items are added from a receipt, barcode or a quick form, with where they are stored."),
                      (h2, "2. Predict", f"A {K['selected_model']} model scores each item's chance of being wasted."),
                      (h3, "3. Act", "Ranked alerts, storage tips and recipes that use risky items first."),
                      (h4, "4. Shop smarter", "A forecast of next week's use builds the shopping list.")]:
        col.markdown(f'<div class="kpi"><div class="label">{t}</div><div class="note" style="color:{INK};font-size:.92rem;margin-top:6px">{d}</div></div>', unsafe_allow_html=True)
    with st.expander("Data, privacy and limitations"):
        st.markdown("""
- This prototype runs on **synthetic data**: 400 simulated households over 12 weeks.
- Savings rest on stated assumptions: 50% of the waste is saved when an alert is acted on; fewer people act when they get many alerts; 2.5 kg CO₂e per kg of food.
- Built for India's **DPDP Act 2023**: consent, data used only to cut waste, deletion on request, partners see only anonymous totals.
- Expiry estimates are guidance – always check the printed date, look and smell before eating.
- Next steps: receipt scanning (OCR), barcode and photo recognition, and a pilot with real households.
""")
    st.caption("Built with Python, pandas, scikit-learn and Streamlit.")
