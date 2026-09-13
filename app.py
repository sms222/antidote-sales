"""
Sales Analytics — MY Antidote Wellness

Design rules
------------
1. The POS export is the source of truth. No value is altered, imputed or
   dropped. Only calendar fields and flags are derived.
2. Nothing is stored. The uploaded file lives in the browser session only.
3. Every tab here also has a matching section in the PDF report.
"""

import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from report import build_pdf

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Sales Analytics", page_icon="▦",
                   layout="wide", initial_sidebar_state="expanded")

INK, TEAL, SAND, CLAY, MUTED = "#16302F", "#0E7C7B", "#B8935A", "#A6452E", "#6B7C7B"

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
      h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
      [data-testid="stMetricValue"] {{ font-size: 1.5rem; color: {INK}; }}
      [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
      .caption-note {{ color: {MUTED}; font-size: 0.82rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

BUSINESS = "MY Antidote Wellness Sdn Bhd"

COL = {
    "date": "Posting Date", "datetime": "Posting Date Time", "doc": "Doc. No.",
    "cust": "Cust Name", "branch": "Sales Branch", "agent": "Sales Agent",
    "item": "Item Code", "desc": "Description", "qty": "Qty", "uom": "UOM",
    "disc": "Discount", "price": "Selling Price", "amount": "Amount",
    "tax": "Tax Amt.", "unitcost": "Unit Cost", "totalcost": "Total Cost",
    "profit": "Profit/Loss", "margin": "Margin",
}
REQUIRED = [COL["doc"], COL["qty"], COL["amount"], COL["totalcost"], COL["profit"]]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner="Reading the export…", max_entries=2)
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    df = pd.read_excel(buf)

    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing expected columns: " + ", ".join(missing))

    if COL["datetime"] in df.columns:
        ts = pd.to_datetime(df[COL["datetime"]], errors="coerce")
    elif COL["date"] in df.columns:
        ts = pd.to_datetime(df[COL["date"]], errors="coerce", dayfirst=True)
    else:
        raise ValueError("No date column found.")
    if ts.isna().all():
        raise ValueError("Dates could not be read.")

    df["_ts"] = ts
    df["_date"] = ts.dt.date
    df["_hour"] = ts.dt.hour
    df["_weekday"] = ts.dt.day_name()
    df["_week"] = ts.dt.to_period("W").dt.start_time.dt.date
    df["_week_label"] = ts.dt.strftime("%G-W%V")
    df["_month"] = ts.dt.to_period("M").dt.start_time.dt.date
    df["_month_label"] = ts.dt.strftime("%Y-%m")
    df["_quarter"] = ts.dt.to_period("Q").dt.start_time.dt.date
    df["_quarter_label"] = ts.dt.to_period("Q").astype(str).str.replace("Q", " Q")
    df["_year"] = ts.dt.to_period("Y").dt.start_time.dt.date
    df["_year_label"] = ts.dt.strftime("%Y")
    df["_no_cost"] = df[COL["totalcost"]] == 0
    if COL["cust"] in df.columns:
        df["_member"] = df[COL["cust"]].astype(str).str.strip().str.upper() != "CASH"
    else:
        df["_member"] = False
    return df


ITEM_MASTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "item_master.csv")


@st.cache_data(show_spinner=False)
def load_item_master(file_bytes=None, filename=None) -> pd.DataFrame:
    """
    Item Code -> Item Group lookup.

    Loads the copy bundled in the repo (item_master.csv) by default, so the
    person never has to re-upload it. If a file is passed (a session-only
    override from the sidebar), that takes priority instead.
    """
    if file_bytes is not None:
        buf = io.BytesIO(file_bytes)
        low = (filename or "").lower()
        im = pd.read_csv(buf) if low.endswith(".csv") else pd.read_excel(buf)
    elif os.path.exists(ITEM_MASTER_PATH):
        im = pd.read_csv(ITEM_MASTER_PATH)
    else:
        return pd.DataFrame(columns=["Item Code", "Item Group"])

    im.columns = [str(c).strip() for c in im.columns]
    needed = {"Item Code", "Item Group"}
    if not needed.issubset(im.columns):
        raise ValueError("Item master must have 'Item Code' and 'Item Group' columns.")
    im["Item Code"] = im["Item Code"].astype(str).str.strip()
    im["Item Group"] = im["Item Group"].astype(str).str.strip()
    return im[["Item Code", "Item Group"]].drop_duplicates(subset="Item Code")


def rm(x, dp=0):
    return "—" if pd.isna(x) else f"RM {x:,.{dp}f}"


def pct(x, dp=1):
    return "—" if (pd.isna(x) or np.isinf(x)) else f"{x:,.{dp}f}%"


def kpis(frame: pd.DataFrame, exclude_no_cost: bool) -> dict:
    """Revenue counts every row. Only profit and margin respond to the lens."""
    blank = dict.fromkeys(
        ["revenue", "profit", "margin", "invoices", "basket", "units", "lines"], np.nan)
    if frame.empty:
        return blank

    revenue = frame[COL["amount"]].sum()
    tc = frame[COL["doc"]].nunique()
    pf = frame[~frame["_no_cost"]] if exclude_no_cost else frame
    profit, base = pf[COL["profit"]].sum(), pf[COL["amount"]].sum()

    return {
        "revenue": revenue,
        "profit": profit,
        "margin": (profit / base * 100) if base else np.nan,
        "invoices": tc,
        "basket": (revenue / tc) if tc else np.nan,
        "units": frame[COL["qty"]].sum(),
        "lines": len(frame),
    }


def delta_str(now, prev, is_pct=False):
    if pd.isna(now) or pd.isna(prev) or prev == 0:
        return None
    return f"{now - prev:+.1f} pp" if is_pct else f"{(now - prev) / abs(prev) * 100:+.1f}%"


def build_trend(frame: pd.DataFrame, key: str, sort_key: str, exclude_no_cost: bool) -> pd.DataFrame:
    """Group a frame into periods and compute KPIs for each. Shared by the
    Overview trend and by each side of the Compare tab."""
    if frame.empty:
        return pd.DataFrame()
    group_cols = [key] if key == sort_key else [key, sort_key]
    t = (
        frame.groupby(group_cols)
        .apply(lambda g: pd.Series(kpis(g, exclude_no_cost)), include_groups=False)
        .reset_index()
        .sort_values(sort_key)
    )
    days_in = frame.groupby(key)["_date"].nunique().rename("days")
    t = t.merge(days_in, left_on=key, right_index=True, how="left")
    t["tc_per_day"] = t["invoices"] / t["days"]
    return t


def multi_entity_trend(frame: pd.DataFrame, entity_col: str, entities: list,
                       key: str, sort_key: str, value_col: str) -> pd.DataFrame:
    """
    Period-by-period totals for a handful of named entities (top items, top
    agents, top customers) — what makes the 'Group by' granularity mean
    something on the Stock, Staff, and Customers tabs, not just Overview.
    Returns a wide frame: index = period label, one column per entity.
    """
    f = frame[frame[entity_col].isin(entities)]
    if f.empty:
        return pd.DataFrame()
    group_cols = [key, entity_col] if key == sort_key else [sort_key, key, entity_col]
    g = f.groupby(group_cols)[value_col].sum().reset_index()
    sort_cols = [key] if key == sort_key else [sort_key, key]
    order = g[sort_cols].drop_duplicates().sort_values(sort_cols[0])[key].tolist()
    pivot = g.pivot(index=key, columns=entity_col, values=value_col)
    pivot = pivot.reindex(order).fillna(0)
    return pivot


def plot_entity_trend(pivot: pd.DataFrame, ylabel: str):
    fig = go.Figure()
    palette = [TEAL, SAND, CLAY, "#4A7A78", "#C9A66B", "#7A9E9C", "#8C5A45"]
    for i, col in enumerate(pivot.columns):
        fig.add_scatter(x=pivot.index, y=pivot[col], mode="lines+markers",
                        name=str(col)[:26], line=dict(color=palette[i % len(palette)], width=2))
    fig.update_layout(height=340, margin=dict(t=10, b=10, l=0, r=0),
                      plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.15, x=0))
    fig.update_yaxes(gridcolor="#EDEFEF", title_text=ylabel)
    return fig


@st.cache_data(show_spinner=False)
def item_summary(frame: pd.DataFrame, exclude_no_cost: bool) -> pd.DataFrame:
    """One row per product, from the filtered frame. Carries Item Group if present."""
    src = frame[~frame["_no_cost"]] if exclude_no_cost else frame
    if src.empty:
        return pd.DataFrame()

    group_cols = [COL["item"], COL["desc"]]
    if "_item_group" in src.columns:
        group_cols.append("_item_group")

    g = src.groupby(group_cols, dropna=False)
    out = g.agg(
        Qty=(COL["qty"], "sum"),
        Revenue=(COL["amount"], "sum"),
        Profit=(COL["profit"], "sum"),
        Lines=(COL["doc"], "size"),
        Transactions=(COL["doc"], "nunique"),
        NoCostLines=("_no_cost", "sum"),
    ).reset_index()
    out["Margin %"] = np.where(out["Revenue"] != 0,
                               out["Profit"] / out["Revenue"] * 100, np.nan)
    rename = {COL["item"]: "Item Code", COL["desc"]: "Description"}
    if "_item_group" in src.columns:
        rename["_item_group"] = "Item Group"
    out = out.rename(columns=rename)
    return out


@st.cache_data(show_spinner=False)
def group_summary(frame: pd.DataFrame, exclude_no_cost: bool) -> pd.DataFrame:
    """One row per Item Group."""
    if "_item_group" not in frame.columns or frame.empty:
        return pd.DataFrame()
    src = frame[~frame["_no_cost"]] if exclude_no_cost else frame
    if src.empty:
        return pd.DataFrame()

    out = src.groupby("_item_group").agg(
        Qty=(COL["qty"], "sum"),
        Revenue=(COL["amount"], "sum"),
        Profit=(COL["profit"], "sum"),
        Transactions=(COL["doc"], "nunique"),
        Items=(COL["item"], "nunique"),
    ).reset_index().rename(columns={"_item_group": "Item Group"})
    out["Margin %"] = np.where(out["Revenue"] != 0, out["Profit"] / out["Revenue"] * 100, np.nan)
    return out.sort_values("Revenue", ascending=False)


@st.cache_data(show_spinner=False)
def agent_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per Sales Agent, exactly as recorded. Blank -> 'Not recorded'."""
    if frame.empty:
        return pd.DataFrame()
    f = frame.copy()
    f["Sales Agent"] = f[COL["agent"]].fillna("Not recorded") if COL["agent"] in f else "Not recorded"
    out = f.groupby("Sales Agent").agg(
        Revenue=(COL["amount"], "sum"),
        Profit=(COL["profit"], "sum"),
        Transactions=(COL["doc"], "nunique"),
        Lines=(COL["doc"], "size"),
    ).reset_index()
    out["Margin %"] = np.where(out["Revenue"] != 0, out["Profit"] / out["Revenue"] * 100, np.nan)
    out["Avg basket"] = np.where(out["Transactions"] != 0, out["Revenue"] / out["Transactions"], np.nan)
    return out.sort_values("Profit", ascending=False)


@st.cache_data(show_spinner=False)
def customer_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per named customer. CASH transactions are excluded here by design."""
    named = frame[frame["_member"]]
    if named.empty:
        return pd.DataFrame()
    out = named.groupby(COL["cust"]).agg(
        Revenue=(COL["amount"], "sum"),
        Profit=(COL["profit"], "sum"),
        Transactions=(COL["doc"], "nunique"),
        Lines=(COL["doc"], "size"),
    ).reset_index().rename(columns={COL["cust"]: "Customer"})
    out["Margin %"] = np.where(out["Revenue"] != 0, out["Profit"] / out["Revenue"] * 100, np.nan)
    out["Avg basket"] = np.where(out["Transactions"] != 0, out["Revenue"] / out["Transactions"], np.nan)
    return out.sort_values("Profit", ascending=False)


# ----------------------------------------------------------------------------
# Sidebar — upload
# ----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Sales data")
    upload = st.file_uploader("POS export (.xlsx)", type=["xlsx"],
                              label_visibility="collapsed")
    st.markdown(
        '<p class="caption-note">Nothing is saved. The file lives in this browser '
        'session only and disappears when you close the tab.</p>',
        unsafe_allow_html=True)

try:
    item_master = load_item_master()
    im_source = "bundled with the app" if len(item_master) else "none found"
except Exception as exc:
    st.sidebar.error(f"Item master could not be read: {exc}")
    item_master = pd.DataFrame(columns=["Item Code", "Item Group"])
    im_source = "failed to load"

with st.sidebar:
    st.caption(f"{len(item_master):,} item codes loaded ({im_source}).")

if upload is None:
    st.title("Sales Analytics")
    st.markdown(
        "Upload your POS sales listing (.xlsx) to begin. Every figure is calculated "
        "from the file as exported — no values are adjusted, filled in, or removed.")
    st.stop()

try:
    raw = load_data(upload.getvalue(), upload.name)
except Exception as exc:
    st.error(f"That file could not be read. {exc}")
    st.stop()

if len(item_master):
    raw = raw.merge(item_master.rename(columns={"Item Code": COL["item"]}),
                    on=COL["item"], how="left")
    raw = raw.rename(columns={"Item Group": "_item_group"})
    raw["_item_group"] = raw["_item_group"].fillna("Unclassified")
else:
    raw["_item_group"] = "Unclassified"

# ----------------------------------------------------------------------------
# Sidebar — filters
# ----------------------------------------------------------------------------

min_d, max_d = raw["_date"].min(), raw["_date"].max()

with st.sidebar:
    st.markdown("### Period")
    date_sel = st.date_input("Date range", value=(min_d, max_d),
                             min_value=min_d, max_value=max_d,
                             label_visibility="collapsed")
    if isinstance(date_sel, tuple) and len(date_sel) == 2:
        start_d, end_d = date_sel
    else:
        start_d = end_d = date_sel[0] if isinstance(date_sel, tuple) else date_sel

    # Only offer a granularity if the selected range actually spans more than
    # one such period — "Annually" on 13 days of data would show a single
    # bar, which isn't a trend, it's just the total relabeled.
    _in_range = raw[(raw["_date"] >= start_d) & (raw["_date"] <= end_d)]
    grain_options = ["Daily"]
    if _in_range["_week"].nunique() >= 2:
        grain_options.append("Weekly")
    if _in_range["_month"].nunique() >= 2:
        grain_options.append("Monthly")
    if _in_range["_quarter"].nunique() >= 2:
        grain_options.append("Quarterly")
    if _in_range["_year"].nunique() >= 2:
        grain_options.append("Annually")

    prior_grain = st.session_state.get("grain_pref", "Daily")
    grain_index = grain_options.index(prior_grain) if prior_grain in grain_options else 0
    grain = st.radio("Group by", grain_options, index=grain_index, horizontal=True)
    st.session_state["grain_pref"] = grain

    _hidden = [g for g in ["Daily", "Weekly", "Monthly", "Quarterly", "Annually"]
              if g not in grain_options]
    if _hidden:
        st.caption(f"{', '.join(_hidden)} hidden — the selected range doesn't "
                   f"cover more than one such period yet.")

    st.markdown("### Filters")
    agents = sorted(raw[COL["agent"]].dropna().unique().tolist()) if COL["agent"] in raw else []
    agent_sel = st.multiselect("Sales agent", agents,
                               help="Blank agent rows are included unless you filter here.")

    cust_scope = st.radio("Customer", ["All", "Members only", "Non-members (CASH)"],
                          help="CASH means no member was recorded at point of sale.")

    search = st.text_input("Item contains", "", placeholder="e.g. VITAMIN C")

    st.markdown("### Margin lens")
    exclude_no_cost = st.toggle(
        "Exclude lines with no cost recorded", value=False,
        help=("These lines show 100% margin because Total Cost is zero. The file does "
              "not say whether that is a genuine giveaway or an uncosted item. "
              "Revenue always counts every line."))

    st.markdown("### Rankings")
    top_n = st.number_input("Show top N", 5, 100, 30, step=5,
                            help="Applies to product, staff, and customer rankings.")
    margin_floor = st.number_input("Minimum revenue for margin ranking (RM)",
                                   0, 100_000, 200, step=50,
                                   help="Keeps single-unit sales out of the margin table.")

# ----------------------------------------------------------------------------
# Apply filters
# ----------------------------------------------------------------------------

def apply_non_date(df):
    out = df
    if agent_sel:
        out = out[out[COL["agent"]].isin(agent_sel)]
    if cust_scope == "Members only" and "_member" in out:
        out = out[out["_member"]]
    elif cust_scope == "Non-members (CASH)" and "_member" in out:
        out = out[~out["_member"]]
    if search.strip():
        out = out[out[COL["desc"]].astype(str)
                  .str.contains(search.strip(), case=False, na=False)]
    return out


scoped = apply_non_date(raw)
df = scoped[(scoped["_date"] >= start_d) & (scoped["_date"] <= end_d)]

span = (end_d - start_d).days + 1
prev_end = pd.Timestamp(start_d) - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=span - 1)
prev = scoped[(scoped["_date"] >= prev_start.date()) & (scoped["_date"] <= prev_end.date())]

if df.empty:
    st.warning("No rows match these filters. Widen the date range or clear a filter.")
    st.stop()

period_label = f"{start_d:%d %b %Y} – {end_d:%d %b %Y}"

filters_note = ", ".join(filter(None, [
    f"agents: {', '.join(agent_sel)}" if agent_sel else "",
    f"customer: {cust_scope}" if cust_scope != "All" else "",
    f"item contains '{search.strip()}'" if search.strip() else "",
])) or "none"

# ----------------------------------------------------------------------------
# Shared computations
# ----------------------------------------------------------------------------

GRAIN_KEY = {"Daily": "_date", "Weekly": "_week_label", "Monthly": "_month_label",
            "Quarterly": "_quarter_label", "Annually": "_year_label"}
GRAIN_SORT = {"Daily": "_date", "Weekly": "_week", "Monthly": "_month",
             "Quarterly": "_quarter", "Annually": "_year"}
key, sort_key = GRAIN_KEY[grain], GRAIN_SORT[grain]

trend = build_trend(df, key, sort_key, exclude_no_cost)

items = item_summary(df, exclude_no_cost)
top_qty = items.nlargest(int(top_n), "Qty") if len(items) else items
top_rev = items.nlargest(int(top_n), "Revenue").copy() if len(items) else items
if len(top_rev):
    top_rev["Cumulative %"] = top_rev["Revenue"].cumsum() / items["Revenue"].sum() * 100
gated = items[items["Revenue"] >= margin_floor] if len(items) else items
top_margin = gated.nlargest(int(top_n), "Margin %") if len(gated) else gated

agents_tbl = agent_summary(df)
customers_tbl = customer_summary(df)
groups_tbl = group_summary(df, exclude_no_cost)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

st.title("Sales Analytics")
st.markdown(
    f"<p class='caption-note'>{period_label} &nbsp;·&nbsp; {span} days &nbsp;·&nbsp; "
    f"{len(df):,} lines &nbsp;·&nbsp; {df[COL['doc']].nunique():,} transactions</p>",
    unsafe_allow_html=True)

(tab_overview, tab_stock, tab_groups, tab_staff, tab_customers,
 tab_compare, tab_report) = st.tabs(
    ["Overview", "Stock", "Item Groups", "Staff", "Customers", "Compare", "Report"])

# ----------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------

with tab_overview:
    k = kpis(df, exclude_no_cost)
    kp = kpis(prev, exclude_no_cost) if not prev.empty else dict.fromkeys(k, np.nan)

    c = st.columns(6)
    c[0].metric("Revenue", rm(k["revenue"]), delta_str(k["revenue"], kp["revenue"]))
    c[1].metric("Gross profit", rm(k["profit"]), delta_str(k["profit"], kp["profit"]))
    c[2].metric("Margin", pct(k["margin"]), delta_str(k["margin"], kp["margin"], True))
    c[3].metric("Transactions", f"{k['invoices']:,}", delta_str(k["invoices"], kp["invoices"]))
    c[4].metric("Purchase per head", rm(k["basket"], 2), delta_str(k["basket"], kp["basket"]))
    c[5].metric("Units sold", f"{k['units']:,.0f}", delta_str(k["units"], kp["units"]))

    n_nocost = int(df["_no_cost"].sum())
    if n_nocost:
        other = kpis(df, not exclude_no_cost)["margin"]
        lens = "excluded from" if exclude_no_cost else "included in"
        st.caption(f"{n_nocost:,} of {len(df):,} lines have no cost recorded and are "
                   f"currently {lens} the margin figure. The other way round, margin "
                   f"reads {pct(other)}.")

    if cust_scope == "All":
        m_tx = df[df["_member"]][COL["doc"]].nunique()
        t_tx = df[COL["doc"]].nunique()
        st.caption(f"{m_tx:,} of {t_tx:,} transactions ({m_tx / t_tx * 100:.1f}%) carry a "
                   f"customer name. The rest are CASH, meaning no member was recorded "
                   f"at point of sale.")

    st.divider()
    st.subheader("Revenue and profit over time")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=trend[key], y=trend["revenue"], name="Revenue", marker_color=TEAL,
                opacity=0.85, hovertemplate="%{x}<br>Revenue RM %{y:,.0f}<extra></extra>")
    fig.add_bar(x=trend[key], y=trend["profit"], name="Gross profit", marker_color=SAND,
                opacity=0.9, hovertemplate="%{x}<br>Profit RM %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=trend[key], y=trend["margin"], name="Margin %", mode="lines+markers",
                    line=dict(color=CLAY, width=2), secondary_y=True,
                    hovertemplate="%{x}<br>Margin %{y:.1f}%<extra></extra>")
    fig.update_layout(barmode="group", height=420, hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0),
                      margin=dict(t=30, b=10, l=0, r=0), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title_text="RM", secondary_y=False, gridcolor="#EDEFEF")
    fig.update_yaxes(title_text="Margin %", secondary_y=True, showgrid=False)
    fig.update_xaxes(gridcolor="#EDEFEF")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Cumulative revenue")
        cum = trend[[key, "revenue"]].copy()
        cum["cumulative"] = cum["revenue"].cumsum()
        f2 = go.Figure()
        f2.add_scatter(x=cum[key], y=cum["cumulative"], mode="lines",
                       line=dict(color=INK, width=2.5), fill="tozeroy",
                       fillcolor="rgba(14,124,123,0.12)",
                       hovertemplate="%{x}<br>RM %{y:,.0f}<extra></extra>")
        f2.update_layout(height=300, margin=dict(t=10, b=10, l=0, r=0),
                         plot_bgcolor="rgba(0,0,0,0)")
        f2.update_yaxes(gridcolor="#EDEFEF")
        st.plotly_chart(f2, use_container_width=True)

    with right:
        st.subheader("Transactions per day and purchase per head")
        f3 = make_subplots(specs=[[{"secondary_y": True}]])
        f3.add_bar(x=trend[key], y=trend["tc_per_day"], name="TC per day",
                   marker_color="#C9D6D5",
                   hovertemplate="%{x}<br>%{y:,.1f} transactions/day<extra></extra>")
        f3.add_scatter(x=trend[key], y=trend["basket"], name="PP", mode="lines+markers",
                       line=dict(color=TEAL, width=2), secondary_y=True,
                       hovertemplate="%{x}<br>RM %{y:,.2f}<extra></extra>")
        f3.update_layout(height=300, hovermode="x unified",
                         legend=dict(orientation="h", y=1.15, x=0),
                         margin=dict(t=10, b=10, l=0, r=0), plot_bgcolor="rgba(0,0,0,0)")
        f3.update_yaxes(gridcolor="#EDEFEF", secondary_y=False)
        f3.update_yaxes(showgrid=False, secondary_y=True)
        st.plotly_chart(f3, use_container_width=True)

    st.divider()
    st.subheader(f"{grain} breakdown")
    show = trend[[key, "revenue", "profit", "margin", "invoices", "tc_per_day",
                  "basket", "units", "lines"]].rename(columns={
        key: "Period", "revenue": "Revenue", "profit": "Gross profit",
        "margin": "Margin %", "invoices": "Transactions", "tc_per_day": "TC per day",
        "basket": "PP", "units": "Units", "lines": "Lines"})
    st.dataframe(show.style.format({
        "Revenue": "RM {:,.2f}", "Gross profit": "RM {:,.2f}", "Margin %": "{:.1f}%",
        "PP": "RM {:,.2f}", "TC per day": "{:,.1f}", "Transactions": "{:,.0f}",
        "Units": "{:,.0f}", "Lines": "{:,.0f}"}),
        use_container_width=True, hide_index=True)
    st.caption("TC = transaction count. PP = purchase per head, revenue ÷ transactions.")

# ----------------------------------------------------------------------------
# Stock
# ----------------------------------------------------------------------------

with tab_stock:
    if items.empty:
        st.warning("No product rows in this selection.")
    else:
        top6_items = top_rev["Description"].head(6).tolist() if len(top_rev) else []
        if top6_items:
            st.subheader(f"{grain} revenue trend — top 6 items by revenue")
            pv = multi_entity_trend(df, COL["desc"], top6_items, key, sort_key, COL["amount"])
            if len(pv):
                st.plotly_chart(plot_entity_trend(pv, "Revenue RM"), use_container_width=True)
            st.caption("Change 'Group by' in the sidebar to see these at a different "
                       "granularity.")
            st.divider()

        s1, s2, s3 = st.tabs(["By quantity", "By revenue", "By margin"])
        fmt = {"Revenue": "RM {:,.2f}", "Profit": "RM {:,.2f}", "Margin %": "{:.1f}%",
               "Qty": "{:,.0f}", "Transactions": "{:,.0f}", "Lines": "{:,.0f}",
               "NoCostLines": "{:,.0f}", "Cumulative %": "{:.1f}%"}
        cols = ["Item Code", "Description", "Qty", "Revenue", "Profit", "Margin %",
                "Transactions"]

        with s1:
            st.subheader(f"Top {top_n} by quantity sold")
            st.dataframe(top_qty[cols].style.format(fmt),
                         use_container_width=True, hide_index=True)
            st.download_button("Download CSV", top_qty.to_csv(index=False).encode(),
                               f"top_{top_n}_quantity_{start_d}_{end_d}.csv", "text/csv")

        with s2:
            st.subheader(f"Top {top_n} by revenue")
            share = top_rev["Revenue"].sum() / items["Revenue"].sum() * 100
            st.caption(f"These {len(top_rev)} products account for {share:.1f}% of revenue "
                       f"in this period, out of {len(items):,} products sold.")
            fp = go.Figure()
            fp.add_bar(x=top_rev["Description"].astype(str).str.slice(0, 30),
                       y=top_rev["Revenue"], marker_color=TEAL,
                       hovertemplate="%{x}<br>RM %{y:,.0f}<extra></extra>")
            fp.update_layout(height=360, margin=dict(t=20, b=10, l=0, r=0),
                             plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                             xaxis_tickangle=-45)
            fp.update_yaxes(gridcolor="#EDEFEF", title_text="RM")
            st.plotly_chart(fp, use_container_width=True)
            st.dataframe(top_rev[cols + ["Cumulative %"]].style.format(fmt),
                         use_container_width=True, hide_index=True)
            st.download_button("Download CSV", top_rev.to_csv(index=False).encode(),
                               f"top_{top_n}_revenue_{start_d}_{end_d}.csv", "text/csv")

        with s3:
            st.subheader(f"Top {top_n} by margin")
            st.caption(
                f"Ranked by margin percentage, limited to products with at least "
                f"RM {margin_floor:,.0f} of revenue. Without that floor the list fills with "
                f"single-unit sales and uncosted lines at 100%. Read the profit column "
                f"alongside the percentage — 80% on RM 40 is trivia.")
            st.dataframe(top_margin[cols + ["NoCostLines"]].style.format(fmt),
                         use_container_width=True, hide_index=True)
            st.download_button("Download CSV", top_margin.to_csv(index=False).encode(),
                               f"top_{top_n}_margin_{start_d}_{end_d}.csv", "text/csv")

# ----------------------------------------------------------------------------
# Item Groups
# ----------------------------------------------------------------------------

with tab_groups:
    if groups_tbl.empty:
        st.info("No item group data available. Check the item master in the sidebar.")
    else:
        unclassified = groups_tbl[groups_tbl["Item Group"] == "Unclassified"]
        if not unclassified.empty and unclassified["Revenue"].iloc[0] > 0:
            share = unclassified["Revenue"].iloc[0] / groups_tbl["Revenue"].sum() * 100
            st.warning(
                f"RM {unclassified['Revenue'].iloc[0]:,.0f} of revenue ({share:.1f}%) "
                f"comes from item codes not found in the item master — new products, "
                f"typos, or codes retired since the master was last updated.")

        fg = go.Figure()
        fg.add_bar(x=groups_tbl["Item Group"], y=groups_tbl["Revenue"], name="Revenue",
                   marker_color=TEAL, hovertemplate="%{x}<br>RM %{y:,.0f}<extra></extra>")
        fg.add_bar(x=groups_tbl["Item Group"], y=groups_tbl["Profit"], name="Gross profit",
                   marker_color=SAND, hovertemplate="%{x}<br>RM %{y:,.0f}<extra></extra>")
        fg.update_layout(barmode="group", height=380, margin=dict(t=10, b=10, l=0, r=0),
                         plot_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-30,
                         legend=dict(orientation="h", y=1.1, x=0))
        fg.update_yaxes(gridcolor="#EDEFEF", title_text="RM")
        st.plotly_chart(fg, use_container_width=True)

        st.dataframe(
            groups_tbl.style.format({
                "Qty": "{:,.0f}", "Revenue": "RM {:,.2f}", "Profit": "RM {:,.2f}",
                "Margin %": "{:.1f}%", "Transactions": "{:,.0f}", "Items": "{:,.0f}"}),
            use_container_width=True, hide_index=True)
        st.download_button("Download CSV", groups_tbl.to_csv(index=False).encode(),
                           f"item_groups_{start_d}_{end_d}.csv", "text/csv")

        st.divider()
        st.subheader("Drill into a group")
        pick = st.selectbox("Item Group", groups_tbl["Item Group"].tolist())
        drill = items[items["Item Group"] == pick] if len(items) else pd.DataFrame()
        if len(drill):
            drill_top = drill.nlargest(min(20, len(drill)), "Profit")
            st.caption(f"Top {len(drill_top)} items in '{pick}' by profit, out of "
                       f"{len(drill)} distinct items sold in this group.")
            st.dataframe(
                drill_top[["Item Code", "Description", "Qty", "Revenue", "Profit",
                          "Margin %", "Transactions"]].style.format({
                    "Qty": "{:,.0f}", "Revenue": "RM {:,.2f}", "Profit": "RM {:,.2f}",
                    "Margin %": "{:.1f}%", "Transactions": "{:,.0f}"}),
                use_container_width=True, hide_index=True)
        else:
            st.info("No items in this group for the current selection.")

# ----------------------------------------------------------------------------
# Staff
# ----------------------------------------------------------------------------

with tab_staff:
    if agents_tbl.empty:
        st.warning("No agent data in this selection.")
    else:
        st.subheader("Performance by Sales Agent")
        st.caption(
            "Shown exactly as recorded in the POS 'Sales Agent' field. Some entries "
            "are order channels (e.g. WhatsApp, GrabMart) rather than staff members — "
            "read names accordingly. 'Not recorded' is revenue with no agent logged at "
            "the till, not a person.")

        not_rec = agents_tbl[agents_tbl["Sales Agent"] == "Not recorded"]
        if not not_rec.empty and not_rec["Revenue"].iloc[0] > 0:
            share = not_rec["Revenue"].iloc[0] / agents_tbl["Revenue"].sum() * 100
            st.warning(
                f"RM {not_rec['Revenue'].iloc[0]:,.0f} of revenue ({share:.1f}% of this "
                f"selection) has no agent recorded. Fixing data entry at the till would "
                f"make this leaderboard trustworthy.")

        fa = go.Figure()
        fa.add_bar(x=agents_tbl["Sales Agent"], y=agents_tbl["Profit"],
                   marker_color=TEAL, name="Profit",
                   hovertemplate="%{x}<br>Profit RM %{y:,.0f}<extra></extra>")
        fa.update_layout(height=360, margin=dict(t=10, b=10, l=0, r=0),
                         plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                         xaxis_tickangle=-30)
        fa.update_yaxes(gridcolor="#EDEFEF", title_text="Gross profit (RM)")
        st.plotly_chart(fa, use_container_width=True)

        st.dataframe(
            agents_tbl.style.format({
                "Revenue": "RM {:,.2f}", "Profit": "RM {:,.2f}", "Margin %": "{:.1f}%",
                "Avg basket": "RM {:,.2f}", "Transactions": "{:,.0f}", "Lines": "{:,.0f}"}),
            use_container_width=True, hide_index=True)
        st.download_button("Download CSV", agents_tbl.to_csv(index=False).encode(),
                           f"staff_performance_{start_d}_{end_d}.csv", "text/csv")

        st.divider()
        top6_agents = agents_tbl["Sales Agent"].head(6).tolist()
        st.subheader(f"{grain} revenue trend — top 6 agents")
        df_agent = df.copy()
        df_agent["_agent_filled"] = df_agent[COL["agent"]].fillna("Not recorded")
        pv = multi_entity_trend(df_agent, "_agent_filled", top6_agents, key, sort_key,
                                COL["amount"])
        if len(pv):
            st.plotly_chart(plot_entity_trend(pv, "Revenue RM"), use_container_width=True)
        st.caption("Change 'Group by' in the sidebar to see this at a different granularity.")

# ----------------------------------------------------------------------------
# Customers
# ----------------------------------------------------------------------------

with tab_customers:
    m_rev = df[df["_member"]][COL["amount"]].sum()
    c_rev = df[~df["_member"]][COL["amount"]].sum()

    cc = st.columns(4)
    cc[0].metric("Named-customer revenue", rm(m_rev))
    cc[1].metric("CASH revenue", rm(c_rev))
    cc[2].metric("Named customers", f"{customers_tbl.shape[0]:,}" if len(customers_tbl) else "0")
    if len(customers_tbl):
        repeat = int((customers_tbl["Transactions"] > 1).sum())
        cc[3].metric("Repeat customers", f"{repeat:,}",
                    f"{repeat / len(customers_tbl) * 100:.0f}% of named")
    else:
        cc[3].metric("Repeat customers", "0")

    st.divider()

    cust_top_n = 30
    if customers_tbl.empty:
        st.info("No named-customer transactions in this selection — every line is CASH.")
    else:
        cust_top_n = st.number_input("Show top", 5, 500, 30, step=5, key="cust_top_n")
        st.caption("CASH transactions are excluded from every ranking below by "
                   "definition — there is no single customer behind that name.")

        fmt_cust = {"Revenue": "RM {:,.2f}", "Profit": "RM {:,.2f}", "Margin %": "{:.1f}%",
                   "Avg basket": "RM {:,.2f}", "Transactions": "{:,.0f}", "Lines": "{:,.0f}"}
        cols_cust = ["Customer", "Revenue", "Profit", "Transactions", "Margin %", "Avg basket"]

        cu1, cu2, cu3 = st.tabs(["By profit", "By transactions", "By total sales"])

        with cu1:
            st.subheader(f"Top {cust_top_n} customers by profit")
            t = customers_tbl.nlargest(int(cust_top_n), "Profit")
            st.dataframe(t[cols_cust].style.format(fmt_cust),
                         use_container_width=True, hide_index=True)
            st.download_button("Download CSV", t.to_csv(index=False).encode(),
                               f"top_{cust_top_n}_customers_profit_{start_d}_{end_d}.csv",
                               "text/csv")

        with cu2:
            st.subheader(f"Top {cust_top_n} customers by number of transactions")
            st.caption("Transactions = unique Doc. No. — how many separate times this "
                       "customer paid, not how many line items they bought.")
            t = customers_tbl.nlargest(int(cust_top_n), "Transactions")
            st.dataframe(t[cols_cust].style.format(fmt_cust),
                         use_container_width=True, hide_index=True)
            st.download_button("Download CSV", t.to_csv(index=False).encode(),
                               f"top_{cust_top_n}_customers_transactions_{start_d}_{end_d}.csv",
                               "text/csv")

        with cu3:
            st.subheader(f"Top {cust_top_n} customers by total sales")
            t = customers_tbl.nlargest(int(cust_top_n), "Revenue")
            st.dataframe(t[cols_cust].style.format(fmt_cust),
                         use_container_width=True, hide_index=True)
            st.download_button("Download CSV", t.to_csv(index=False).encode(),
                               f"top_{cust_top_n}_customers_revenue_{start_d}_{end_d}.csv",
                               "text/csv")

        st.divider()
        top6_cust = customers_tbl.nlargest(6, "Profit")["Customer"].tolist()
        st.subheader(f"{grain} revenue trend — top 6 customers by profit")
        pv = multi_entity_trend(df[df["_member"]], COL["cust"], top6_cust, key, sort_key,
                                COL["amount"])
        if len(pv):
            st.plotly_chart(plot_entity_trend(pv, "Revenue RM"), use_container_width=True)
        st.caption("Change 'Group by' in the sidebar to see this at a different granularity.")

# ----------------------------------------------------------------------------
# Compare
# ----------------------------------------------------------------------------

def _resolve_range(sel, fallback):
    if isinstance(sel, tuple) and len(sel) == 2:
        return sel
    d = sel[0] if isinstance(sel, tuple) else sel
    return d, d if d else fallback


with tab_compare:
    st.subheader("Compare two periods")
    st.caption("Both periods respect the agent, document-type, customer, and item "
               "filters in the sidebar. Only the date ranges differ below.")

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Period A**")
        a_sel = st.date_input("Period A range", value=(start_d, end_d),
                              min_value=min_d, max_value=max_d, key="cmp_a",
                              label_visibility="collapsed")
    # Default Period B to "the period immediately before A", clamped into the
    # data's actual date range — the unclamped previous period may fall
    # entirely before the earliest date on file (e.g. when A already spans
    # the whole file), which Streamlit's date_input rejects outright.
    b_default_start = min(max(prev_start.date(), min_d), max_d)
    b_default_end = min(max(prev_end.date(), min_d), max_d)
    if b_default_start > b_default_end:
        b_default_start = b_default_end

    with cb:
        st.markdown("**Period B**")
        b_sel = st.date_input("Period B range", value=(b_default_start, b_default_end),
                              min_value=min_d, max_value=max_d, key="cmp_b",
                              label_visibility="collapsed")

    a_start, a_end = _resolve_range(a_sel, (start_d, end_d))
    b_start, b_end = _resolve_range(b_sel, (b_default_start, b_default_end))

    frame_a = scoped[(scoped["_date"] >= a_start) & (scoped["_date"] <= a_end)]
    frame_b = scoped[(scoped["_date"] >= b_start) & (scoped["_date"] <= b_end)]

    if frame_a.empty or frame_b.empty:
        st.warning("One of the selected periods has no data. Adjust the range above.")
    else:
        ka = kpis(frame_a, exclude_no_cost)
        kb = kpis(frame_b, exclude_no_cost)
        span_a, span_b = (a_end - a_start).days + 1, (b_end - b_start).days + 1

        label_a = f"A: {a_start:%d %b %y}–{a_end:%d %b %y}"
        label_b = f"B: {b_start:%d %b %y}–{b_end:%d %b %y}"

        comp_rows = [
            ["Revenue", rm(ka["revenue"]), rm(kb["revenue"]),
             delta_str(ka["revenue"], kb["revenue"])],
            ["Gross profit", rm(ka["profit"]), rm(kb["profit"]),
             delta_str(ka["profit"], kb["profit"])],
            ["Margin", pct(ka["margin"]), pct(kb["margin"]),
             delta_str(ka["margin"], kb["margin"], True)],
            ["Transactions", f"{ka['invoices']:,}", f"{kb['invoices']:,}",
             delta_str(ka["invoices"], kb["invoices"])],
            ["Purchase per head", rm(ka["basket"], 2), rm(kb["basket"], 2),
             delta_str(ka["basket"], kb["basket"])],
            ["TC per day", f"{ka['invoices'] / span_a:,.1f}", f"{kb['invoices'] / span_b:,.1f}",
             delta_str(ka["invoices"] / span_a, kb["invoices"] / span_b)],
            ["Units sold", f"{ka['units']:,.0f}", f"{kb['units']:,.0f}",
             delta_str(ka["units"], kb["units"])],
        ]
        comp_df = pd.DataFrame(comp_rows, columns=["Metric", label_a, label_b, "Δ (A vs B)"])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        if span_a != span_b:
            st.caption(
                f"Period A is {span_a} days, Period B is {span_b} days. Totals aren't "
                f"like-for-like at different lengths — lean on TC per day, PP, and "
                f"Margin rather than raw Revenue or Units when the spans differ.")

        left, right = st.columns(2)
        with left:
            st.markdown("**Revenue & profit**")
            figc = go.Figure()
            figc.add_bar(x=["Revenue", "Gross profit"], y=[ka["revenue"], ka["profit"]],
                        name="Period A", marker_color=TEAL)
            figc.add_bar(x=["Revenue", "Gross profit"], y=[kb["revenue"], kb["profit"]],
                        name="Period B", marker_color=SAND)
            figc.update_layout(barmode="group", height=320,
                              margin=dict(t=30, b=10, l=0, r=0),
                              plot_bgcolor="rgba(0,0,0,0)",
                              legend=dict(orientation="h", y=1.12, x=0))
            figc.update_yaxes(gridcolor="#EDEFEF", title_text="RM")
            st.plotly_chart(figc, use_container_width=True)

        with right:
            st.markdown("**Margin & purchase per head**")
            figm = go.Figure()
            figm.add_bar(x=["Margin %", "PP (RM)"], y=[ka["margin"], ka["basket"]],
                        name="Period A", marker_color=TEAL)
            figm.add_bar(x=["Margin %", "PP (RM)"], y=[kb["margin"], kb["basket"]],
                        name="Period B", marker_color=SAND)
            figm.update_layout(barmode="group", height=320,
                              margin=dict(t=30, b=10, l=0, r=0),
                              plot_bgcolor="rgba(0,0,0,0)",
                              legend=dict(orientation="h", y=1.12, x=0))
            figm.update_yaxes(gridcolor="#EDEFEF")
            st.plotly_chart(figm, use_container_width=True)

        st.divider()
        st.subheader(f"{grain} trend, aligned by position in each period")
        st.caption(
            f"Point 1 is the first {grain.lower()[:-2] or grain.lower()} of each period, "
            f"point 2 the second, and so on — this lines up 'week 1 of this month' "
            f"against 'week 1 of last month' even when the calendar dates don't match.")

        ta = build_trend(frame_a, key, sort_key, exclude_no_cost)
        tb = build_trend(frame_b, key, sort_key, exclude_no_cost)

        fig_t = make_subplots(specs=[[{"secondary_y": False}]])
        fig_t.add_scatter(x=list(range(1, len(ta) + 1)), y=ta["revenue"],
                          mode="lines+markers", name=f"Revenue — {label_a}",
                          line=dict(color=TEAL, width=2))
        fig_t.add_scatter(x=list(range(1, len(tb) + 1)), y=tb["revenue"],
                          mode="lines+markers", name=f"Revenue — {label_b}",
                          line=dict(color=SAND, width=2, dash="dot"))
        fig_t.update_layout(height=340, margin=dict(t=10, b=10, l=0, r=0),
                            plot_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", y=1.15, x=0),
                            xaxis_title=f"{grain} # into period")
        fig_t.update_yaxes(gridcolor="#EDEFEF", title_text="Revenue RM")
        st.plotly_chart(fig_t, use_container_width=True)

        with st.expander("Full breakdown for both periods"):
            st.markdown(f"**{label_a}**")
            st.dataframe(ta[[key, "revenue", "profit", "margin", "invoices", "basket"]]
                        .rename(columns={key: "Period", "revenue": "Revenue",
                                        "profit": "Profit", "margin": "Margin %",
                                        "invoices": "Transactions", "basket": "PP"})
                        .style.format({"Revenue": "RM {:,.2f}", "Profit": "RM {:,.2f}",
                                      "Margin %": "{:.1f}%", "PP": "RM {:,.2f}",
                                      "Transactions": "{:,.0f}"}),
                        use_container_width=True, hide_index=True)
            st.markdown(f"**{label_b}**")
            st.dataframe(tb[[key, "revenue", "profit", "margin", "invoices", "basket"]]
                        .rename(columns={key: "Period", "revenue": "Revenue",
                                        "profit": "Profit", "margin": "Margin %",
                                        "invoices": "Transactions", "basket": "PP"})
                        .style.format({"Revenue": "RM {:,.2f}", "Profit": "RM {:,.2f}",
                                      "Margin %": "{:.1f}%", "PP": "RM {:,.2f}",
                                      "Transactions": "{:,.0f}"}),
                        use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

with tab_report:
    st.subheader("Generate a PDF report")
    st.markdown("The report uses the period and filters currently set in the sidebar.")

    r1, r2, r3 = st.columns(3)
    with r1:
        sec_summary = st.checkbox("Summary KPIs", True)
        sec_trend = st.checkbox(f"{grain} performance and chart", True)
    with r2:
        sec_qty = st.checkbox(f"Top {top_n} by quantity", True)
        sec_rev = st.checkbox(f"Top {top_n} by revenue", True)
        sec_margin = st.checkbox(f"Top {top_n} by margin", True)
    with r3:
        sec_groups = st.checkbox("Item groups", True)
        sec_staff = st.checkbox("Staff performance", True)
        sec_customers = st.checkbox(f"Top {cust_top_n} customers (by profit)", True)

    business = st.text_input("Business name on the report", BUSINESS)

    if st.button("Build report", type="primary"):
        with st.spinner("Building PDF…"):
            try:
                pdf_bytes = build_pdf(
                    business_name=business,
                    period_label=period_label,
                    grain=grain,
                    kpi=kpis(df, exclude_no_cost),
                    kpi_prev=kpis(prev, exclude_no_cost) if not prev.empty else None,
                    trend=trend if sec_trend else None,
                    trend_key=key,
                    stock_qty=top_qty if sec_qty else None,
                    stock_rev=top_rev if sec_rev else None,
                    stock_margin=top_margin if sec_margin else None,
                    margin_floor=margin_floor,
                    exclude_no_cost=exclude_no_cost,
                    no_cost_lines=int(df["_no_cost"].sum()),
                    total_lines=len(df),
                    filters_note=filters_note,
                    groups=groups_tbl if sec_groups and len(groups_tbl) else None,
                    staff=agents_tbl if sec_staff else None,
                    customers=(customers_tbl.nlargest(int(cust_top_n), "Profit")
                              if sec_customers and len(customers_tbl) else None),
                    sections={"summary": sec_summary, "trend": sec_trend,
                              "stock_qty": sec_qty, "stock_rev": sec_rev,
                              "stock_margin": sec_margin, "groups": sec_groups,
                              "staff": sec_staff, "customers": sec_customers},
                )
            except Exception as exc:
                st.error(f"The report could not be built. {exc}")
            else:
                st.success("Report ready.")
                st.download_button(
                    "Download PDF", pdf_bytes,
                    file_name=f"sales_report_{start_d}_{end_d}.pdf",
                    mime="application/pdf", type="primary")
