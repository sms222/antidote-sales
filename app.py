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
    low = filename.lower()
    if low.endswith(".csv"):
        df = pd.read_csv(buf)
    elif low.endswith(".parquet"):
        df = pd.read_parquet(buf)
    else:
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
    df["_month"] = ts.dt.to_period("M").dt.start_time.dt.date
    df["_month_label"] = ts.dt.strftime("%b %Y")
    df["_quarter"] = ts.dt.to_period("Q").dt.start_time.dt.date
    df["_quarter_label"] = ts.dt.to_period("Q").astype(str).str.replace("Q", " Q")
    df["_year"] = ts.dt.to_period("Y").dt.start_time.dt.date
    df["_year_label"] = ts.dt.strftime("%Y")
    df["_doctype"] = df[COL["doc"]].astype(str).str.extract(r"^([A-Za-z]+)")[0].fillna("-")
    df["_no_cost"] = df[COL["totalcost"]] == 0
    if COL["cust"] in df.columns:
        df["_member"] = df[COL["cust"]].astype(str).str.strip().str.upper() != "CASH"
    else:
        df["_member"] = False
    return df


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


@st.cache_data(show_spinner=False)
def item_summary(frame: pd.DataFrame, exclude_no_cost: bool) -> pd.DataFrame:
    """One row per product, from the filtered frame."""
    src = frame[~frame["_no_cost"]] if exclude_no_cost else frame
    if src.empty:
        return pd.DataFrame()

    g = src.groupby([COL["item"], COL["desc"]], dropna=False)
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
    out = out.rename(columns={COL["item"]: "Item Code", COL["desc"]: "Description"})
    return out


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


@st.cache_data(show_spinner=False)
def membership_monthly(scoped_all_dates: pd.DataFrame, view_start, view_end) -> tuple:
    """
    Recruitment and member-activity by month.

    First-purchase dates are computed from `scoped_all_dates` (respects every
    filter except the date range) so a narrow date selection doesn't make a
    long-standing customer look newly recruited. The returned table is then
    limited to months touching the selected view window.

    Returns (monthly_df, first_month_is_data_start: bool).
    """
    named = scoped_all_dates[scoped_all_dates["_member"]]
    if named.empty:
        return pd.DataFrame(), False

    first_dt = named.groupby(COL["cust"])["_date"].min()
    recruit_month = pd.to_datetime(first_dt).dt.to_period("M")
    recruits = recruit_month.value_counts().sort_index()
    recruits.index = recruits.index.to_timestamp()

    m = named.copy()
    m["_m"] = pd.to_datetime(m["_month"])
    activity = m.groupby("_m").agg(
        MemberRevenue=(COL["amount"], "sum"),
        MemberTransactions=(COL["doc"], "nunique"),
        MemberDays=("_date", "nunique"),
    ).reset_index().rename(columns={"_m": "Month"})
    activity["MemberTCperDay"] = activity["MemberTransactions"] / activity["MemberDays"]
    activity["MemberAvgBasket"] = activity["MemberRevenue"] / activity["MemberTransactions"]

    out = activity.merge(
        recruits.rename("NewMembers"), left_on="Month", right_index=True, how="left")
    out["NewMembers"] = out["NewMembers"].fillna(0).astype(int)
    out = out.sort_values("Month")

    view_start_m = pd.Timestamp(view_start).to_period("M").to_timestamp()
    view_end_m = pd.Timestamp(view_end).to_period("M").to_timestamp()
    out = out[(out["Month"] >= view_start_m) & (out["Month"] <= view_end_m)]

    data_start_m = pd.to_datetime(scoped_all_dates["_date"]).min()
    first_shown_is_start = (not out.empty) and (out["Month"].min() <= pd.Timestamp(data_start_m).to_period("M").to_timestamp())

    return out, first_shown_is_start


# ----------------------------------------------------------------------------
# Sidebar — upload
# ----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Sales data")
    upload = st.file_uploader("POS export", type=["xlsx", "xls", "csv", "parquet"],
                              label_visibility="collapsed")
    st.markdown(
        '<p class="caption-note">Nothing is saved. The file lives in this browser '
        'session only and disappears when you close the tab.</p>',
        unsafe_allow_html=True)

if upload is None:
    st.title("Sales Analytics")
    st.markdown(
        "Upload a POS sales listing to begin. Every figure is calculated from the "
        "file as exported — no values are adjusted, filled in, or removed.\n\n"
        "Large exports load faster as Parquet than as Excel.")
    st.stop()

try:
    raw = load_data(upload.getvalue(), upload.name)
except Exception as exc:
    st.error(f"That file could not be read. {exc}")
    st.stop()

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

    grain = st.radio("Group by", ["Daily", "Weekly", "Monthly", "Quarterly", "Annually"],
                     index=0, horizontal=True)

    st.markdown("### Filters")
    agents = sorted(raw[COL["agent"]].dropna().unique().tolist()) if COL["agent"] in raw else []
    agent_sel = st.multiselect("Sales agent", agents,
                               help="Blank agent rows are included unless you filter here.")

    doctypes = sorted(raw["_doctype"].unique().tolist())
    doc_sel = st.multiselect("Document type", doctypes, default=doctypes)

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
    if doc_sel:
        out = out[out["_doctype"].isin(doc_sel)]
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
prev_end = start_d - pd.Timedelta(days=1)
prev_start = prev_end - pd.Timedelta(days=span - 1)
prev = scoped[(scoped["_date"] >= prev_start.date()) & (scoped["_date"] <= prev_end.date())]

if df.empty:
    st.warning("No rows match these filters. Widen the date range or clear a filter.")
    st.stop()

period_label = f"{start_d:%d %b %Y} – {end_d:%d %b %Y}"

filters_note = ", ".join(filter(None, [
    f"agents: {', '.join(agent_sel)}" if agent_sel else "",
    f"document types: {', '.join(doc_sel)}" if len(doc_sel) != len(doctypes) else "",
    f"customer: {cust_scope}" if cust_scope != "All" else "",
    f"item contains '{search.strip()}'" if search.strip() else "",
])) or "none"

# ----------------------------------------------------------------------------
# Shared computations
# ----------------------------------------------------------------------------

GRAIN_KEY = {"Daily": "_date", "Weekly": "_week", "Monthly": "_month_label",
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
top_customers = customers_tbl.head(int(top_n)) if len(customers_tbl) else customers_tbl

member_monthly, first_is_data_start = membership_monthly(scoped, start_d, end_d)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------

st.title("Sales Analytics")
st.markdown(
    f"<p class='caption-note'>{period_label} &nbsp;·&nbsp; {span} days &nbsp;·&nbsp; "
    f"{len(df):,} lines &nbsp;·&nbsp; {df[COL['doc']].nunique():,} transactions</p>",
    unsafe_allow_html=True)

(tab_overview, tab_stock, tab_staff, tab_customers,
 tab_membership, tab_compare, tab_report) = st.tabs(
    ["Overview", "Stock", "Staff", "Customers", "Membership", "Compare", "Report"])

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

# ----------------------------------------------------------------------------
# Customers
# ----------------------------------------------------------------------------

with tab_customers:
    m_rev = df[df["_member"]][COL["amount"]].sum()
    c_rev = df[~df["_member"]][COL["amount"]].sum()
    m_tx = df[df["_member"]][COL["doc"]].nunique()
    c_tx = df[~df["_member"]][COL["doc"]].nunique()

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

    if customers_tbl.empty:
        st.info("No named-customer transactions in this selection — every line is CASH.")
    else:
        st.subheader(f"Top {top_n} customers by profit")
        st.caption("CASH transactions are excluded from this ranking by definition — "
                   "there is no single customer behind that name.")
        st.dataframe(
            top_customers.style.format({
                "Revenue": "RM {:,.2f}", "Profit": "RM {:,.2f}", "Margin %": "{:.1f}%",
                "Avg basket": "RM {:,.2f}", "Transactions": "{:,.0f}", "Lines": "{:,.0f}"}),
            use_container_width=True, hide_index=True)
        st.download_button("Download CSV", top_customers.to_csv(index=False).encode(),
                           f"top_{top_n}_customers_{start_d}_{end_d}.csv", "text/csv")

# ----------------------------------------------------------------------------
# Membership
# ----------------------------------------------------------------------------

with tab_membership:
    if member_monthly.empty:
        st.info("No named-customer activity in this selection.")
    else:
        if first_is_data_start:
            st.warning(
                "The first month shown counts every named customer's first purchase "
                "within this file as a 'new member' — some may have joined before the "
                "data starts. Treat recruitment figures for that month as an upper "
                "bound, not a fact.")

        fm = make_subplots(specs=[[{"secondary_y": True}]])
        fm.add_bar(x=member_monthly["Month"], y=member_monthly["NewMembers"],
                   name="New members", marker_color=SAND,
                   hovertemplate="%{x|%b %Y}<br>%{y:,.0f} new<extra></extra>")
        fm.add_scatter(x=member_monthly["Month"], y=member_monthly["MemberTCperDay"],
                       name="Member TC/day", mode="lines+markers",
                       line=dict(color=TEAL, width=2), secondary_y=True,
                       hovertemplate="%{x|%b %Y}<br>%{y:,.1f} tx/day<extra></extra>")
        fm.update_layout(height=380, hovermode="x unified",
                         legend=dict(orientation="h", y=1.12, x=0),
                         margin=dict(t=20, b=10, l=0, r=0), plot_bgcolor="rgba(0,0,0,0)")
        fm.update_yaxes(gridcolor="#EDEFEF", title_text="New members", secondary_y=False)
        fm.update_yaxes(showgrid=False, title_text="TC / day", secondary_y=True)
        st.plotly_chart(fm, use_container_width=True)

        show_m = member_monthly.rename(columns={
            "Month": "Month", "NewMembers": "New members",
            "MemberRevenue": "Member revenue", "MemberTransactions": "Member transactions",
            "MemberTCperDay": "Member TC/day", "MemberAvgBasket": "Member avg basket"})
        show_m = show_m[["Month", "New members", "Member transactions", "Member TC/day",
                         "Member revenue", "Member avg basket"]]
        st.dataframe(
            show_m.style.format({
                "Month": lambda d: d.strftime("%b %Y"),
                "New members": "{:,.0f}", "Member transactions": "{:,.0f}",
                "Member TC/day": "{:,.1f}", "Member revenue": "RM {:,.2f}",
                "Member avg basket": "RM {:,.2f}"}),
            use_container_width=True, hide_index=True)
        st.download_button("Download CSV", show_m.to_csv(index=False).encode(),
                           f"membership_monthly_{start_d}_{end_d}.csv", "text/csv")
        st.caption("Recruitment month = the calendar month of a customer's first "
                   "purchase found in the uploaded file. TC = transaction count.")

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
    with cb:
        st.markdown("**Period B**")
        b_sel = st.date_input("Period B range", value=(prev_start.date(), prev_end.date()),
                              min_value=min_d, max_value=max_d, key="cmp_b",
                              label_visibility="collapsed")

    a_start, a_end = _resolve_range(a_sel, (start_d, end_d))
    b_start, b_end = _resolve_range(b_sel, (prev_start.date(), prev_end.date()))

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
            figc = go.Figure()
            figc.add_bar(x=["Revenue", "Gross profit"], y=[ka["revenue"], ka["profit"]],
                        name="Period A", marker_color=TEAL)
            figc.add_bar(x=["Revenue", "Gross profit"], y=[kb["revenue"], kb["profit"]],
                        name="Period B", marker_color=SAND)
            figc.update_layout(barmode="group", height=320, title="Revenue & profit",
                              margin=dict(t=40, b=10, l=0, r=0),
                              plot_bgcolor="rgba(0,0,0,0)",
                              legend=dict(orientation="h", y=1.15, x=0))
            figc.update_yaxes(gridcolor="#EDEFEF", title_text="RM")
            st.plotly_chart(figc, use_container_width=True)

        with right:
            figm = go.Figure()
            figm.add_bar(x=["Margin %", "PP (RM)"], y=[ka["margin"], ka["basket"]],
                        name="Period A", marker_color=TEAL)
            figm.add_bar(x=["Margin %", "PP (RM)"], y=[kb["margin"], kb["basket"]],
                        name="Period B", marker_color=SAND)
            figm.update_layout(barmode="group", height=320, title="Margin & purchase per head",
                              margin=dict(t=40, b=10, l=0, r=0),
                              plot_bgcolor="rgba(0,0,0,0)",
                              legend=dict(orientation="h", y=1.15, x=0))
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
        sec_staff = st.checkbox("Staff performance", True)
        sec_customers = st.checkbox(f"Top {top_n} customers", True)
        sec_membership = st.checkbox("Membership by month", True)

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
                    staff=agents_tbl if sec_staff else None,
                    customers=top_customers if sec_customers else None,
                    membership=member_monthly if sec_membership else None,
                    membership_caveat=first_is_data_start,
                    sections={"summary": sec_summary, "trend": sec_trend,
                              "stock_qty": sec_qty, "stock_rev": sec_rev,
                              "stock_margin": sec_margin, "staff": sec_staff,
                              "customers": sec_customers, "membership": sec_membership},
                )
            except Exception as exc:
                st.error(f"The report could not be built. {exc}")
            else:
                st.success("Report ready.")
                st.download_button(
                    "Download PDF", pdf_bytes,
                    file_name=f"sales_report_{start_d}_{end_d}.pdf",
                    mime="application/pdf", type="primary")
