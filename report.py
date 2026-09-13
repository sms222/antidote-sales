"""
PDF report builder.

Uses matplotlib rather than Plotly image export: Plotly's static export
needs kaleido, which pulls a headless Chrome binary and fails silently on
Streamlit Community Cloud. Matplotlib renders to PNG with no system deps.

No Streamlit imports here, so this module can be tested on its own.
"""

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

INK = colors.HexColor("#16302F")
TEAL = colors.HexColor("#0E7C7B")
SAND = colors.HexColor("#B8935A")
CLAY = colors.HexColor("#A6452E")
MUTED = colors.HexColor("#6B7C7B")
HAIR = colors.HexColor("#D8DEDD")
BAND = colors.HexColor("#F3F6F5")

INK_HEX, TEAL_HEX, SAND_HEX, CLAY_HEX = "#16302F", "#0E7C7B", "#B8935A", "#A6452E"


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def _styles():
    s = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=s["Title"], fontName="Helvetica-Bold",
                                fontSize=19, leading=23, textColor=INK,
                                alignment=0, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=s["Normal"], fontName="Helvetica",
                              fontSize=9.5, leading=13, textColor=MUTED,
                              spaceAfter=14),
        "h2": ParagraphStyle("h2", parent=s["Heading2"], fontName="Helvetica-Bold",
                             fontSize=12.5, leading=16, textColor=INK,
                             spaceBefore=16, spaceAfter=7),
        "body": ParagraphStyle("b", parent=s["Normal"], fontName="Helvetica",
                               fontSize=9, leading=13, textColor=INK),
        "note": ParagraphStyle("n", parent=s["Normal"], fontName="Helvetica-Oblique",
                               fontSize=8.2, leading=11.5, textColor=MUTED,
                               spaceBefore=5),
        "warn": ParagraphStyle("w", parent=s["Normal"], fontName="Helvetica-Bold",
                               fontSize=8.4, leading=12, textColor=CLAY,
                               spaceBefore=5, spaceAfter=4),
        "foot": ParagraphStyle("f", parent=s["Normal"], fontName="Helvetica",
                               fontSize=7.5, textColor=MUTED, alignment=TA_RIGHT),
    }


def _fmt_rm(x, dp=2):
    try:
        return f"{float(x):,.{dp}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(x, dp=0):
    try:
        return f"{float(x):,.{dp}f}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def _table(header, rows, widths, align_right_from=1):
    data = [header] + rows
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, HAIR),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
    ]
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _dual_axis_png(x_labels, bars, line, bar_label, line_label,
                   bar_color=TEAL_HEX, line_color=CLAY_HEX,
                   y1_label="RM", y2_label="", width_px=1600, height_in=3.2):
    fig, ax = plt.subplots(figsize=(width_px / 160, height_in), dpi=160)
    x = range(len(x_labels))

    ax.bar(list(x), bars, width=0.55, color=bar_color, label=bar_label)
    ax.set_ylabel(y1_label, fontsize=8, color=INK_HEX)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="both", labelsize=7, colors=INK_HEX)
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=6.5)
    ax.grid(axis="y", color="#EDEFEF", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEDD")

    ax2 = ax.twinx()
    ax2.plot(list(x), line, color=line_color, linewidth=1.6, marker="o",
             markersize=3, label=line_label)
    ax2.set_ylabel(y2_label, fontsize=8, color=line_color)
    ax2.tick_params(axis="y", labelsize=7, colors=line_color)
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, frameon=False, ncol=2)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _trend_png(trend, key, width_px=1600):
    fig, ax = plt.subplots(figsize=(width_px / 160, 3.3), dpi=160)
    x = range(len(trend))
    labels = [str(v) for v in trend[key]]

    ax.bar([i - 0.2 for i in x], trend["revenue"], width=0.4,
           color=TEAL_HEX, label="Revenue")
    ax.bar([i + 0.2 for i in x], trend["profit"], width=0.4,
           color=SAND_HEX, label="Gross profit")
    ax.set_ylabel("RM", fontsize=8, color=INK_HEX)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="both", labelsize=7, colors=INK_HEX)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
    ax.grid(axis="y", color="#EDEFEF", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEDD")

    ax2 = ax.twinx()
    ax2.plot(list(x), trend["margin"], color=CLAY_HEX, linewidth=1.6,
             marker="o", markersize=3, label="Margin %")
    ax2.set_ylabel("Margin %", fontsize=8, color=CLAY_HEX)
    ax2.tick_params(axis="y", labelsize=7, colors=CLAY_HEX)
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, frameon=False, ncol=3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _pareto_png(rev_df, width_px=1600):
    fig, ax = plt.subplots(figsize=(width_px / 160, 2.6), dpi=160)
    y = rev_df["Cumulative %"].tolist()
    x = list(range(1, len(y) + 1))
    ax.plot(x, y, color=TEAL_HEX, linewidth=1.8)
    ax.fill_between(x, y, color=TEAL_HEX, alpha=0.12)
    ax.axhline(80, color=CLAY_HEX, linewidth=1, linestyle="--")
    ax.text(len(x) * 0.99, 81.5, "80% of revenue", fontsize=7,
            color=CLAY_HEX, ha="right")
    ax.set_xlabel("Products, ranked by revenue", fontsize=8, color=INK_HEX)
    ax.set_ylabel("Cumulative revenue %", fontsize=8, color=INK_HEX)
    ax.tick_params(labelsize=7, colors=INK_HEX)
    ax.grid(axis="y", color="#EDEFEF", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 105)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _bar_png(labels, values, color=TEAL_HEX, ylabel="RM", width_px=1600, height_in=3.0):
    fig, ax = plt.subplots(figsize=(width_px / 160, height_in), dpi=160)
    x = range(len(labels))
    ax.bar(list(x), values, color=color)
    ax.set_ylabel(ylabel, fontsize=8, color=INK_HEX)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="both", labelsize=7, colors=INK_HEX)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30 if max(len(str(l)) for l in labels) < 12 else 45,
                       ha="right", fontsize=6.5)
    ax.grid(axis="y", color="#EDEFEF", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D8DEDD")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------

def _page_decor(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(HAIR)
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, doc.report_title)
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_pdf(
    business_name,
    period_label,
    grain,
    kpi,
    kpi_prev,
    trend,
    trend_key,
    stock_qty=None,
    stock_rev=None,
    stock_margin=None,
    margin_floor=None,
    exclude_no_cost=False,
    no_cost_lines=0,
    total_lines=0,
    filters_note="",
    staff=None,
    customers=None,

    sections=None,
):
    """Return PDF bytes. Any frame passed as None is skipped."""
    sections = sections or {}
    st_ = _styles()
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=20 * mm,
        title=f"Sales report — {period_label}",
        author=business_name,
    )
    doc.report_title = f"{business_name} · {period_label}"

    story = []

    # --- Cover ---------------------------------------------------------
    story.append(Paragraph("Sales report", st_["title"]))
    story.append(Paragraph(
        f"{business_name}<br/>{period_label}<br/>"
        f"Generated {datetime.now():%d %b %Y, %H:%M}",
        st_["sub"]))
    if filters_note and filters_note != "none":
        story.append(Paragraph(f"Filters applied: {filters_note}", st_["note"]))
        story.append(Spacer(1, 6))

    # --- Summary ---------------------------------------------------------
    if sections.get("summary", True):
        story.append(Paragraph("Summary", st_["h2"]))

        def _delta(now, prev, pp=False):
            if prev in (None, 0) or now is None:
                return "—"
            if pp:
                return f"{now - prev:+.1f} pp"
            return f"{(now - prev) / abs(prev) * 100:+.1f}%"

        rows = [
            ["Revenue", f"RM {_fmt_rm(kpi['revenue'])}",
             _delta(kpi["revenue"], (kpi_prev or {}).get("revenue"))],
            ["Gross profit", f"RM {_fmt_rm(kpi['profit'])}",
             _delta(kpi["profit"], (kpi_prev or {}).get("profit"))],
            ["Gross margin", f"{_fmt_num(kpi['margin'], 1)}%",
             _delta(kpi["margin"], (kpi_prev or {}).get("margin"), pp=True)],
            ["Transactions", _fmt_num(kpi["invoices"]),
             _delta(kpi["invoices"], (kpi_prev or {}).get("invoices"))],
            ["Purchase per head", f"RM {_fmt_rm(kpi['basket'])}",
             _delta(kpi["basket"], (kpi_prev or {}).get("basket"))],
            ["Units sold", _fmt_num(kpi["units"]),
             _delta(kpi["units"], (kpi_prev or {}).get("units"))],
            ["Line items", _fmt_num(kpi["lines"]), "—"],
        ]
        story.append(_table(["Metric", "Value", "vs previous period"], rows,
                            [58 * mm, 48 * mm, 42 * mm]))

        if no_cost_lines:
            state = "excluded from" if exclude_no_cost else "included in"
            story.append(Paragraph(
                f"{no_cost_lines:,} of {total_lines:,} lines carry no recorded cost "
                f"and are {state} the margin figures above. Figures are taken from "
                f"the POS export without adjustment.", st_["note"]))

    # --- Trend -------------------------------------------------------------
    if sections.get("trend", True) and trend is not None and len(trend):
        story.append(Paragraph(f"{grain} performance", st_["h2"]))
        story.append(Image(_trend_png(trend, trend_key),
                           width=170 * mm, height=170 * mm * 0.30))
        story.append(Spacer(1, 8))

        rows = []
        for _, r in trend.iterrows():
            rows.append([
                str(r[trend_key]), _fmt_rm(r["revenue"]), _fmt_rm(r["profit"]),
                f"{_fmt_num(r['margin'], 1)}%", _fmt_num(r["invoices"]),
                _fmt_rm(r["basket"]), _fmt_num(r["units"]),
            ])
        story.append(_table(
            ["Period", "Revenue RM", "Profit RM", "Margin", "TC", "PP RM", "Units"],
            rows, [30 * mm, 26 * mm, 24 * mm, 18 * mm, 18 * mm, 24 * mm, 20 * mm]))
        story.append(Paragraph(
            "TC = transaction count. PP = purchase per head, revenue divided by "
            "transactions.", st_["note"]))

    # --- Stock ---------------------------------------------------------
    def _stock_block(sdf, title, widths, note=None, chart=None):
        story.append(PageBreak())
        story.append(Paragraph(title, st_["h2"]))
        if chart is not None:
            story.append(Image(chart, width=170 * mm, height=170 * mm * 0.24))
            story.append(Spacer(1, 8))
        rows = []
        for _, r in sdf.iterrows():
            rows.append([
                str(r["Description"])[:46], _fmt_num(r["Qty"]), _fmt_rm(r["Revenue"]),
                _fmt_rm(r["Profit"]), f"{_fmt_num(r['Margin %'], 1)}%",
            ])
        story.append(_table(["Item", "Qty", "Revenue RM", "Profit RM", "Margin"],
                            rows, [74 * mm, 18 * mm, 28 * mm, 26 * mm, 18 * mm]))
        if note:
            story.append(Paragraph(note, st_["note"]))

    if sections.get("stock_qty", True) and stock_qty is not None and len(stock_qty):
        _stock_block(stock_qty, f"Top {len(stock_qty)} items by quantity", None,
                     "Ranked by units moved. High volume does not imply high profit — "
                     "compare against the revenue and margin tables.")

    if sections.get("stock_rev", True) and stock_rev is not None and len(stock_rev):
        chart = _pareto_png(stock_rev) if "Cumulative %" in stock_rev.columns else None
        _stock_block(stock_rev, f"Top {len(stock_rev)} items by revenue", None, None, chart)

    if sections.get("stock_margin", True) and stock_margin is not None and len(stock_margin):
        note = "Ranked by margin percentage. "
        if margin_floor:
            note += (f"Items below RM {margin_floor:,.0f} revenue are excluded, so that "
                     f"single-unit sales and uncosted lines do not dominate the ranking.")
        _stock_block(stock_margin, f"Top {len(stock_margin)} items by margin", None, note)

    # --- Staff ---------------------------------------------------------
    if sections.get("staff", True) and staff is not None and len(staff):
        story.append(PageBreak())
        story.append(Paragraph("Staff performance", st_["h2"]))
        story.append(Paragraph(
            "Shown exactly as recorded in the POS 'Sales Agent' field. Some entries "
            "are order channels rather than staff members. 'Not recorded' is revenue "
            "with no agent logged at the till, not a person.", st_["note"]))

        not_rec = staff[staff["Sales Agent"] == "Not recorded"]
        if not not_rec.empty and not_rec["Revenue"].iloc[0] > 0:
            share = not_rec["Revenue"].iloc[0] / staff["Revenue"].sum() * 100
            story.append(Paragraph(
                f"RM {not_rec['Revenue'].iloc[0]:,.0f} of revenue ({share:.1f}%) has no "
                f"agent recorded in this period.", st_["warn"]))

        story.append(Image(
            _bar_png(staff["Sales Agent"].tolist(), staff["Profit"].tolist(),
                    color=TEAL_HEX, ylabel="Gross profit (RM)"),
            width=170 * mm, height=170 * mm * 0.26))
        story.append(Spacer(1, 8))

        rows = []
        for _, r in staff.iterrows():
            rows.append([
                str(r["Sales Agent"])[:28], _fmt_rm(r["Revenue"]), _fmt_rm(r["Profit"]),
                f"{_fmt_num(r['Margin %'], 1)}%", _fmt_num(r["Transactions"]),
                _fmt_rm(r["Avg basket"]),
            ])
        story.append(_table(
            ["Agent", "Revenue RM", "Profit RM", "Margin", "Transactions", "Avg basket RM"],
            rows, [40 * mm, 28 * mm, 26 * mm, 18 * mm, 26 * mm, 28 * mm]))

    # --- Customers ---------------------------------------------------------
    if sections.get("customers", True) and customers is not None and len(customers):
        story.append(PageBreak())
        story.append(Paragraph(f"Top {len(customers)} customers by profit", st_["h2"]))
        story.append(Paragraph(
            "CASH transactions are excluded from this ranking — there is no single "
            "customer behind that name.", st_["note"]))

        rows = []
        for _, r in customers.iterrows():
            rows.append([
                str(r["Customer"])[:40], _fmt_rm(r["Revenue"]), _fmt_rm(r["Profit"]),
                f"{_fmt_num(r['Margin %'], 1)}%", _fmt_num(r["Transactions"]),
            ])
        story.append(_table(
            ["Customer", "Revenue RM", "Profit RM", "Margin", "Transactions"],
            rows, [70 * mm, 30 * mm, 28 * mm, 20 * mm, 22 * mm]))

    doc.build(story, onFirstPage=_page_decor, onLaterPages=_page_decor)
    buf.seek(0)
    return buf.getvalue()
