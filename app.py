from __future__ import annotations

import io
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import feedparser

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# =============================================================================
# NEXT TESLA WATCHLIST — Monthly Scan + Thesis Health + Action Guidance + PDF
# Single-file Streamlit app
# =============================================================================

# =============================================================================
# CONFIG: 20-stock bench (Core + Growth + Optionality)
# =============================================================================

WATCHLIST = [
    # ticker, display_name, tags, thesis
    # --- Grid / Electrification / Data centers (core) ---
    ("GEV", "GE Vernova", ["Grid/Electrification", "AI power demand"], "Grid capex + electrification; equipment + services."),
    ("SU.PA", "Schneider Electric", ["Power mgmt", "Data centers"], "Picks-and-shovels power + automation for electrification."),
    ("ETN", "Eaton", ["Power mgmt", "Data centers"], "Electrical equipment + data center power chain."),
    ("ABB", "ABB", ["Electrification", "Automation"], "Industrial electrification + automation exposure."),
    ("VRT", "Vertiv", ["Data centers", "Cooling/Power"], "Data center power + thermal infrastructure tailwind."),

    # --- Storage / Grid flexibility ---
    ("FLNC", "Fluence Energy", ["Energy Storage", "Grid"], "Grid-scale storage systems; adoption + integration."),

    # --- Nuclear / Uranium / Power producers ---
    ("BWXT", "BWX Technologies", ["Nuclear/Defense", "Industrial"], "Defense nuclear + SMR ecosystem exposure."),
    ("CEG", "Constellation Energy", ["Nuclear power", "Grid"], "Large nuclear fleet; power demand tailwind."),
    ("CCJ", "Cameco", ["Uranium", "Supply chain"], "Uranium miner/refiner leverage to nuclear buildout."),
    ("LEU", "Centrus Energy", ["Nuclear fuel", "Optionality"], "Enrichment/fuel cycle exposure; higher variance."),
    ("OKLO", "Oklo", ["SMR", "Optionality"], "Early-stage SMR optionality; execution risk."),

    # --- AI / Software / Semis (platform/growth) ---
    ("PLTR", "Palantir", ["AI/Software", "Gov/Defense"], "Sticky deployments; AI ops platform."),
    ("ASML", "ASML Holding", ["Semiconductors", "Chokepoint"], "Lithography bottleneck; AI compute scaling constraint."),
    ("QCOM", "Qualcomm", ["Semiconductors", "Edge AI", "Compute efficiency"], "Edge compute ecosystem; on-device AI + connectivity leverage."),
    ("AVGO", "Broadcom", ["Semiconductors", "AI infrastructure", "Networking"], "AI infrastructure + networking; hyperscaler backbone."),

    # --- Space / Defense (optional/growth blend) ---
    ("RKLB", "Rocket Lab", ["Space/Defense", "Manufacturing"], "Vertical integration; launch + defense + satellites."),
    ("ASTS", "AST SpaceMobile", ["Space/Telecom", "Binary optionality"], "Direct-to-device satellite cellular; execution-dependent."),
    ("LMT", "Lockheed Martin", ["Defense prime", "Resilience"], "Defense prime; steadier anchor vs smallcaps."),
    ("LHX", "L3Harris", ["Defense", "Sensors/Comms"], "Defense electronics + space/ISR exposure."),

    # --- Batteries (high-risk optionality) ---
    ("QS", "QuantumScape", ["Batteries", "Binary optionality"], "Solid-state battery breakthrough optionality."),
]

DEFAULT_TICKERS = [t for t, _, _, _ in WATCHLIST]

# =============================================================================
# Utility helpers
# =============================================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def month_key(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y_%m")

def safe_num(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)) and (not math.isnan(x)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None

def pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x*100:.1f}%"

def fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    absx = abs(x)
    if absx >= 1e12:
        return f"{x/1e12:.2f}T"
    if absx >= 1e9:
        return f"{x/1e9:.2f}B"
    if absx >= 1e6:
        return f"{x/1e6:.2f}M"
    if absx >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:.0f}"

def max_drawdown(close: pd.Series) -> Optional[float]:
    if close is None or len(close) < 2:
        return None
    roll_max = close.cummax()
    dd = (close / roll_max) - 1.0
    return float(dd.min())

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def is_optionality(tags: List[str]) -> bool:
    joined = " | ".join(tags).lower()
    if "binary optionality" in joined:
        return True
    if "space" in joined:
        return True
    if "optional" in joined:
        return True
    if "smr" in joined:
        return True
    return False

def is_us_ticker(ticker: str) -> bool:
    return "." not in ticker  # crude heuristic

# =============================================================================
# Persistence: snapshots + notes + portfolio state
# =============================================================================

def snapshot_path_for_month(mkey: str) -> str:
    return os.path.join(DATA_DIR, f"watchlist_snapshot_{mkey}.json")

def load_snapshot_by_month(mkey: str) -> Dict[str, Any]:
    try:
        with open(snapshot_path_for_month(mkey), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_latest_snapshot_before(mkey: str) -> Tuple[str, Dict[str, Any]]:
    try:
        files = [fn for fn in os.listdir(DATA_DIR) if fn.startswith("watchlist_snapshot_") and fn.endswith(".json")]
        keys = []
        for fn in files:
            mm = fn.replace("watchlist_snapshot_", "").replace(".json", "")
            if re.fullmatch(r"\d{4}_\d{2}", mm):
                keys.append(mm)
        keys = sorted(set(keys))
        prev_keys = [k for k in keys if k < mkey]
        if not prev_keys:
            return "", {}
        prev = prev_keys[-1]
        return prev, load_snapshot_by_month(prev)
    except Exception:
        return "", {}

def save_snapshot(snapshot: Dict[str, Any], mkey: str) -> None:
    try:
        with open(snapshot_path_for_month(mkey), "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

THESIS_NOTES_PATH = os.path.join(DATA_DIR, "thesis_notes.json")

def load_thesis_notes() -> Dict[str, Any]:
    try:
        with open(THESIS_NOTES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_thesis_notes(notes: Dict[str, Any]) -> None:
    try:
        with open(THESIS_NOTES_PATH, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

PORTFOLIO_STATE_PATH = os.path.join(DATA_DIR, "portfolio_state.json")

def load_portfolio_state() -> Dict[str, Any]:
    """
    Stores which tickers are 'Active' (allocatable) vs 'Inactive' (still tracked).
    """
    try:
        with open(PORTFOLIO_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
            if isinstance(state, dict) and "active" in state and isinstance(state["active"], dict):
                return state
    except Exception:
        pass
    return {"active": {t: True for t in DEFAULT_TICKERS}, "updated_at": iso(utc_now())}

def save_portfolio_state(state: Dict[str, Any]) -> None:
    try:
        state["updated_at"] = iso(utc_now())
        with open(PORTFOLIO_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# =============================================================================
# Data sources: Market & financials (yfinance)
# =============================================================================

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def yf_info(ticker: str) -> Dict[str, Any]:
    t = yf.Ticker(ticker)
    try:
        return t.get_info() or {}
    except Exception:
        return {}

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def yf_history(ticker: str, period: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        h = t.history(period=period, auto_adjust=True)
        if h is None or h.empty:
            return pd.DataFrame()
        return h.reset_index()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def yf_quarterly_financials(ticker: str) -> Dict[str, pd.DataFrame]:
    out = {}
    t = yf.Ticker(ticker)
    try:
        qf = t.quarterly_financials
        if isinstance(qf, pd.DataFrame) and not qf.empty:
            out["quarterly_financials"] = qf
    except Exception:
        pass
    try:
        qbs = t.quarterly_balance_sheet
        if isinstance(qbs, pd.DataFrame) and not qbs.empty:
            out["quarterly_balance_sheet"] = qbs
    except Exception:
        pass
    try:
        qcf = t.quarterly_cashflow
        if isinstance(qcf, pd.DataFrame) and not qcf.empty:
            out["quarterly_cashflow"] = qcf
    except Exception:
        pass
    return out

def extract_metric(df: pd.DataFrame, row_name_candidates: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    idx = [str(i).lower() for i in df.index]
    for cand in row_name_candidates:
        c = cand.lower()
        if c in idx:
            return df.loc[df.index[idx.index(c)]]
        for i, name in enumerate(idx):
            if c in name:
                return df.loc[df.index[i]]
    return None

def compute_financial_features(ticker: str) -> Dict[str, Any]:
    feats: Dict[str, Any] = {
        "rev_latest": None,
        "rev_prev": None,
        "rev_yoy": None,
        "rev_qoq": None,
        "op_income_latest": None,
        "gross_profit_latest": None,
        "cash_latest": None,
        "debt_latest": None,
        "ocf_latest": None,
    }

    pack = yf_quarterly_financials(ticker)
    qf = pack.get("quarterly_financials")
    qbs = pack.get("quarterly_balance_sheet")
    qcf = pack.get("quarterly_cashflow")

    if isinstance(qf, pd.DataFrame) and not qf.empty:
        rev = extract_metric(qf, ["Total Revenue", "Revenue"])
        op_inc = extract_metric(qf, ["Operating Income", "Operating Income or Loss"])
        gp = extract_metric(qf, ["Gross Profit"])

        if rev is not None and len(rev) >= 2:
            s = rev.copy()
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            latest = safe_num(s.iloc[-1])
            prev = safe_num(s.iloc[-2]) if len(s) >= 2 else None
            feats["rev_latest"] = latest
            feats["rev_prev"] = prev
            feats["rev_qoq"] = ((latest / prev) - 1.0) if (latest is not None and prev not in (None, 0)) else None
            if len(s) >= 5:
                yoy_base = safe_num(s.iloc[-5])
                feats["rev_yoy"] = ((latest / yoy_base) - 1.0) if (latest is not None and yoy_base not in (None, 0)) else None

        if op_inc is not None:
            s = op_inc.copy()
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            feats["op_income_latest"] = safe_num(s.iloc[-1])

        if gp is not None:
            s = gp.copy()
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            feats["gross_profit_latest"] = safe_num(s.iloc[-1])

    if isinstance(qbs, pd.DataFrame) and not qbs.empty:
        cash = extract_metric(qbs, ["Cash And Cash Equivalents", "Cash"])
        debt = extract_metric(qbs, ["Total Debt", "Long Term Debt", "Long Term Debt And Capital Lease Obligation"])
        if cash is not None:
            s = cash.copy()
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            feats["cash_latest"] = safe_num(s.iloc[-1])
        if debt is not None:
            s = debt.copy()
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            feats["debt_latest"] = safe_num(s.iloc[-1])

    if isinstance(qcf, pd.DataFrame) and not qcf.empty:
        ocf = extract_metric(qcf, ["Total Cash From Operating Activities", "Operating Cash Flow"])
        if ocf is not None:
            s = ocf.copy()
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            feats["ocf_latest"] = safe_num(s.iloc[-1])

    return feats

# =============================================================================
# SEC EDGAR filings (US tickers only)
# =============================================================================

SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
IMPORTANT_FORMS = {"10-Q", "10-K", "8-K", "S-1", "20-F", "6-K", "424B", "F-1", "F-3", "DEF 14A"}
INSIDER_FORMS = {"4", "4/A", "5"}

def sec_headers(user_agent: str) -> Dict[str, str]:
    return {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def sec_ticker_cik_map(user_agent: str) -> Dict[str, str]:
    r = requests.get(SEC_TICKER_CIK_URL, headers=sec_headers(user_agent), timeout=30)
    r.raise_for_status()
    data = r.json()
    out: Dict[str, str] = {}
    for _, row in data.items():
        t = (row.get("ticker") or "").upper()
        cik = str(row.get("cik_str") or "").zfill(10)
        if t and cik:
            out[t] = cik
    return out

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def sec_recent_filings(cik10: str, user_agent: str, max_items: int = 25) -> List[Dict[str, Any]]:
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    r = requests.get(url, headers=sec_headers(user_agent), timeout=30)
    if r.status_code != 200:
        return []
    sub = r.json()
    recent = (sub.get("filings") or {}).get("recent") or {}

    forms = recent.get("form", [])
    accession = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    out: List[Dict[str, Any]] = []
    n = min(len(forms), max_items)
    for i in range(n):
        form = forms[i]
        acc_no_dash = (accession[i] or "").replace("-", "")
        fdate = dates[i] if i < len(dates) else ""
        pdoc = primary_docs[i] if i < len(primary_docs) else None
        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc_no_dash}/{pdoc}"
            if pdoc else f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{acc_no_dash}/"
        )
        out.append({"form": form, "filing_date": fdate, "url": doc_url})
    return out

# =============================================================================
# News: Google News RSS (+ optional extra RSS)
# =============================================================================

def google_news_rss_query(query: str) -> str:
    q = requests.utils.quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl=en-GB&gl=GB&ceid=GB:en"

@st.cache_data(ttl=3 * 60 * 60, show_spinner=False)
def fetch_rss(feed_url: str, lookback_days: int = 35) -> List[Dict[str, Any]]:
    d = feedparser.parse(feed_url)
    cutoff = utc_now() - timedelta(days=lookback_days)
    items: List[Dict[str, Any]] = []

    for e in d.entries:
        title = getattr(e, "title", "") or ""
        link = getattr(e, "link", "") or ""
        summary = getattr(e, "summary", "") or ""
        published = getattr(e, "published", None)

        pub_dt = None
        if published and getattr(e, "published_parsed", None):
            try:
                pub_dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            except Exception:
                pub_dt = None

        if pub_dt and pub_dt < cutoff:
            continue
        if title and link:
            items.append({
                "title": re.sub(r"\s+", " ", title).strip(),
                "link": link,
                "published": pub_dt.isoformat() if pub_dt else "",
                "summary": re.sub(r"\s+", " ", summary).strip()[:320],
                "source": feed_url,
            })
    return items

def summarize_news_titles(items: List[Dict[str, Any]], max_titles: int = 8) -> List[str]:
    titles = []
    seen = set()
    for it in items:
        t = it.get("title", "")
        if t and t not in seen:
            titles.append(t)
            seen.add(t)
        if len(titles) >= max_titles:
            break
    return titles

def sentiment_proxy(items: List[Dict[str, Any]]) -> Dict[str, int]:
    pos_words = ["beats", "record", "award", "contract", "win", "approval", "launch", "growth", "upgrade", "expands", "profits", "selected", "order"]
    neg_words = ["miss", "delay", "cuts", "probe", "lawsuit", "resign", "recall", "loss", "downgrade", "halt", "cancel", "bankrupt", "restatement"]
    pos = 0
    neg = 0
    for it in items:
        txt = (it.get("title", "") + " " + it.get("summary", "")).lower()
        pos += sum(1 for w in pos_words if w in txt)
        neg += sum(1 for w in neg_words if w in txt)
    return {"pos_hits": pos, "neg_hits": neg}

# =============================================================================
# Scoring model (0-100)
# =============================================================================

def score_company(
    dd_12m: Optional[float],
    rev_yoy: Optional[float],
    rev_qoq: Optional[float],
    op_income: Optional[float],
    gross_profit: Optional[float],
    cash: Optional[float],
    debt: Optional[float],
    ocf: Optional[float],
    important_filings: int,
    insider_filings: int,
    news_pos: int,
    news_neg: int,
) -> Tuple[int, Dict[str, int], List[str]]:
    reasons: List[str] = []
    comp: Dict[str, int] = {}

    # 1) Fundamentals momentum (0..35)
    fund = 0
    if rev_yoy is not None:
        if rev_yoy >= 0.30:
            fund += 14; reasons.append("Strong revenue YoY acceleration.")
        elif rev_yoy >= 0.15:
            fund += 10
        elif rev_yoy >= 0.05:
            fund += 6
        elif rev_yoy >= 0.0:
            fund += 3
        else:
            fund += 0; reasons.append("Revenue YoY contraction/weakness.")
    else:
        fund += 5
        reasons.append("Revenue YoY unavailable (coverage limitation).")

    if rev_qoq is not None:
        if rev_qoq >= 0.10:
            fund += 7
        elif rev_qoq >= 0.03:
            fund += 5
        elif rev_qoq >= 0.0:
            fund += 2
        else:
            fund += 0
    else:
        fund += 2

    if op_income is not None:
        fund += 7 if op_income > 0 else 2
    else:
        fund += 2

    fund = int(clamp(fund, 0, 35))
    comp["Fundamentals"] = fund

    # 2) Resilience (0..20)
    resil = 0
    if cash is not None and debt is not None:
        resil += 10 if cash > debt else 6
    elif cash is not None:
        resil += 7
    else:
        resil += 5

    if ocf is not None:
        if ocf > 0:
            resil += 8
        else:
            resil += 3; reasons.append("Operating cash flow negative/weak (watch runway).")
    else:
        resil += 5

    resil = int(clamp(resil, 0, 20))
    comp["Resilience"] = resil

    # 3) Filings (0..15)
    reg = 0
    reg += min(important_filings, 3) * 3
    reg += min(insider_filings, 3) * 2
    if important_filings >= 2:
        reasons.append("Multiple major SEC filings recently (potential catalysts/disclosures).")
    if insider_filings >= 1:
        reasons.append("Recent insider filing activity (Form 4/5) — review context.")
    reg = int(clamp(reg, 0, 15))
    comp["Filings/Reg"] = reg

    # 4) News (0..15)
    news = 8
    news += min(news_pos, 6)
    news -= min(news_neg, 8)
    if news_neg >= 2:
        reasons.append("Multiple negative-news keyword hits — validate specifics.")
    news = int(clamp(news, 0, 15))
    comp["News/Catalysts"] = news

    # 5) Market stress (0..15)
    mkt = 10
    if dd_12m is not None:
        if dd_12m <= -0.50:
            mkt += 3; reasons.append("Deep drawdown: opportunity or structural trouble — prioritize review.")
        elif dd_12m <= -0.30:
            mkt += 2
        elif dd_12m <= -0.15:
            mkt += 1
    mkt = int(clamp(mkt, 0, 15))
    comp["Market Stress"] = mkt

    total = int(clamp(fund + resil + reg + news + mkt, 0, 100))
    return total, comp, reasons

# =============================================================================
# Thesis Health + Action Engine
# =============================================================================

@dataclass
class ThesisSignal:
    name: str
    level: str  # "green" | "amber" | "red"
    reason: str

def thesis_health_signals(row: Dict[str, Any]) -> Tuple[List[ThesisSignal], Dict[str, Any]]:
    signals: List[ThesisSignal] = []

    rev_yoy = row.get("rev_yoy")
    rev_qoq = row.get("rev_qoq")
    ocf = row.get("ocf_latest")
    cash = row.get("cash_latest")
    debt = row.get("debt_latest")
    dd12 = row.get("dd_12m")
    news_neg = (row.get("news_proxy") or {}).get("neg_hits", 0)
    important_filings = row.get("important_filings", 0)

    if rev_yoy is None:
        signals.append(ThesisSignal("Growth", "amber", "Revenue YoY unavailable; verify in filings/deck."))
    elif rev_yoy < 0:
        signals.append(ThesisSignal("Growth", "red", f"Revenue YoY negative ({pct(rev_yoy)})."))
    elif rev_yoy < 0.05:
        signals.append(ThesisSignal("Growth", "amber", f"Revenue YoY low ({pct(rev_yoy)})."))
    else:
        signals.append(ThesisSignal("Growth", "green", f"Revenue YoY healthy ({pct(rev_yoy)})."))

    if rev_qoq is None:
        signals.append(ThesisSignal("Momentum", "amber", "Revenue QoQ unavailable/unclear."))
    elif rev_qoq < -0.05:
        signals.append(ThesisSignal("Momentum", "red", f"Revenue QoQ drop ({pct(rev_qoq)})."))
    elif rev_qoq < 0.02:
        signals.append(ThesisSignal("Momentum", "amber", f"Revenue QoQ flat ({pct(rev_qoq)})."))
    else:
        signals.append(ThesisSignal("Momentum", "green", f"Revenue QoQ positive ({pct(rev_qoq)})."))

    if cash is None:
        signals.append(ThesisSignal("Survivability", "amber", "Cash unavailable; confirm balance sheet."))
    else:
        if debt is not None and cash < debt and (ocf is not None and ocf < 0):
            signals.append(ThesisSignal("Survivability", "red", "Cash < Debt and OCF negative (financing risk elevated)."))
        elif ocf is not None and ocf < 0:
            signals.append(ThesisSignal("Survivability", "amber", "OCF negative; watch burn/runway."))
        else:
            signals.append(ThesisSignal("Survivability", "green", "No immediate survivability red flags detected."))

    if isinstance(dd12, float) and dd12 <= -0.60:
        signals.append(ThesisSignal("Market", "amber", f"Very large drawdown ({pct(dd12)}). Prioritize review."))
    elif isinstance(dd12, float) and dd12 <= -0.40:
        signals.append(ThesisSignal("Market", "amber", f"Large drawdown ({pct(dd12)})."))
    else:
        signals.append(ThesisSignal("Market", "green", f"Drawdown within normal range ({pct(dd12)})."))

    if important_filings >= 3:
        signals.append(ThesisSignal("Disclosures", "amber", "Many recent major SEC filings. Skim for risks/catalysts."))
    else:
        signals.append(ThesisSignal("Disclosures", "green", "No unusual major filing intensity detected."))

    if news_neg >= 3:
        signals.append(ThesisSignal("Narrative", "amber", "Higher negative keyword pressure in recent news. Validate specifics."))
    else:
        signals.append(ThesisSignal("Narrative", "green", "No strong negative keyword pressure detected."))

    reds = sum(1 for s in signals if s.level == "red")
    ambers = sum(1 for s in signals if s.level == "amber")
    health = {
        "reds": reds,
        "ambers": ambers,
        "health_label": "Healthy" if reds == 0 and ambers <= 2 else ("Watchlist" if reds == 0 else "At Risk"),
    }
    return signals, health

def action_recommendation(row: Dict[str, Any], prev: Optional[Dict[str, Any]] = None) -> Tuple[str, str, List[str]]:
    score = int(row.get("score") or 0)
    signals, health = thesis_health_signals(row)
    delta = None
    if prev and isinstance(prev.get("score"), int):
        delta = score - int(prev["score"])

    reds = health["reds"]
    ambers = health["ambers"]

    rev_yoy = row.get("rev_yoy")
    ocf = row.get("ocf_latest")
    cash = row.get("cash_latest")
    debt = row.get("debt_latest")

    exit_conditions = []
    if (rev_yoy is not None and rev_yoy < 0) and (ocf is not None and ocf < 0) and (cash is not None and debt is not None and cash < debt):
        exit_conditions.append("Revenue shrinking + OCF negative + cash < debt (financing/thesis risk).")
    if score <= 45 and reds >= 2:
        exit_conditions.append("Very low score and multiple red thesis flags.")

    reasons: List[str] = []
    if delta is not None:
        reasons.append(f"Score vs last snapshot: {prev['score']} → {score} ({delta:+d}).")

    if exit_conditions:
        action = "EXIT"
        conf = "High (rules-triggered)"
        reasons.extend(exit_conditions)
    else:
        opt = is_optionality(row.get("tags") or [])
        if reds >= 1:
            if score >= 75 and not opt:
                action, conf = "HOLD", "Medium"
                reasons.append("Red flag present, but overall score strong — hold and validate the red flag.")
            else:
                action, conf = "PAUSE", "Medium"
                reasons.append("At least one red thesis signal — pause new capital until validated.")
        else:
            if score >= 78 and ambers <= 2 and not opt:
                action, conf = "ADD", "Medium"
                reasons.append("High score + no red flags — candidate for continued accumulation (rules-based).")
            elif score >= 70 and ambers <= 3:
                action, conf = ("HOLD", "Low–Medium") if opt else ("ADD", "Medium")
                reasons.append("Good score with manageable ambers — steady adds (or hold if optionality).")
            elif score >= 60:
                action, conf = "HOLD", "Medium"
                reasons.append("Mixed signals — hold and monitor for improving fundamentals or clearer catalysts.")
            else:
                action, conf = "PAUSE", "Medium"
                reasons.append("Lower score — pause new capital and re-check thesis / survivability.")

    top_sig = [s for s in signals if s.level in ("red", "amber")][:4]
    for s in top_sig:
        reasons.append(f"{s.name}: {s.reason}")

    return action, conf, reasons[:8]

# =============================================================================
# Month-over-month change tracking
# =============================================================================

def diff_summary(current: Dict[str, Any], previous: Dict[str, Any]) -> Dict[str, Any]:
    changes = {
        "score_moves": [],
        "action_changes": [],
        "new_major_filings": [],
        "news_spikes": [],
        "drawdown_shocks": [],
        "risk_flag_spikes": [],
    }

    for t, cur in (current.get("companies") or {}).items():
        prev = (previous.get("companies") or {}).get(t, {})

        cur_score = cur.get("score")
        prev_score = prev.get("score")
        if isinstance(cur_score, int) and isinstance(prev_score, int):
            delta = cur_score - prev_score
            if abs(delta) >= 8:
                changes["score_moves"].append((t, delta, prev_score, cur_score))

        cur_action = cur.get("action")
        prev_action = prev.get("action")
        if isinstance(cur_action, str) and isinstance(prev_action, str) and cur_action != prev_action:
            changes["action_changes"].append((t, prev_action, cur_action))

        cur_major = cur.get("important_filings", 0)
        prev_major = prev.get("important_filings", 0)
        if isinstance(cur_major, int) and isinstance(prev_major, int) and (cur_major - prev_major) >= 2:
            changes["new_major_filings"].append((t, prev_major, cur_major))

        cur_news = cur.get("news_count", 0)
        prev_news = prev.get("news_count", 0)
        if isinstance(cur_news, int) and isinstance(prev_news, int) and (cur_news >= 8 and cur_news > prev_news * 1.5 + 2):
            changes["news_spikes"].append((t, prev_news, cur_news))

        cur_dd = cur.get("dd_12m")
        prev_dd = prev.get("dd_12m")
        if isinstance(cur_dd, float) and isinstance(prev_dd, float):
            if (cur_dd - prev_dd) <= -0.15:
                changes["drawdown_shocks"].append((t, prev_dd, cur_dd))

        cur_reds = cur.get("thesis_reds", 0)
        prev_reds = prev.get("thesis_reds", 0)
        if isinstance(cur_reds, int) and isinstance(prev_reds, int) and (cur_reds - prev_reds) >= 1 and cur_reds >= 1:
            changes["risk_flag_spikes"].append((t, prev_reds, cur_reds))

    changes["score_moves"].sort(key=lambda x: abs(x[1]), reverse=True)
    changes["drawdown_shocks"].sort(key=lambda x: x[2])
    return changes

# =============================================================================
# Allocation Engine
# =============================================================================

def normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in raw.values())
    if total <= 0:
        n = len(raw) if raw else 1
        return {k: 1.0 / n for k in raw}
    return {k: max(0.0, v) / total for k, v in raw.items()}

def allocate_pounds(
    rows: List[Dict[str, Any]],
    monthly_budget: float,
    strategy: str,
    max_stock_weight: float,
    min_stock_weight: float,
    max_optionality_bucket: float,
    optionality_cap_each: float,
    include_only_actions: Optional[List[str]] = None,
    active_only: bool = True,
    portfolio_active_map: Optional[Dict[str, bool]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    filt_rows = rows

    if active_only and portfolio_active_map:
        filt_rows = [r for r in filt_rows if bool(portfolio_active_map.get(r["ticker"], True))]

    if include_only_actions:
        include_only_actions = [a.upper() for a in include_only_actions]
        filt_rows = [r for r in filt_rows if str(r.get("action", "")).upper() in include_only_actions]

    if not filt_rows:
        return pd.DataFrame(), {"error": "No tickers match allocation filters (active/actions)."}

    base_raw: Dict[str, float] = {}
    for r in filt_rows:
        s = float(r.get("score") or 0)
        dd = r.get("dd_12m")
        dd_mag = abs(float(dd)) if isinstance(dd, float) else 0.25

        if strategy == "Score-weighted":
            w = max(0.0, s)
        elif strategy == "Score / Drawdown (risk-adjusted proxy)":
            w = max(0.0, s) / max(0.15, dd_mag)
        else:
            w = max(0.0, s)

        base_raw[r["ticker"]] = float(w)

    weights = normalize_weights(base_raw)
    opt_set = set([r["ticker"] for r in filt_rows if is_optionality(r.get("tags") or [])])

    def cap_and_redistribute(w: Dict[str, float]) -> Dict[str, float]:
        w = dict(w)

        for t in list(w.keys()):
            if t in opt_set:
                w[t] = min(w[t], optionality_cap_each)

        for t in list(w.keys()):
            w[t] = min(w[t], max_stock_weight)

        for t in list(w.keys()):
            w[t] = max(w[t], min_stock_weight)

        opt_total = sum(w[t] for t in w.keys() if t in opt_set)
        if opt_total > max_optionality_bucket and opt_total > 0:
            scale = max_optionality_bucket / opt_total
            for t in list(w.keys()):
                if t in opt_set:
                    w[t] *= scale

        return normalize_weights(w)

    weights = cap_and_redistribute(weights)

    alloc_rows = []
    for r in filt_rows:
        t = r["ticker"]
        w = float(weights.get(t, 0.0))
        alloc_rows.append({
            "Ticker": t,
            "Name": r.get("name", ""),
            "Score": int(r.get("score") or 0),
            "Action": str(r.get("action") or ""),
            "Optionality": bool(t in opt_set),
            "TargetWeight": w,
            "TargetGBP": float(monthly_budget) * w,
            "Tags": ", ".join(r.get("tags") or []),
        })

    df = pd.DataFrame(alloc_rows).sort_values(["TargetWeight", "Score"], ascending=False)

    diagnostics = {
        "total_weight": float(df["TargetWeight"].sum()) if not df.empty else 0.0,
        "optionality_bucket_weight": float(df.loc[df["Optionality"], "TargetWeight"].sum()) if not df.empty else 0.0,
        "max_stock_weight": float(df["TargetWeight"].max()) if not df.empty else 0.0,
        "min_stock_weight": float(df["TargetWeight"].min()) if not df.empty else 0.0,
    }
    return df, diagnostics

# =============================================================================
# PDF generation (ReportLab) — FULL multi-page report
# =============================================================================

def draw_wrapped_text(c: canvas.Canvas, text: str, x: float, y: float, max_width: float, leading: float, font="Helvetica", size=10) -> float:
    c.setFont(font, size)
    words = (text or "").split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, font, size) <= max_width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= leading
            line = w
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y

def build_pdf_report(
    generated_at: datetime,
    universe_rows: List[Dict[str, Any]],
    changes: Dict[str, Any],
    prev_exists: bool,
    prev_key: str,
    current_key: str,
    alloc_df: Optional[pd.DataFrame] = None,
    alloc_meta: Optional[Dict[str, Any]] = None,
) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    left = 2.0 * cm
    right = 2.0 * cm
    top = height - 2.0 * cm
    maxw = width - left - right

    def new_page(title: Optional[str] = None) -> float:
        c.showPage()
        y = top
        if title:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(left, y, title)
            y -= 18
        return y

    # ---------------- Page 1: Executive Summary ----------------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, top, "Monthly 'Next Tesla' Watchlist Report")
    c.setFont("Helvetica", 10)
    c.drawString(left, top - 18, f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}   Month: {current_key}")
    if prev_exists:
        c.drawString(left, top - 32, f"Compared to previous snapshot: {prev_key}")

    y = top - 55
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Executive summary — what changed since last month")
    y -= 18

    c.setFont("Helvetica", 10)
    if not prev_exists:
        y = draw_wrapped_text(
            c,
            "No prior snapshot found. This month establishes the baseline for month-over-month change detection. "
            "Next month, this section will highlight score moves, action changes, filing spikes, news spikes, drawdown shocks, and risk-flag changes.",
            left, y, maxw, 12
        )
    else:
        # Action changes
        if changes.get("action_changes"):
            c.setFont("Helvetica-Bold", 10)
            c.drawString(left, y, "Action changes (ADD/HOLD/PAUSE/EXIT):")
            y -= 14
            c.setFont("Helvetica", 10)
            for t, prev_a, cur_a in changes["action_changes"][:12]:
                if y < 3.2 * cm:
                    y = new_page("Executive summary (continued)")
                    c.setFont("Helvetica", 10)
                c.drawString(left, y, f"- {t}: {prev_a} → {cur_a}")
                y -= 12
        else:
            y = draw_wrapped_text(c, "No action changes detected this month.", left, y, maxw, 12)
        y -= 6

        # Score moves
        if changes.get("score_moves"):
            c.setFont("Helvetica-Bold", 10)
            c.drawString(left, y, "Largest score moves (>= 8 points):")
            y -= 14
            c.setFont("Helvetica", 10)
            for t, delta, prev_s, cur_s in changes["score_moves"][:10]:
                if y < 3.2 * cm:
                    y = new_page("Executive summary (continued)")
                    c.setFont("Helvetica", 10)
                c.drawString(left, y, f"- {t}: {prev_s} → {cur_s} ({delta:+d})")
                y -= 12
        else:
            y = draw_wrapped_text(c, "No large score moves detected this month.", left, y, maxw, 12)
        y -= 6

        # Risk spikes
        if changes.get("risk_flag_spikes"):
            c.setFont("Helvetica-Bold", 10)
            c.drawString(left, y, "Thesis risk spikes (red flags increased):")
            y -= 14
            c.setFont("Helvetica", 10)
            for t, prev_r, cur_r in changes["risk_flag_spikes"][:12]:
                if y < 3.2 * cm:
                    y = new_page("Executive summary (continued)")
                    c.setFont("Helvetica", 10)
                c.drawString(left, y, f"- {t}: red flags {prev_r} → {cur_r}")
                y -= 12
        else:
            y = draw_wrapped_text(c, "No thesis risk spikes detected.", left, y, maxw, 12)
        y -= 6

        # Filing spikes
        if changes.get("new_major_filings"):
            c.setFont("Helvetica-Bold", 10)
            c.drawString(left, y, "Major SEC filing activity increases:")
            y -= 14
            c.setFont("Helvetica", 10)
            for t, prev_m, cur_m in changes["new_major_filings"][:10]:
                if y < 3.2 * cm:
                    y = new_page("Executive summary (continued)")
                    c.setFont("Helvetica", 10)
                c.drawString(left, y, f"- {t}: major filings {prev_m} → {cur_m}")
                y -= 12
        else:
            y = draw_wrapped_text(c, "No major filing activity spikes detected.", left, y, maxw, 12)
        y -= 6

        # News spikes
        if changes.get("news_spikes"):
            c.setFont("Helvetica-Bold", 10)
            c.drawString(left, y, "News spikes (RSS volume jump):")
            y -= 14
            c.setFont("Helvetica", 10)
            for t, prev_n, cur_n in changes["news_spikes"][:10]:
                if y < 3.2 * cm:
                    y = new_page("Executive summary (continued)")
                    c.setFont("Helvetica", 10)
                c.drawString(left, y, f"- {t}: news hits {prev_n} → {cur_n}")
                y -= 12
        else:
            y = draw_wrapped_text(c, "No news spikes detected.", left, y, maxw, 12)
        y -= 6

        # Drawdown shocks
        if changes.get("drawdown_shocks"):
            c.setFont("Helvetica-Bold", 10)
            c.drawString(left, y, "Drawdown shocks (12m max drawdown worsened materially):")
            y -= 14
            c.setFont("Helvetica", 10)
            for t, prev_dd, cur_dd in changes["drawdown_shocks"][:10]:
                if y < 3.2 * cm:
                    y = new_page("Executive summary (continued)")
                    c.setFont("Helvetica", 10)
                c.drawString(left, y, f"- {t}: {pct(prev_dd)} → {pct(cur_dd)}")
                y -= 12
        else:
            y = draw_wrapped_text(c, "No drawdown shocks detected.", left, y, maxw, 12)

    y -= 8
    c.setFont("Helvetica-Oblique", 9)
    y = draw_wrapped_text(
        c,
        "Note: This report is a triage/validation tool based on public signals (market, quarterly financials where available, SEC filings for US tickers, and RSS news). "
        "It is not investment advice. Use it to prioritize reading and thesis validation.",
        left, y, maxw, 11, font="Helvetica-Oblique", size=9
    )

    # Allocation plan table (on page 1 if it fits; else new page)
    if alloc_df is not None and not alloc_df.empty:
        if y < 8.0 * cm:
            y = new_page("Rules-based allocation plan (planning aid)")
        else:
            y -= 6
            c.setFont("Helvetica-Bold", 12)
            c.drawString(left, y, "Rules-based allocation plan (planning aid)")
            y -= 14

        meta = alloc_meta or {}
        c.setFont("Helvetica", 10)
        y = draw_wrapped_text(
            c,
            f"Optionality bucket weight: {meta.get('optionality_bucket_weight', 0.0)*100:.1f}% | "
            f"Max single-stock weight: {meta.get('max_stock_weight', 0.0)*100:.1f}% | "
            f"Min single-stock weight: {meta.get('min_stock_weight', 0.0)*100:.1f}%",
            left, y, maxw, 12
        )
        y -= 4

        c.setFont("Helvetica-Bold", 9)
        c.drawString(left, y, "Ticker")
        c.drawString(left + 55, y, "Action")
        c.drawString(left + 110, y, "Target %")
        c.drawString(left + 170, y, "Target £")
        c.drawString(left + 235, y, "Optionality")
        y -= 10
        c.setLineWidth(0.5)
        c.line(left, y, left + maxw, y)
        y -= 12

        c.setFont("Helvetica", 9)
        for _, row in alloc_df.iterrows():
            if y < 3.2 * cm:
                y = new_page("Rules-based allocation plan (continued)")
                c.setFont("Helvetica-Bold", 9)
                c.drawString(left, y, "Ticker")
                c.drawString(left + 55, y, "Action")
                c.drawString(left + 110, y, "Target %")
                c.drawString(left + 170, y, "Target £")
                c.drawString(left + 235, y, "Optionality")
                y -= 10
                c.setLineWidth(0.5)
                c.line(left, y, left + maxw, y)
                y -= 12
                c.setFont("Helvetica", 9)

            c.drawString(left, y, str(row["Ticker"]))
            c.drawString(left + 55, y, str(row.get("Action", ""))[:8])
            c.drawString(left + 110, y, f"{float(row['TargetWeight'])*100:.1f}%")
            c.drawString(left + 170, y, f"£{float(row['TargetGBP']):.2f}")
            c.drawString(left + 235, y, "Yes" if bool(row["Optionality"]) else "No")
            y -= 12

    # ---------------- Universe Overview ----------------
    y = new_page("Universe overview (top by score)")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(left, y, "Ticker")
    c.drawString(left + 55, y, "Score")
    c.drawString(left + 95, y, "Action")
    c.drawString(left + 145, y, "12M DD")
    c.drawString(left + 205, y, "Rev YoY")
    c.drawString(left + 265, y, "Reds")
    c.drawString(left + 300, y, "Filings")
    c.drawString(left + 350, y, "News")
    c.drawString(left + 395, y, "Tags")
    y -= 10
    c.setLineWidth(0.5)
    c.line(left, y, left + maxw, y)
    y -= 12
    c.setFont("Helvetica", 9)

    top_rows = sorted(universe_rows, key=lambda r: r.get("score", 0), reverse=True)[:18]
    for r in top_rows:
        if y < 3.2 * cm:
            y = new_page("Universe overview (continued)")
            c.setFont("Helvetica", 9)
        c.drawString(left, y, r["ticker"])
        c.drawString(left + 55, y, str(r.get("score", "")))
        c.drawString(left + 95, y, str(r.get("action", "")))
        c.drawString(left + 145, y, pct(r.get("dd_12m")))
        c.drawString(left + 205, y, pct(r.get("rev_yoy")))
        c.drawString(left + 265, y, str(r.get("thesis_reds", 0)))
        c.drawString(left + 300, y, f"{r.get('important_filings', 0)}")
        c.drawString(left + 350, y, f"{r.get('news_count', 0)}")
        c.drawString(left + 395, y, (", ".join(r["tags"]))[:45])
        y -= 12

    # ---------------- Per-company pages ----------------
    for r in sorted(universe_rows, key=lambda x: x.get("score", 0), reverse=True):
        y = new_page(f"{r['ticker']} — {r['name']}")
        c.setFont("Helvetica", 10)
        c.drawString(left, y, f"Tags: {', '.join(r['tags'])}")
        y -= 14
        y = draw_wrapped_text(c, f"Thesis: {r['thesis']}", left, y, maxw, 12)

        y -= 6
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, f"Score: {r['score']} / 100   |   Action: {r.get('action','')} ({r.get('action_confidence','')})")
        y -= 14

        c.setFont("Helvetica", 10)
        c.drawString(left, y, "Score breakdown: " + ", ".join([f"{k}={v}" for k, v in (r.get("score_components") or {}).items()]))
        y -= 16

        # Thesis flags
        c.setFont("Helvetica-Bold", 11)
        c.drawString(left, y, "Thesis health flags")
        y -= 14
        c.setFont("Helvetica", 10)
        y = draw_wrapped_text(
            c,
            f"Health: {r.get('thesis_health_label','')}  |  Reds: {r.get('thesis_reds',0)}  |  Ambers: {r.get('thesis_ambers',0)}",
            left, y, maxw, 12
        )
        for sig in (r.get("thesis_signals") or [])[:10]:
            if y < 3.2 * cm:
                y = new_page(f"{r['ticker']} — thesis flags (continued)")
            y = draw_wrapped_text(c, f"- [{sig['level'].upper()}] {sig['name']}: {sig['reason']}", left, y, maxw, 12)

        # Market snapshot
        y -= 6
        if y < 5 * cm:
            y = new_page(f"{r['ticker']} — snapshot (continued)")
        c.setFont("Helvetica-Bold", 11)
        c.drawString(left, y, "Market snapshot")
        y -= 14
        c.setFont("Helvetica", 10)
        snap_lines = [
            f"Price: {r.get('price','n/a')}   Market cap: {fmt_money(r.get('market_cap'))}   Beta: {r.get('beta','n/a')}",
            f"12M max drawdown: {pct(r.get('dd_12m'))}   6M max drawdown: {pct(r.get('dd_6m'))}",
            f"P/E (trailing/forward): {r.get('trailing_pe','n/a')} / {r.get('forward_pe','n/a')}",
        ]
        for line in snap_lines:
            y = draw_wrapped_text(c, line, left, y, maxw, 12)

        # Financials
        y -= 6
        if y < 5 * cm:
            y = new_page(f"{r['ticker']} — financials (continued)")
        c.setFont("Helvetica-Bold", 11)
        c.drawString(left, y, "Financial momentum & resilience (where available)")
        y -= 14
        c.setFont("Helvetica", 10)
        fin_lines = [
            f"Revenue latest quarter: {fmt_money(r.get('rev_latest'))}   QoQ: {pct(r.get('rev_qoq'))}   YoY: {pct(r.get('rev_yoy'))}",
            f"Operating income (latest): {fmt_money(r.get('op_income_latest'))}   Gross profit (latest): {fmt_money(r.get('gross_profit_latest'))}",
            f"Cash: {fmt_money(r.get('cash_latest'))}   Debt: {fmt_money(r.get('debt_latest'))}   Operating cash flow: {fmt_money(r.get('ocf_latest'))}",
        ]
        for line in fin_lines:
            y = draw_wrapped_text(c, line, left, y, maxw, 12)

        # Filings
        y -= 6
        if y < 5 * cm:
            y = new_page(f"{r['ticker']} — filings (continued)")
        c.setFont("Helvetica-Bold", 11)
        c.drawString(left, y, "Regulatory & filings (SEC, US tickers only)")
        y -= 14
        c.setFont("Helvetica", 10)
        if r.get("sec_filings"):
            c.drawString(left, y, f"Recent filings (up to 8): Major={r.get('important_filings',0)}, Insider={r.get('insider_filings',0)}")
            y -= 12
            for f in (r.get("sec_filings") or [])[:8]:
                if y < 3.2 * cm:
                    y = new_page(f"{r['ticker']} — filings (continued)")
                c.drawString(left, y, f"- {f['filing_date']} — {f['form']}")
                y -= 12
        else:
            y = draw_wrapped_text(c, "No SEC filings available (non-US listing or lookup unavailable).", left, y, maxw, 12)

        # News
        y -= 6
        if y < 5 * cm:
            y = new_page(f"{r['ticker']} — news (continued)")
        c.setFont("Helvetica-Bold", 11)
        c.drawString(left, y, "News & catalysts (Google News RSS, lookback window)")
        y -= 14
        c.setFont("Helvetica", 10)
        if r.get("news_items"):
            proxy = r.get("news_proxy", {}) or {}
            c.drawString(left, y, f"News hits: {r.get('news_count',0)}   +kw hits: {proxy.get('pos_hits',0)}   -kw hits: {proxy.get('neg_hits',0)}")
            y -= 12
            for title in summarize_news_titles(r.get("news_items") or [], max_titles=10):
                if y < 3.2 * cm:
                    y = new_page(f"{r['ticker']} — news (continued)")
                y = draw_wrapped_text(c, f"- {title}", left, y, maxw, 12)
        else:
            y = draw_wrapped_text(c, "No recent news items found via RSS search (coverage limitation possible).", left, y, maxw, 12)

        # Rationale
        y -= 6
        if y < 5 * cm:
            y = new_page(f"{r['ticker']} — rationale (continued)")
        c.setFont("Helvetica-Bold", 11)
        c.drawString(left, y, "Auto-rationale (triage notes)")
        y -= 14
        c.setFont("Helvetica", 10)
        reasons = r.get("action_reasons", []) or []
        if reasons:
            for rr in reasons[:10]:
                if y < 3.2 * cm:
                    y = new_page(f"{r['ticker']} — rationale (continued)")
                y = draw_wrapped_text(c, f"- {rr}", left, y, maxw, 12)
        else:
            y = draw_wrapped_text(c, "No notable automated rationale generated.", left, y, maxw, 12)

    c.save()
    buf.seek(0)
    return buf.read()

# =============================================================================
# Core run
# =============================================================================

def run_monthly_scan(
    user_agent: str,
    lookback_days_news: int = 35,
    max_sec_filings: int = 25,
    extra_rss: Optional[List[str]] = None,
    prev_snapshot: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if extra_rss is None:
        extra_rss = []
    if prev_snapshot is None:
        prev_snapshot = {}

    cik_map = {}
    if user_agent.strip():
        try:
            cik_map = sec_ticker_cik_map(user_agent)
        except Exception:
            cik_map = {}

    universe_rows: List[Dict[str, Any]] = []
    snapshot: Dict[str, Any] = {"generated_at": iso(utc_now()), "companies": {}}
    prev_companies = prev_snapshot.get("companies") or {}

    for ticker, display_name, tags, thesis in WATCHLIST:
        info = yf_info(ticker)
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        market_cap = safe_num(info.get("marketCap"))
        trailing_pe = safe_num(info.get("trailingPE"))
        forward_pe = safe_num(info.get("forwardPE"))
        beta = safe_num(info.get("beta"))
        name = info.get("longName") or info.get("shortName") or display_name

        dd6 = None
        dd12 = None
        h6 = yf_history(ticker, "6mo")
        if not h6.empty and "Close" in h6.columns:
            dd6 = max_drawdown(h6["Close"])
        h12 = yf_history(ticker, "1y")
        if not h12.empty and "Close" in h12.columns:
            dd12 = max_drawdown(h12["Close"])

        feats = compute_financial_features(ticker)

        filings: List[Dict[str, Any]] = []
        important = 0
        insider = 0
        if user_agent.strip() and is_us_ticker(ticker):
            cik = cik_map.get(ticker.upper())
            if cik:
                filings = sec_recent_filings(cik, user_agent, max_items=max_sec_filings)
                important = sum(1 for f in filings if f["form"] in IMPORTANT_FORMS)
                insider = sum(1 for f in filings if f["form"] in INSIDER_FORMS)

        q = f'"{name}" OR {ticker} stock'
        gfeed = google_news_rss_query(q)
        news_items = fetch_rss(gfeed, lookback_days=lookback_days_news)

        if extra_rss:
            extra_items: List[Dict[str, Any]] = []
            for f in extra_rss[:10]:
                try:
                    extra_items.extend(fetch_rss(f, lookback_days=lookback_days_news))
                except Exception:
                    pass
            if extra_items:
                tokens = set([ticker.lower()])
                for w in re.split(r"[\W_]+", name.lower()):
                    if len(w) >= 4:
                        tokens.add(w)
                filt = []
                for it in extra_items:
                    txt = (it.get("title","") + " " + it.get("summary","")).lower()
                    if any(tok in txt for tok in tokens):
                        filt.append(it)
                news_items = news_items + filt

        seen = set()
        deduped = []
        for it in news_items:
            key = (it.get("title",""), it.get("link",""))
            if key not in seen:
                deduped.append(it)
                seen.add(key)
        news_items = deduped[:25]
        proxy = sentiment_proxy(news_items)

        score, components, _score_reasons = score_company(
            dd_12m=dd12,
            rev_yoy=feats.get("rev_yoy"),
            rev_qoq=feats.get("rev_qoq"),
            op_income=feats.get("op_income_latest"),
            gross_profit=feats.get("gross_profit_latest"),
            cash=feats.get("cash_latest"),
            debt=feats.get("debt_latest"),
            ocf=feats.get("ocf_latest"),
            important_filings=important,
            insider_filings=insider,
            news_pos=proxy["pos_hits"],
            news_neg=proxy["neg_hits"],
        )

        prev_row = prev_companies.get(ticker, {}) if isinstance(prev_companies, dict) else {}
        action, conf, action_reasons = action_recommendation(
            row={
                "ticker": ticker,
                "tags": tags,
                "score": score,
                "rev_yoy": feats.get("rev_yoy"),
                "rev_qoq": feats.get("rev_qoq"),
                "ocf_latest": feats.get("ocf_latest"),
                "cash_latest": feats.get("cash_latest"),
                "debt_latest": feats.get("debt_latest"),
                "dd_12m": dd12,
                "news_proxy": proxy,
                "important_filings": important,
            },
            prev=prev_row,
        )

        thesis_sigs, thesis_health = thesis_health_signals({
            "rev_yoy": feats.get("rev_yoy"),
            "rev_qoq": feats.get("rev_qoq"),
            "ocf_latest": feats.get("ocf_latest"),
            "cash_latest": feats.get("cash_latest"),
            "debt_latest": feats.get("debt_latest"),
            "dd_12m": dd12,
            "news_proxy": proxy,
            "important_filings": important,
            "tags": tags,
        })

        row = {
            "ticker": ticker,
            "name": name,
            "tags": tags,
            "thesis": thesis,
            "price": price,
            "market_cap": market_cap,
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "beta": beta,
            "dd_6m": dd6,
            "dd_12m": dd12,
            **feats,
            "sec_filings": filings,
            "important_filings": important,
            "insider_filings": insider,
            "news_items": news_items,
            "news_proxy": proxy,
            "news_count": len(news_items),
            "score": score,
            "score_components": components,
            "thesis_signals": [{"name": s.name, "level": s.level, "reason": s.reason} for s in thesis_sigs],
            "thesis_reds": thesis_health["reds"],
            "thesis_ambers": thesis_health["ambers"],
            "thesis_health_label": thesis_health["health_label"],
            "action": action,
            "action_confidence": conf,
            "action_reasons": action_reasons,
        }
        universe_rows.append(row)

        snapshot["companies"][ticker] = {
            "score": score,
            "dd_12m": float(dd12) if isinstance(dd12, float) else None,
            "important_filings": important,
            "news_count": len(news_items),
            "thesis_reds": int(thesis_health["reds"]),
            "thesis_ambers": int(thesis_health["ambers"]),
            "action": action,
        }

        time.sleep(0.06)

    return universe_rows, snapshot

# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="Next Tesla — Monthly Watchlist + Thesis + Actions", layout="wide")
st.title("Next Tesla Watchlist — Monthly Scan + Thesis Health + Actions + PDF")
st.caption("Bench (20) → score → thesis flags → actions → snapshots → allocation plan → PDF. Not investment advice.")

thesis_notes = load_thesis_notes()
portfolio_state = load_portfolio_state()
active_map = portfolio_state.get("active", {t: True for t in DEFAULT_TICKERS})

with st.sidebar:
    st.header("Run settings")
    user_agent = st.text_input(
        "SEC User-Agent (required for EDGAR; include real email)",
        value=thesis_notes.get("_sec_user_agent", "JuanEstradaWatchlist/1.0 (your_email@example.com)")
    )
    lookback_days = st.slider("News lookback days", 14, 90, 35)
    max_sec = st.slider("Max SEC filings per company", 10, 50, 25)

    st.subheader("Optional extra RSS feeds")
    extra_rss_text = st.text_area("Extra RSS feeds (one per line)", value=thesis_notes.get("_extra_rss_text", ""), height=90)
    extra_rss = [x.strip() for x in extra_rss_text.splitlines() if x.strip()]

    st.divider()
    st.header("Portfolio state (Active vs Inactive)")
    st.caption("Inactive = still analysed, but excluded from allocation if Active-only is on.")

    if st.button("Set ALL Active"):
        for t in active_map:
            active_map[t] = True
        save_portfolio_state({"active": active_map})
        st.success("All tickers set Active.")

    for t in DEFAULT_TICKERS:
        active_map[t] = st.checkbox(f"{t} Active", value=bool(active_map.get(t, True)), key=f"active_{t}")
    save_portfolio_state({"active": active_map})

    st.divider()
    st.header("Allocation plan")
    monthly_budget = st.number_input("Monthly budget (£)", min_value=0.0, value=float(thesis_notes.get("_monthly_budget", 200.0)), step=10.0)
    strategy = st.selectbox(
        "Allocation strategy",
        options=["Score-weighted", "Score / Drawdown (risk-adjusted proxy)"],
        index=int(thesis_notes.get("_strategy_index", 0))
    )
    max_stock_weight = st.slider("Max per stock (%)", 5, 40, int(thesis_notes.get("_max_stock_weight_pct", 15))) / 100.0
    min_stock_weight = st.slider("Min per stock (%)", 0, 10, int(thesis_notes.get("_min_stock_weight_pct", 2))) / 100.0
    max_optionality_bucket = st.slider("Max optionality bucket total (%)", 0, 60, int(thesis_notes.get("_max_optionality_bucket_pct", 25))) / 100.0
    optionality_cap_each = st.slider("Max per optionality stock (%)", 0, 30, int(thesis_notes.get("_optionality_cap_each_pct", 8))) / 100.0

    alloc_filter = st.multiselect("Allocate only to actions", options=["ADD", "HOLD", "PAUSE", "EXIT"], default=["ADD", "HOLD"])
    active_only_alloc = st.checkbox("Allocate only to Active tickers", value=True)

    st.divider()
    run_btn = st.button("Run monthly scan (store snapshot) + build PDF", type="primary")

# persist a few prefs
thesis_notes["_sec_user_agent"] = user_agent
thesis_notes["_extra_rss_text"] = extra_rss_text
thesis_notes["_monthly_budget"] = monthly_budget
thesis_notes["_strategy_index"] = ["Score-weighted", "Score / Drawdown (risk-adjusted proxy)"].index(strategy)
thesis_notes["_max_stock_weight_pct"] = int(max_stock_weight * 100)
thesis_notes["_min_stock_weight_pct"] = int(min_stock_weight * 100)
thesis_notes["_max_optionality_bucket_pct"] = int(max_optionality_bucket * 100)
thesis_notes["_optionality_cap_each_pct"] = int(optionality_cap_each * 100)
save_thesis_notes(thesis_notes)

if not run_btn:
    st.info("Configure the sidebar, then click **Run monthly scan (store snapshot) + build PDF**.")
    st.stop()

mkey = month_key(utc_now())
prev_key, prev_snapshot = load_latest_snapshot_before(mkey)
prev_exists = bool((prev_snapshot.get("companies") or {}))

with st.spinner("Gathering signals (market, financials, filings, news), scoring, and generating actions..."):
    universe_rows, snapshot = run_monthly_scan(
        user_agent=user_agent.strip(),
        lookback_days_news=int(lookback_days),
        max_sec_filings=int(max_sec),
        extra_rss=extra_rss,
        prev_snapshot=prev_snapshot if prev_exists else {},
    )

save_snapshot(snapshot, mkey)
changes = diff_summary(snapshot, prev_snapshot) if prev_exists else {
    "score_moves": [], "action_changes": [], "new_major_filings": [], "news_spikes": [], "drawdown_shocks": [], "risk_flag_spikes": []
}

# Dashboard
df = pd.DataFrame([{
    "Ticker": r["ticker"],
    "Active": bool(active_map.get(r["ticker"], True)),
    "Name": r["name"],
    "Tags": ", ".join(r["tags"]),
    "Score": r["score"],
    "Action": r.get("action", ""),
    "Health": r.get("thesis_health_label", ""),
    "Reds": r.get("thesis_reds", 0),
    "Ambers": r.get("thesis_ambers", 0),
    "Optionality": is_optionality(r["tags"]),
    "12M Max DD": pct(r.get("dd_12m")),
    "Rev YoY": pct(r.get("rev_yoy")),
    "Rev QoQ": pct(r.get("rev_qoq")),
    "Cash": fmt_money(r.get("cash_latest")),
    "Debt": fmt_money(r.get("debt_latest")),
    "OCF": fmt_money(r.get("ocf_latest")),
    "Major Filings": r.get("important_filings", 0),
    "News hits": r.get("news_count", 0),
} for r in universe_rows]).sort_values(["Active", "Score", "Reds", "Ambers"], ascending=[False, False, True, True])

st.subheader("Executive dashboard")
st.dataframe(df, use_container_width=True)

st.divider()
st.subheader("What changed since last snapshot (urgency)")
if not prev_exists:
    st.write(f"No previous snapshot found before **{mkey}** — baseline created this month.")
else:
    st.caption(f"Comparing **{mkey}** vs **{prev_key}**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Action changes:**")
        if changes["action_changes"]:
            for t, prev_a, cur_a in changes["action_changes"][:12]:
                st.write(f"- {t}: {prev_a} → **{cur_a}**")
        else:
            st.write("None detected.")
        st.markdown("**Risk spikes (reds increased):**")
        if changes["risk_flag_spikes"]:
            for t, prev_r, cur_r in changes["risk_flag_spikes"][:12]:
                st.write(f"- {t}: red flags {prev_r} → **{cur_r}**")
        else:
            st.write("None detected.")
    with col2:
        st.markdown("**Largest score moves (>= 8 points):**")
        if changes["score_moves"]:
            for t, delta, prev_s, cur_s in changes["score_moves"][:12]:
                st.write(f"- {t}: {prev_s} → {cur_s} ({delta:+d})")
        else:
            st.write("None detected.")

st.divider()
st.subheader("Allocation plan (planning aid)")
alloc_df, alloc_meta = allocate_pounds(
    rows=sorted(universe_rows, key=lambda x: x.get("score", 0), reverse=True),
    monthly_budget=float(monthly_budget),
    strategy=strategy,
    max_stock_weight=float(max_stock_weight),
    min_stock_weight=float(min_stock_weight),
    max_optionality_bucket=float(max_optionality_bucket),
    optionality_cap_each=float(optionality_cap_each),
    include_only_actions=alloc_filter if alloc_filter else None,
    active_only=bool(active_only_alloc),
    portfolio_active_map=active_map,
)

if alloc_df.empty:
    st.warning(f"Allocation could not be computed. {alloc_meta.get('error','')}")
else:
    st.dataframe(
        alloc_df.assign(
            TargetWeightPct=(alloc_df["TargetWeight"] * 100).round(2),
            TargetGBP=alloc_df["TargetGBP"].round(2),
        )[["Ticker","Name","Score","Action","Optionality","TargetWeightPct","TargetGBP","Tags"]],
        use_container_width=True
    )
    st.bar_chart(alloc_df[["Ticker","TargetWeight"]].set_index("Ticker"))

st.divider()
st.subheader("Monthly PDF report")
with st.spinner("Generating PDF report..."):
    pdf_bytes = build_pdf_report(
        generated_at=utc_now(),
        universe_rows=universe_rows,
        changes=changes,
        prev_exists=prev_exists,
        prev_key=prev_key,
        current_key=mkey,
        alloc_df=alloc_df if not alloc_df.empty else None,
        alloc_meta=alloc_meta,
    )

st.success(f"PDF generated and snapshot saved for {mkey} (data/watchlist_snapshot_{mkey}.json).")
st.download_button(
    label="Download monthly PDF report",
    data=pdf_bytes,
    file_name=f"next_tesla_monthly_report_{utc_now().strftime('%Y_%m_%d')}.pdf",
    mime="application/pdf"
)

