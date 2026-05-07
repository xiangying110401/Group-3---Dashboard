import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

# =========================================================
# CUSTOM COLORMAP
# =========================================================
RB_CMAP = LinearSegmentedColormap.from_list(
    "rb_div",
    ["#d73027", "#f46d43", "#fdae61", "#fee090",
     "#ffffff",
     "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"],
    N=256,
)

# =========================================================
# PAGE CONFIG 
# =========================================================
st.set_page_config(
    page_title="Group 3 Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# GLOBAL STYLE
# =========================================================
st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Segoe UI", sans-serif; }
.main { background-color: #f4f6fb; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
    border-right: 1px solid #2b3648;
}
section[data-testid="stSidebar"] * { color: #f9fafb !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label { color: #e5e7eb !important; font-weight: 600; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: #374151 !important; border: 1px solid #4b5563 !important; }
section[data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] { padding-top: 8px; }
.sidebar-title { font-size: 12px; font-weight: 700; color: #93c5fd; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.sidebar-group { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 10px 12px; margin-bottom: 10px; }

/* CARDS */
.card { background: white; border-radius: 16px; padding: 16px 18px; border: 1px solid #e5e7eb; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05); height: 100%; }
.small-card { background: white; border-radius: 14px; padding: 14px 16px; border: 1px solid #e5e7eb; box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04); }
.section-title { font-size: 18px; font-weight: 800; color: #111827; margin-bottom: 10px; }
.table-title { font-size: 16px; font-weight: 800; color: #111827; margin-bottom: 8px; }

/* KPI */
.kpi-wrap { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.kpi-left { flex: 1; }
.kpi-title { font-size: 12px; color: #6b7280; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }
.kpi-value { font-size: 28px; font-weight: 800; color: #111827; margin-top: 6px; line-height: 1.1; }
.kpi-delta-pos { color: #16a34a; font-size: 13px; font-weight: 700; margin-top: 4px; }
.kpi-delta-neg { color: #dc2626; font-size: 13px; font-weight: 700; margin-top: 4px; }
.kpi-icon { width: 46px; height: 46px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; font-weight: 700; }
.icon-blue { background: #dbeafe; color: #2563eb; }
.icon-green { background: #dcfce7; color: #16a34a; }
.icon-orange { background: #ffedd5; color: #ea580c; }
.icon-red { background: #fee2e2; color: #dc2626; }

/* PROGRESS */
.progress-label { font-size: 12px; font-weight: 700; color: #6b7280; margin-bottom: 6px; }
.progress-track { width: 100%; height: 9px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }
.progress-fill-blue { height: 100%; background: linear-gradient(90deg, #60a5fa, #2563eb); border-radius: 999px; }
.progress-fill-green { height: 100%; background: linear-gradient(90deg, #86efac, #16a34a); border-radius: 999px; }
.progress-fill-orange { height: 100%; background: linear-gradient(90deg, #fdba74, #f59e0b); border-radius: 999px; }
.progress-fill-red { height: 100%; background: linear-gradient(90deg, #fca5a5, #dc2626); border-radius: 999px; }
.progress-value { font-size: 13px; font-weight: 700; margin-top: 5px; color: #111827; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# APP ROUTING 
# =========================================================
st.sidebar.markdown('<div class="sidebar-title">NAVIGATION</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    ["Task 1: Asset Selection", "Task 2: Portfolio Performance", "Task 3: Risk & Stress Testing"]
)
st.sidebar.markdown("---")

# =========================================================
# PAGE 1: TASK 1 (YOUR DASHBOARD)
# =========================================================
if page == "Task 1: Asset Selection":
    
    TOP30_INFO_FILE = "rolling_regression_top30_with_company_info.csv"
    TOP30_RETURN_FILE = "rolling_regression_top30_log_returns.csv"
    MARKET_RETURN_FILE = "market_return.csv"

    TRADING_DAYS = 252
    RISK_FREE_RATE = 0.0275

    # Helpers
    def standardize_columns(df):
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df
    def find_date_column(df):
        for c in df.columns:
            if c.strip().lower() == "date": return c
        return None
    def find_market_column(df):
        preferred = ["Market_Return", "market_return", "KLCI", "KLCI_Return", "Return", "return", "LogReturn"]
        cols = {c.lower(): c for c in df.columns}
        for p in preferred:
            if p.lower() in cols: return cols[p.lower()]
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]): return c
        return None
    def detect_col(df, names):
        for n in names:
            if n in df.columns: return n
        return None 
    def annualized_return(log_ret):
        log_ret = pd.to_numeric(log_ret, errors="coerce").dropna()
        if len(log_ret) == 0: return np.nan
        return np.exp(log_ret.mean() * TRADING_DAYS) - 1
    def annualized_vol(log_ret):
        log_ret = pd.to_numeric(log_ret, errors="coerce").dropna()
        if len(log_ret) == 0: return np.nan
        return log_ret.std() * np.sqrt(TRADING_DAYS)
    def sharpe_ratio(log_ret, rf=RISK_FREE_RATE):
        r = annualized_return(log_ret)
        v = annualized_vol(log_ret)
        if pd.isna(r) or pd.isna(v) or v == 0: return np.nan
        return (r - rf) / v
    def beta_vs_market(asset, market):
        x = pd.to_numeric(market, errors="coerce")
        y = pd.to_numeric(asset, errors="coerce")
        df = pd.concat([x, y], axis=1).dropna()
        if len(df) < 2: return np.nan
        m = df.iloc[:, 0]
        a = df.iloc[:, 1]
        var_m = np.var(m, ddof=1)
        if var_m == 0: return np.nan
        cov = np.cov(a, m, ddof=1)[0, 1]
        return cov / var_m
    def fmt_pct(x): return "N/A" if pd.isna(x) else f"{x*100:,.2f}%"
    def fmt_num(x): return "N/A" if pd.isna(x) else f"{x:,.2f}"
    def curve_from_log(log_ret, start=100):
        log_ret = pd.to_numeric(log_ret, errors="coerce").fillna(0)
        return start * np.exp(log_ret.cumsum())
    def clamp_percent(value, min_value=0, max_value=100):
        if pd.isna(value): return 0
        return max(min_value, min(max_value, value))
    def progress_html(label, value, fill_class):
        value = clamp_percent(value)
        return f"""
        <div style="margin-bottom:12px;">
            <div class="progress-label">{label}</div>
            <div class="progress-track">
                <div class="{fill_class}" style="width:{value}%;"></div>
            </div>
            <div class="progress-value">{value:.0f}%</div>
        </div>
        """

    @st.cache_data
    def load_data_t1():
        info = standardize_columns(pd.read_csv(TOP30_INFO_FILE))
        ret = standardize_columns(pd.read_csv(TOP30_RETURN_FILE))
        market = standardize_columns(pd.read_csv(MARKET_RETURN_FILE))
        ret_date = find_date_column(ret)
        market_date = find_date_column(market)
        ret[ret_date] = pd.to_datetime(ret[ret_date], errors="coerce")
        market[market_date] = pd.to_datetime(market[market_date], errors="coerce")
        ret = ret.dropna(subset=[ret_date]).sort_values(ret_date)
        market = market.dropna(subset=[market_date]).sort_values(market_date)
        return info, ret, market

    info_df, returns_df, market_df = load_data_t1()

    ticker_col = detect_col(info_df, ["Ticker", "ticker", "Code", "code", "Symbol"])
    company_col = detect_col(info_df, ["Company", "CompanyName", "Name", "name"])
    sector_col = detect_col(info_df, ["Sector", "sector", "Industry", "industry"])
    m1_col = detect_col(info_df, ["method1_score", "Method1_Score", "method_1_score"])
    m2_col = detect_col(info_df, ["method2_score", "Method2_Score", "method_2_score"])
    beta_std_col = detect_col(info_df, ["beta_std_across_windows", "beta_std_across_windows"])
    alpha_std_col = detect_col(info_df, ["alpha_std_across_windows", "alpha_std_across_windows"])
    r2_std_col = detect_col(info_df, ["r2_std_across_windows", "r2_std_across_windows"])

    ret_date_col = find_date_column(returns_df)
    market_date_col = find_date_column(market_df)
    market_col = find_market_column(market_df)

    all_tickers = [str(x).strip() for x in info_df[ticker_col].dropna().astype(str)]
    all_tickers = [t for t in all_tickers if t in returns_df.columns]

    st.sidebar.markdown("### Asset Selection")
    st.sidebar.markdown('<div class="sidebar-group">', unsafe_allow_html=True)

    sector_options = ["All"]
    if sector_col:
        sector_options += sorted(info_df[sector_col].dropna().astype(str).unique().tolist())

    selected_sector = st.sidebar.selectbox("Sector", sector_options)

    search_tickers = st.sidebar.multiselect(
        "🔍 Search & Add Ticker(s)", 
        options=all_tickers, 
        default=[], 
        max_selections=10,  
        placeholder="👉Type here to search (e.g. 6432.KL)...", 
        help="Click here and TYPE the stock name to search and add it."
    )
    top_n = st.sidebar.slider("Portfolio Size", 5, min(30, len(all_tickers)), min(30, len(all_tickers)))
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    filtered_info = info_df.copy()
    if selected_sector != "All" and sector_col:
        filtered_info = filtered_info[filtered_info[sector_col].astype(str) == selected_sector]
    if search_tickers:
        filtered_info = filtered_info[filtered_info[ticker_col].astype(str).isin(search_tickers)]
    
    filtered_info = filtered_info.head(top_n)
    selected_tickers = [t for t in filtered_info[ticker_col].astype(str) if t in returns_df.columns]

    if len(selected_tickers) == 0:
        st.warning("No stocks available after filter.")
        st.stop()

    portfolio_series = returns_df[selected_tickers].mean(axis=1, skipna=True)
    portfolio_df = pd.DataFrame({"Date": returns_df[ret_date_col], "Portfolio": portfolio_series})

    merged = portfolio_df.merge(
        market_df[[market_date_col, market_col]].rename(columns={market_date_col: "Date", market_col: "Market"}),
        on="Date", how="inner"
    ).dropna()

    portfolio_ret = annualized_return(merged["Portfolio"])
    market_ret = annualized_return(merged["Market"])
    portfolio_vol = annualized_vol(merged["Portfolio"])
    market_vol = annualized_vol(merged["Market"])
    portfolio_sharpe = sharpe_ratio(merged["Portfolio"])
    market_sharpe = sharpe_ratio(merged["Market"])
    portfolio_beta = beta_vs_market(merged["Portfolio"], merged["Market"])

    rows = []
    for t in selected_tickers:
        s = returns_df[[ret_date_col, t]].merge(
            market_df[[market_date_col, market_col]].rename(columns={market_date_col: ret_date_col}),
            on=ret_date_col, how="inner"
        ).dropna()

        rows.append({
            "Ticker": t,
            "Annual Return": annualized_return(s[t]),
            "Annual Volatility": annualized_vol(s[t]),
            "Sharpe Ratio": sharpe_ratio(s[t]),
            "Beta vs KLCI": beta_vs_market(s[t], s[market_col])
        })
    metrics_df = pd.DataFrame(rows)

    display_df = filtered_info.copy()
    display_df["Ticker"] = display_df[ticker_col].astype(str)
    display_df = display_df.merge(metrics_df, on="Ticker", how="left")

    return_score = clamp_percent(((portfolio_ret if pd.notna(portfolio_ret) else 0) + 0.20) / 0.40 * 100)
    vol_score = clamp_percent((1 - min((portfolio_vol if pd.notna(portfolio_vol) else 1), 0.40) / 0.40) * 100)
    sharpe_score = clamp_percent(((portfolio_sharpe if pd.notna(portfolio_sharpe) else 0) + 0.5) / 2.5 * 100)
    beta_score = clamp_percent((1 - abs((portfolio_beta if pd.notna(portfolio_beta) else 1) - 0.7) / 1.0) * 100)

    st.markdown('<div class="dashboard-title" style="font-size: 28px; font-weight: 800; color: #111827;">Top 30 Stable Stocks Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle" style="font-size: 13px; color: #6b7280; margin-bottom: 1rem;">Dashboards / Asset Selection / Investor View</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        delta = portfolio_ret - market_ret if pd.notna(portfolio_ret) and pd.notna(market_ret) else np.nan
        if pd.isna(delta):
            status_text, color_class = "N/A", "kpi-delta-pos"
        elif delta >= 0:
            status_text, color_class = f"vs KLCI (increase {fmt_pct(delta)})", "kpi-delta-pos"
        else:
            status_text, color_class = f"vs KLCI Annual (decrease {fmt_pct(abs(delta))})", "kpi-delta-neg"

        st.markdown(f"""
        <div class="card"><div class="kpi-wrap"><div class="kpi-left"><div class="kpi-title">Top 30 Annual Return</div>
        <div class="kpi-value">{fmt_pct(portfolio_ret)}</div><div class="{color_class}">{status_text}</div>
        </div><div class="kpi-icon icon-green">↗</div></div></div>
        """, unsafe_allow_html=True)

    with k2:
        delta = portfolio_vol - market_vol if pd.notna(portfolio_vol) and pd.notna(market_vol) else np.nan
        if pd.isna(delta):
            status_text, color_class = "N/A", "kpi-delta-pos"
        elif delta >= 0:
            status_text, color_class = f"vs KLCI (increase {fmt_pct(abs(delta))})", "kpi-delta-neg"
        else:
            status_text, color_class = f"vs KLCI (decrease {fmt_pct(delta)})", "kpi-delta-pos"

        st.markdown(f"""
        <div class="card"><div class="kpi-wrap"><div class="kpi-left"><div class="kpi-title">Top 30 Avg Volatility</div>
        <div class="kpi-value">{fmt_pct(portfolio_vol)}</div><div class="{color_class}">{status_text}</div>
        </div><div class="kpi-icon icon-red">↓</div></div></div>
        """, unsafe_allow_html=True)

    with k3:
        delta = portfolio_sharpe - market_sharpe if pd.notna(portfolio_sharpe) and pd.notna(market_sharpe) else np.nan
        if pd.isna(delta):
            status_text, color_class = "N/A", "kpi-delta-pos"
        elif delta >= 0:
            status_text, color_class = f"vs KLCI (increase {fmt_pct(delta)})", "kpi-delta-pos"
        else:
            status_text, color_class = f"vs KLCI (decrease {fmt_pct(abs(delta))})", "kpi-delta-neg"

        st.markdown(f"""
        <div class="card"><div class="kpi-wrap"><div class="kpi-left"><div class="kpi-title">Top 30 Sharpe Ratio</div>
        <div class="kpi-value">{fmt_num(portfolio_sharpe)}</div><div class="{color_class}">{status_text}</div>
        </div><div class="kpi-icon icon-orange">⚖</div></div></div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="card"><div class="kpi-wrap"><div class="kpi-left"><div class="kpi-title">Top 30 Avg Beta</div>
        <div class="kpi-value">{fmt_num(portfolio_beta)}</div>
        <div class="{'kpi-delta-pos' if pd.notna(portfolio_beta) and portfolio_beta < 1 else 'kpi-delta-neg'}">Target below 1.00</div>
        </div><div class="kpi-icon icon-blue">β</div></div></div>
        """, unsafe_allow_html=True)
    st.write("")
    
    left, right = st.columns([2.2, 1])
    with left:
        st.markdown('<div class="card"><div class="section-title">Top 30 Cumulative Performance Vs KLCI </div>', unsafe_allow_html=True)
        st.caption("Note: Chart scales and start dates adjust automatically based on the selected portfolio's available data.")
        perf = merged.copy()
        perf["Top 30 Portfolio"] = curve_from_log(perf["Portfolio"], 100)
        perf["KLCI"] = curve_from_log(perf["Market"], 100)
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(perf["Date"], perf["Top 30 Portfolio"], linewidth=2.6, label="Top 30 Portfolio")
        ax.plot(perf["Date"], perf["KLCI"], linewidth=2.2, label="KLCI")
        ax.set_ylabel("Growth of RM100")
        ax.set_xlabel("")
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><div class="section-title">Top 30 Portfolio Quality</div>', unsafe_allow_html=True)
        st.markdown(progress_html("Return Strength", return_score, "progress-fill-green"), unsafe_allow_html=True)
        st.markdown(progress_html("Risk Control", vol_score, "progress-fill-red"), unsafe_allow_html=True)
        st.markdown(progress_html("Risk-Adjusted Return", sharpe_score, "progress-fill-orange"), unsafe_allow_html=True)
        st.markdown(progress_html("Market Stability", beta_score, "progress-fill-blue"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    left2, right2 = st.columns([1.25, 1.25])
    with left2:
        st.markdown('<div class="card"><div class="section-title">Risk Return Profile</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.scatter(metrics_df["Annual Volatility"], metrics_df["Annual Return"], s=70, alpha=0.85)
        for _, row in metrics_df.head(min(10, len(metrics_df))).iterrows():
            ax.annotate(row["Ticker"], (row["Annual Volatility"], row["Annual Return"]), fontsize=8)
        ax.scatter([portfolio_vol], [portfolio_ret], s=220, marker="*", label="Portfolio")
        ax.scatter([market_vol], [market_ret], s=180, marker="X", label="KLCI")
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Return")
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right2:
        st.markdown('<div class="card"><div class="section-title">Sector Distribution</div>', unsafe_allow_html=True)
        if sector_col:
            sector_counts = filtered_info[sector_col].astype(str).value_counts()
            fig, ax = plt.subplots(figsize=(6.8, 4.8))
            ax.pie(sector_counts.values, labels=sector_counts.index, autopct="%1.0f%%", startangle=90)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Sector column not found.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""<div class="small-card"><div class="kpi-wrap"><div class="kpi-left"><div class="kpi-title">KLCI Annual Return</div> 
        <div class="kpi-value" style="font-size:24px;">{fmt_pct(market_ret)}</div></div><div class="kpi-icon icon-green">K</div></div></div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="small-card"><div class="kpi-wrap"><div class="kpi-left"><div class="kpi-title">KLCI Annaul Volatility</div> 
        <div class="kpi-value" style="font-size:24px;">{fmt_pct(market_vol)}</div></div><div class="kpi-icon icon-red">!</div></div></div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""<div class="small-card"><div class="kpi-wrap"><div class="kpi-left"><div class="kpi-title">KLCI Annual Sharpe</div> 
        <div class="kpi-value" style="font-size:24px;">{fmt_num(market_sharpe)}</div></div><div class="kpi-icon icon-orange">S</div></div></div>""", unsafe_allow_html=True)
    with s4:
        st.markdown(f"""<div class="small-card"><div class="kpi-wrap"><div class="kpi-left"><div class="kpi-title">Stocks Selected</div>
        <div class="kpi-value" style="font-size:24px;">{len(selected_tickers)}</div></div><div class="kpi-icon icon-blue">30</div></div></div>""", unsafe_allow_html=True)

    st.write("")

    left3, right3 = st.columns([1.1, 1.4])
    with left3:
        st.markdown('<div class="card"><div class="section-title">Stability Scoreboard</div>', unsafe_allow_html=True)
        required_cols = ["beta_std_across_windows", "alpha_std_across_windows", "r2_std_across_windows", "stability_score"]
        missing_cols = [col for col in required_cols if col not in filtered_info.columns]
        if not missing_cols:
            temp = filtered_info[[ticker_col, "beta_std_across_windows", "alpha_std_across_windows", "r2_std_across_windows", "stability_score"]].copy()
            temp = temp.sort_values("stability_score", ascending=False).head(30)
            show_df = temp.rename(columns={ticker_col: "Ticker", "beta_std_across_windows": "Beta", "alpha_std_across_windows": "Alpha", "r2_std_across_windows": "R2", "stability_score": "Stability Score"}).set_index("Ticker")
            styled_df = show_df.style.background_gradient(cmap="Blues", subset=["Stability Score"]).bar(subset=["Beta", "Alpha", "R2"], color='#e5e7eb').format("{:.3f}")
            st.dataframe(styled_df, use_container_width=True, height=390)
            st.markdown("""<div style="display: flex; justify-content: center; font-size: 13px; color: #4b5563; margin-top: 12px; background: #f8fafc; padding: 10px 14px; border-radius: 8px; border: 1px solid #e5e7eb;"><div style="display: flex; align-items: center; gap: 8px;"><div style="width: 14px; height: 14px; background: linear-gradient(90deg, #dbeafe, #1e3a8a); border-radius: 3px;"></div><span><b>Stability Score:</b> The darker the color, the greater the stability.</span></div></div>""", unsafe_allow_html=True)
        else:
            st.error(f"Missing columns: {missing_cols}")
        st.markdown('</div>', unsafe_allow_html=True)

    with right3:
        st.markdown('<div class="card"><div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = returns_df[selected_tickers].corr()
        n = corr.shape[0]
        
        # Adjusted figsize to fit well inside the Streamlit column layout
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        
        sns.heatmap(
            corr, ax=ax,
            cmap=RB_CMAP, vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.3, linecolor="#dddddd",
            annot=(n <= 30), fmt=".2f", annot_kws={"size": 6},
            cbar_kws={"shrink": 0.75},
        )
        
        # Format the colorbar to match Chart.py
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.set_ticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0"])
        cbar.ax.tick_params(labelsize=8)
        
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.tick_params(axis="y", rotation=0,  labelsize=7)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PAGE 2: TASK 2 
# =========================================================
elif page == "Task 2: Portfolio Performance":

    @st.cache_data
    def load_data_t2():
        data = pd.read_csv("rolling_regression_top30_log_returns.csv", index_col=0).sort_index(axis=1, ascending=True)
        info = pd.read_csv("rolling_regression_top30_with_company_info.csv")
        info.set_index('Ticker', inplace=True)
        sharpe = pd.read_csv("Optimal_Weights(Sharpe).csv", index_col=0)
        comparison = pd.read_csv("Comparison(Sharpe).csv", index_col=0).sort_index(axis=1, ascending=True)
        df_line = pd.read_csv("Portfolio_Cum_Returns.csv", index_col = 0)
        performance = pd.read_csv("Performance.csv", index_col = 0)
        return data, info, sharpe, comparison, df_line, performance

    try:
        data, info, sharpe, comparison, df_line, performance = load_data_t2()
    except Exception as e:
        st.error(f"Error loading Task 2 data. Please ensure CSV files are in the directory. Error: {e}")
        st.stop()

    trading_day = 252
    mean_returns = data.mean().values * trading_day
    cov_matrix = data.cov().values * trading_day
    tickers = data.columns

    st.sidebar.markdown("### Sensitivity Analysis")
    st.sidebar.markdown('<div class="sidebar-group">', unsafe_allow_html=True)
    
    ticker_to_name = info['Company'].to_dict()
    delta_lot_options = [-3, -2, -1, 0, 1, 2, 3] 
    DEFAULT_RF = 3.19
    DEFAULT_LOT = 0 

    def optimize_portfolio(mean_returns, cov_matrix, R_f=0.0275):
        n = len(mean_returns)
        rho = 10

        def port_return(w):
            return np.dot(w, mean_returns)

        def port_var(w):
            return np.dot(w.T, np.dot(cov_matrix, w))

        def port_vol(w):
            return np.sqrt(port_var(w) + 1e-12)

        def obj_f(w):
            return -(port_return(w) - R_f) / port_vol(w)

        def h(w):
            return np.sum(w) - 1

        def g(w):
            return -w

        def grad_L(w, lam, mu):
            ret = np.dot(w, mean_returns)
            sigma = np.sqrt(w.T @ cov_matrix @ w + 1e-12)

            grad_sharpe = (mean_returns / sigma
                        - ((ret - R_f) / sigma**3) * (cov_matrix @ w))

            grad_obj = -grad_sharpe

            grad_h = np.ones_like(w)
            eq_grad = (lam + rho * h(w)) * grad_h

            ineq_grad = -(mu + rho * np.maximum(0, g(w)))

            return grad_obj + eq_grad + ineq_grad

        w = np.ones(n) / n
        lam = 0.5
        mu = np.zeros(n)
        w_store = []

        for _ in range(1000):
            for _ in range(1000):
                w_store.append(w)
                grad = grad_L(w, lam, mu)
                w = w - 0.001 * grad

                if np.linalg.norm(grad) < 1e-8:
                    break

            w = w_store[-1]
            lam = lam + rho * h(w)
            mu = np.maximum(0, mu + rho * g(w))
            grad = grad_L(w,lam,mu)

            if np.linalg.norm(grad) < 1e-8:
                break

        w[w < 0] = 0
        w = w.round(4) / w.round(4).sum()

        return w

    if "rf_slider" not in st.session_state:
        st.session_state["rf_slider"] = DEFAULT_RF
        
    if "delta_lot_select" not in st.session_state:
        st.session_state["delta_lot_select"] = DEFAULT_LOT

    def reset_values():
        st.session_state["rf_slider"] = DEFAULT_RF
        st.session_state["delta_lot_select"] = DEFAULT_LOT

    rf_val = st.sidebar.slider(
        "Risk-Free Rate (%)",
        2.8, 3.5, 
        key="rf_slider",  
        step=0.05
    )
    rf = rf_val / 100

    @st.cache_data
    def get_optimized_weights(mean_returns, cov_matrix, rf):
        return optimize_portfolio(mean_returns, cov_matrix, R_f=rf)

    w = get_optimized_weights(mean_returns, cov_matrix, rf)

    stock = st.sidebar.selectbox(
        "Select stock", 
        options=tickers, 
        format_func=lambda x: ticker_to_name.get(x, x) 
    )

    delta_lot = st.sidebar.selectbox(
        "Changing lot", 
        options=delta_lot_options,
        key="delta_lot_select" 
    )

    st.sidebar.button("Reset to Default", on_click=reset_values, type="primary", use_container_width=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    open_price = comparison['Price']
    fund = 5000000
    theoritical_capital = w * fund
    shares = theoritical_capital / open_price
    lots = np.floor(shares / 100) 

    new_lot = lots.copy()
    new_lot[stock] = max(0, new_lot[stock] + delta_lot)  

    investment = new_lot * open_price * 100
    total_cost = investment.sum().round(2)
    remain = (fund - total_cost).round(2)
        
    w = investment / total_cost

    portfolio_return = mean_returns @ w
    portfolio_risk = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe_ratio = (portfolio_return - rf) / portfolio_risk
    w = pd.Series(w, index=tickers)
    count_stocks = w[w>0].count()

    w_top15 = w.sort_values(ascending=False).iloc[:15]
    w_top15.index = w_top15.index.map(lambda x: ticker_to_name.get(x, x))

    industry = info['Sector']
    df1 = pd.DataFrame({
        "Company": info['Company'],
        "Industry": sharpe.index.map(industry),
        "Lots(original)": comparison['Lots'],
        "Lots(after)": new_lot
    })
    industry_counts = df1["Industry"].value_counts()

    base_color = "#2563eb"
    n_industries = len(industry_counts)
    cmap = mcolors.LinearSegmentedColormap.from_list("my_gradient", [base_color, "#dbeafe"])
    colors = [cmap(i) for i in np.linspace(0, 0.8, max(1, n_industries))] 

    fig2, ax2 = plt.subplots(figsize=(3, 3))
    wedges, texts = ax2.pie(
        industry_counts, startangle=90,
        wedgeprops={'width': 0.3, 'edgecolor': 'white'}, colors=colors  
    )
    ax2.legend(wedges, industry_counts.index, bbox_to_anchor=(1, 0.8), frameon=False, fontsize=8)
    ax2.text(0, 0, "Industry", ha='center', va='center', fontsize=12, fontweight='bold', color="#111827")

    s = performance["Sharpe Portfolio"]
    e = performance["Equal Weight"]

    sharpe_line = df_line["Sharpe_Return"]
    equal_line = df_line["Equal_Weight_Return"]
    df = pd.DataFrame({
        "Sharpe Portfolio": sharpe_line,
        "Equal Weight Portfolio": equal_line
    })
    df.index = pd.to_datetime(df.index)

    performance = pd.DataFrame({
        "Sharpe": [
            s["Cumulative Return"], s["Profit Today"],
            s["Annualized Return"], s["Annualized Volatility"],
            s["Sharpe Ratio"], s["Max Drawdown"]
        ],
        "Equal Weight": [
            e["Cumulative Return"], e["Profit Today"],
            e["Annualized Return"], e["Annualized Volatility"],
            e["Sharpe Ratio"], e["Max Drawdown"]        
            ]
    }, index=["Cumulative Return","Profit today", "Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown"])

    st.markdown('<div class="dashboard-title" style="font-size: 28px; font-weight: 800; color: #111827;">Portfolio Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle" style="font-size: 13px; color: #6b7280; margin-bottom: 1rem;">Dashboards / Portfolio Performance / Scenario Analysis</div>', unsafe_allow_html=True)

    def build_kpi(title, value, icon, color_class):
        return f"""<div class="small-card" style="height: 100%;"><div class="kpi-wrap"><div class="kpi-left"><div class="kpi-title">{title}</div>
        <div class="kpi-value" style="font-size:22px;">{value}</div></div><div class="kpi-icon {color_class}">{icon}</div></div></div>"""

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: st.markdown(build_kpi("Initial Fund", "RM 5M", "$", "icon-green"), unsafe_allow_html=True)
    with k2: st.markdown(build_kpi("Stocks", count_stocks, "📊", "icon-blue"), unsafe_allow_html=True)
    with k3: st.markdown(build_kpi("Industries", len(industry_counts), "🏢", "icon-orange"), unsafe_allow_html=True)
    with k4: st.markdown(build_kpi("Return", f"{portfolio_return:.2%}", "↗", "icon-green"), unsafe_allow_html=True)
    with k5: st.markdown(build_kpi("Risk", f"{portfolio_risk:.2%}", "!", "icon-red"), unsafe_allow_html=True)
    with k6: st.markdown(build_kpi("Sharpe", f"{sharpe_ratio:.2f}", "⚖", "icon-blue"), unsafe_allow_html=True)


    st.write("")

    col1, col2 = st.columns([1.1, 1.9])
    with col1:
        st.markdown('<div class="card"><div class="section-title">Stock Details & Industry</div>', unsafe_allow_html=True)
        st.write("")
        st.write("")
        price = open_price.get(stock) if isinstance(open_price, dict) else open_price.loc[stock]
        company_name = ticker_to_name.get(stock, stock)
        
        st.markdown(f"""
        <div style="margin-bottom: 24px; display: flex; flex-direction: column; gap: 14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #f3f4f6; padding-bottom:10px;">
                <span style="color:#6b7280; font-size:16px; font-weight:700;">Risk Free Rate</span>
                <span style="color:#111827; font-size:18px; font-weight:800;">{rf*100:.2f}%</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #f3f4f6; padding-bottom:10px;">
                <span style="color:#6b7280; font-size:16px; font-weight:700;">Company</span>
                <span style="color:#111827; font-size:18px; font-weight:800; text-align:right;">{company_name}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #f3f4f6; padding-bottom:10px;">
                <span style="color:#6b7280; font-size:16px; font-weight:700;">Price</span>
                <span style="color:#111827; font-size:18px; font-weight:800;">RM {price:.2f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #f3f4f6; padding-bottom:10px;">
                <span style="color:#6b7280; font-size:16px; font-weight:700;">Changing Stock (Lots)</span>
                <span style="color:#111827; font-size:18px; font-weight:800;">{delta_lot}</span>
            </div>
                        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #f3f4f6; padding-bottom:10px;">
                <span style="color:#6b7280; font-size:16px; font-weight:700;">Remain</span>
                <span style="color:#111827; font-size:18px; font-weight:800;">{remain}</span>
            </div>

        </div>
        """, unsafe_allow_html=True)
        st.pyplot(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="section-title">Top 15 Stocks Weightage</div>', unsafe_allow_html=True)
        st.write("")
        fig = px.bar(w_top15, color_discrete_sequence=["#3b82f6"])
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Weightage", margin=dict(t=10, b=20, l=0, r=0), xaxis=dict(tickangle=90), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        fig.update_yaxes(gridcolor="#e5e7eb")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card"><div class="section-title">2026 Portfolio\'s Performance</div>', unsafe_allow_html=True)
    st.write("")
    col3, col4 = st.columns([1.2, 1])
    with col3: 
        fig1, ax1 = plt.subplots(figsize=(8, 4.4))
        ax1.plot(df.index, df["Sharpe Portfolio"], linewidth=2.4, label="Sharpe Portfolio", color="#2563eb")
        ax1.plot(df.index, df["Equal Weight Portfolio"], linewidth=2.4, label="Equal Weight Portfolio", color="#f59e0b")
        ax1.tick_params(axis='x', rotation=0)
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax1.set_xlabel("")
        ax1.set_ylabel("Portfolio Value")
        ax1.grid(alpha=0.25)
        ax1.legend()
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        
    with col4:
        df_reset = performance.reset_index()
        df_reset.rename(columns={'index': 'Metric'}, inplace=True)
        st.dataframe(df_reset, hide_index=True, use_container_width=True, height=250)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PAGE 3: TASK 3 - Risk & Stress Testing
# =========================================================
elif page == "Task 3: Risk & Stress Testing":

    def calculate_max_drawdown(portfolio_paths):
        cumulative_max = np.maximum.accumulate(portfolio_paths, axis=1)
        drawdown = (portfolio_paths - cumulative_max) / cumulative_max
        return -np.min(drawdown, axis=1)

    def calculate_metrics(returns, portfolio_paths):
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
        avg_mdd = np.mean(calculate_max_drawdown(portfolio_paths))
        return {"Return": mean_ret, "Volatility": std_ret, "VaR_95": var_95, "CVaR_95": cvar_95, "Avg_Max_Drawdown": avg_mdd}

    def run_portfolio_simulation(mu_input, sigma_input, L, weights, n_sims=1000, T=1.0, S0=100):
        dt = 1/252; n_steps = int(T/dt); n_stocks = len(mu_input)
        Z = np.matmul(np.random.randn(n_sims, n_steps, n_stocks), L.T)
        drift = (mu_input - 0.5 * sigma_input**2) * dt
        diffusion = sigma_input * np.sqrt(dt)
        prices = np.zeros((n_sims, n_steps + 1, n_stocks))
        prices[:, 0, :] = S0
        for t in range(1, n_steps + 1):
            prices[:, t, :] = prices[:, t-1, :] * np.exp(drift + diffusion * Z[:, t-1, :])
        portfolio_values = np.sum(prices * weights[np.newaxis, np.newaxis, :], axis=2)
        returns = (portfolio_values[:, -1] / S0) - 1
        return returns, portfolio_values, prices

    def analyze_rebalancing(prices, target_weights, threshold=0.10):
        w_exp = target_weights[np.newaxis, np.newaxis, :]
        asset_val = prices * w_exp
        port_val = np.sum(asset_val, axis=2)
        curr_w = asset_val / port_val[:, :, np.newaxis]
        dev = np.abs(curr_w - w_exp)
        trigger = np.any(dev > threshold, axis=2)
        freq = np.mean(np.sum(trigger[:, 1:], axis=1))
        
        fp_count, checked = 0, 0
        for sim in range(prices.shape[0]):
            days_triggered = np.where(trigger[sim, 1:])[0]
            for d in days_triggered:
                if d < prices.shape[1] - 5:
                    checked += 1
                    if np.any(np.all(dev[sim, d+1:d+6] < threshold, axis=1)):
                        fp_count += 1
        fp_rate = fp_count / checked if checked > 0 else 0.0
        return freq, fp_rate

    @st.cache_data(ttl=3600)
    def load_and_run_simulations():
        files = ['GBM_Parameters_All30.csv', 'cholesky_matrix.csv', 
                 'Optimal_Weights(Sharpe).csv', 'rolling_regression_top30_with_company_info.csv']
        for f in files:
            if not Path(f).exists():
                st.error(f"❌ Missing file: {f}. Please run previous steps first.")
                st.stop()

        params = pd.read_csv('GBM_Parameters_All30.csv', index_col=0)
        cholesky = pd.read_csv('cholesky_matrix.csv', index_col=0)
        weights_df = pd.read_csv('Optimal_Weights(Sharpe).csv', index_col=0)
        info_df = pd.read_csv('rolling_regression_top30_with_company_info.csv')

        tickers = params.index.sort_values()
        mu_base = params.loc[tickers, 'Mu_Annual'].values
        sigma_base = params.loc[tickers, 'Sigma_Annual'].values
        L = cholesky.loc[tickers, tickers].values
        w = weights_df.loc[tickers, 'Optimized_Weights'].values

        ret_b, paths_b, prices_b = run_portfolio_simulation(mu_base, sigma_base, L, w)
        metrics_b = calculate_metrics(ret_b, paths_b)
        freq_b, fp_b = analyze_rebalancing(prices_b, w)
        
        results = [{"Scenario": "Base Case", "Type": "Normal Market", **metrics_b, 
                    "Rebal_Freq": freq_b, "FP_Rate": fp_b, "Est_Cost": freq_b*0.003}]
        
        sector_dict = info_df.groupby('Sector')['Ticker'].apply(list).to_dict()
        for sector, t_list in sector_dict.items():
            valid = [t for t in t_list if t in tickers]
            if not valid: continue
            mu_s, sigma_s = mu_base.copy(), sigma_base.copy()
            idx = [list(tickers).index(t) for t in valid]
            mu_s[idx] = -0.20; sigma_s[idx] *= 2.0
            ret_s, paths_s, prices_s = run_portfolio_simulation(mu_s, sigma_s, L, w)
            m_s = calculate_metrics(ret_s, paths_s)
            freq_s, fp_s = analyze_rebalancing(prices_s, w)
            results.append({"Scenario": f"Sector Crash: {sector}", "Type": "Sector Shock", **m_s,
                            "Rebal_Freq": freq_s, "FP_Rate": fp_s, "Est_Cost": freq_s*0.003})
            
        mu_22 = np.full_like(mu_base, -0.15); sigma_22 = sigma_base * 1.8
        ret_22, paths_22, prices_22 = run_portfolio_simulation(mu_22, sigma_22, L, w)
        m_22 = calculate_metrics(ret_22, paths_22)
        freq_22, fp_22 = analyze_rebalancing(prices_22, w)
        results.append({"Scenario": "2022 Volatility Spike", "Type": "Macro Shock", **m_22,
                        "Rebal_Freq": freq_22, "FP_Rate": fp_22, "Est_Cost": freq_22*0.003})

        df_all = pd.DataFrame(results)
        return metrics_b, paths_b, df_all

    with st.spinner("🔄 Running Monte Carlo Simulations & Stress Tests (Approx 10-15s)..."):
        metrics_b, paths_b, df_all = load_and_run_simulations()

    st.markdown('<div class="dashboard-title" style="font-size: 28px; font-weight: 800; color: #111827;">Portfolio Risk & Stress Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle" style="font-size: 13px; color: #6b7280; margin-bottom: 1rem;">Dashboards / Risk & Stress Analysis / 1,000 Simulations / 1 year horizon</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-title">🎯 Base Case Metrics</div>', unsafe_allow_html=True)
    st.write("")
    def build_kpi(title, value, icon, color_class):
        return f"""<div class="small-card" style="height: 100%;"><div class="kpi-wrap"><div class="kpi-left"><div class="kpi-title">{title}</div>
        <div class="kpi-value" style="font-size:22px;">{value}</div></div><div class="kpi-icon {color_class}">{icon}</div></div></div>"""

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.markdown(build_kpi("Expected Return", f"{metrics_b['Return']:.2%}", "↗", "icon-green"), unsafe_allow_html=True)
    with k2: st.markdown(build_kpi("Annual Volatility", f"{metrics_b['Volatility']:.2%}", "!", "icon-red"), unsafe_allow_html=True)
    with k3: st.markdown(build_kpi("95% VaR", f"{metrics_b['VaR_95']:.2%}", "📉", "icon-orange"), unsafe_allow_html=True)
    with k4: st.markdown(build_kpi("95% CVaR", f"{metrics_b['CVaR_95']:.2%}", "📉", "icon-orange"), unsafe_allow_html=True)
    with k5: st.markdown(build_kpi("Avg Max Drawdown", f"{metrics_b['Avg_Max_Drawdown']:.2%}", "↓", "icon-red"), unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card"><div class="section-title">📈 Monte Carlo Portfolio Paths</div>', unsafe_allow_html=True)
    days = np.arange(paths_b.shape[1])
    fig = go.Figure()
    for i in np.random.choice(paths_b.shape[0], 80, replace=False):
        fig.add_trace(go.Scatter(x=days, y=paths_b[i], mode='lines', line=dict(color='gray', width=0.6), showlegend=False, opacity=0.25))
    fig.add_trace(go.Scatter(x=days, y=np.percentile(paths_b, 5, axis=0), mode='lines', name='5% Tail', line=dict(color='#EF553B', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=days, y=np.percentile(paths_b, 50, axis=0), mode='lines', name='Median', line=dict(color='#00CC96', width=3)))
    fig.add_trace(go.Scatter(x=days, y=np.percentile(paths_b, 95, axis=0), mode='lines', name='95% Tail', line=dict(color='#19D3F3', width=2, dash='dash')))
    fig.update_layout(xaxis_title="Trading Days (1 Year)", yaxis_title="Portfolio Value (Init=100)", 
                      hovermode="x unified", height=420, margin=dict(t=10, b=20, l=10, r=10), showlegend=True, legend=dict(orientation="h", y=1.05), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(gridcolor="#e5e7eb")
    fig.update_xaxes(gridcolor="#e5e7eb")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    col_bottom_left, col_bottom_right = st.columns(2)
    
    with col_bottom_left:
        st.markdown('<div class="card"><div class="section-title">⚡ Stress Test Sensitivity</div>', unsafe_allow_html=True)
        df_stress = df_all[['Scenario', 'Return', 'Volatility', 'Avg_Max_Drawdown', 'VaR_95']].copy()
        styled_stress = df_stress.style.format({
            'Return': '{:.2%}', 'Volatility': '{:.2%}', 'Avg_Max_Drawdown': '{:.2%}', 'VaR_95': '{:.2%}'
        })
        st.dataframe(styled_stress, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_bottom_right:
        st.markdown('<div class="card"><div class="section-title">🔄 Rebalancing Signals & Cost</div>', unsafe_allow_html=True)
        df_rebal = df_all[['Scenario', 'Type', 'Rebal_Freq', 'FP_Rate', 'Est_Cost']].copy()
        styled_rebal = df_rebal.style.format({
            'Rebal_Freq': '{:.2f}', 'FP_Rate': '{:.1%}', 'Est_Cost': '{:.2%}'
        })
        st.dataframe(styled_rebal, use_container_width=True, hide_index=True)
        st.markdown('<div class="dashboard-subtitle" style="font-size: 13px; color: #6b7280; margin-bottom: 1rem;">⚙️ Threshold: 10% drift | Est. Cost = Freq × 0.30% per execution</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)