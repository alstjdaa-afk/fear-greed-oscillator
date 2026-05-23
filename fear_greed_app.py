"""
Fear & Greed Oscillator Dashboard
==================================
Streamlit 앱 - 매일 자동 업데이트, 모바일 최적화
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# ── 페이지 설정 ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fear & Greed Oscillator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 모바일 친화적 CSS
st.markdown("""
<style>
    .main { padding: 0.5rem 1rem; }
    .block-container { padding-top: 1rem; max-width: 1000px; }
    h1 { font-size: 1.4rem !important; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 12px 16px;
        margin: 4px;
        text-align: center;
    }
    .metric-label { font-size: 11px; color: #888; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── 데이터 다운로드 (캐시 1시간) ────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    end   = datetime.today()
    start = end - timedelta(days=365 * 3)   # 125일 MA + 여유분

    tickers = {
        "NASDAQ": "^IXIC",
        "S&P500": "^GSPC",
        "VIX":    "^VIX",
        "10Y":    "^TNX",
        "5Y":     "^FVX",
        "HYG":    "HYG",
        "IEF":    "IEF"
    }

    frames = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if not df.empty:
                frames[name] = df["Close"].rename(name)
        except Exception:
            pass

    if len(frames) < 6:
        st.error("❌ 데이터 다운로드 실패. 잠시 후 새로고침해 주세요.")
        st.stop()

    data = pd.concat(frames.values(), axis=1).dropna()
    data["Risk_Appetite"] = data["HYG"] / data["IEF"]
    return data


# ── 지표 계산 함수 ───────────────────────────────────────────
def calc_rsi(series: pd.Series, window: int = 10) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def calc_macd_hist(series: pd.Series,
                   fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_f  = series.ewm(span=fast,   adjust=False).mean()
    ema_s  = series.ewm(span=slow,   adjust=False).mean()
    macd   = ema_f - ema_s
    sig    = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def calc_fear_greed(data: pd.DataFrame, index_col: str, label: str) -> pd.DataFrame:
    df = data.copy()
    df[f"{label}_125MA"]     = df[index_col].rolling(125).mean()
    df[f"{label}_Momentum"]  = (df[index_col] - df[f"{label}_125MA"]) / df[f"{label}_125MA"] * 100
    df[f"{label}_RSI"]       = calc_rsi(df[index_col])
    df[f"{label}_BondSpread"]= df["10Y"] - df["5Y"]
    df[f"{label}_VIX"]       = df["VIX"]
    df[f"{label}_RiskApp"]   = df["Risk_Appetite"]

    scaler = MinMaxScaler()
    cols   = [f"{label}_Momentum", f"{label}_RSI",
              f"{label}_BondSpread", f"{label}_VIX", f"{label}_RiskApp"]
    df[cols] = scaler.fit_transform(df[cols])

    df[f"{label}_FGI"] = (
        df[f"{label}_Momentum"]  * 0.2 +
        df[f"{label}_RiskApp"]   * 0.2 +
        (1 - df[f"{label}_VIX"]) * 0.2 +
        df[f"{label}_BondSpread"]* 0.2 +
        df[f"{label}_RSI"]       * 0.2
    )
    df[f"{label}_Osc"] = calc_macd_hist(df[f"{label}_FGI"])
    return df


def get_signal(osc: float) -> tuple[str, str]:
    if osc > 0.015:   return "🔴 Extreme Greed", "#e05252"
    if osc > 0.005:   return "🟠 Greed",         "#e8a04b"
    if osc > -0.005:  return "⚪ Neutral",        "#888888"
    if osc > -0.015:  return "🔵 Fear",           "#5b8fe8"
    return "🟣 Extreme Fear", "#9c36b5"


# ── 메인 ────────────────────────────────────────────────────
st.title("📊 Fear & Greed Oscillator")

# 데이터 로드 + 계산
with st.spinner("데이터 로딩 중..."):
    raw  = load_data()
    data = calc_fear_greed(raw, "NASDAQ", "NDX")
    data = calc_fear_greed(data, "S&P500", "SPX")

# 최근 6개월
cutoff = data.index.max() - pd.DateOffset(months=6)
recent = data[data.index >= cutoff].dropna(subset=["NDX_Osc", "SPX_Osc"])

last        = recent.iloc[-1]
last_date   = recent.index[-1].strftime("%Y-%m-%d")
last_nasdaq = last["NASDAQ"]
last_osc    = last["NDX_Osc"]
sig_text, sig_color = get_signal(last_osc)

# ── 지표 카드 ────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("NASDAQ", f"{last_nasdaq:,.0f}")
with col2:
    st.metric("Oscillator (NDX)", f"{last_osc:.4f}")
with col3:
    st.metric("S&P500", f"{last['S&P500']:,.0f}")
with col4:
    st.metric("최신 날짜", last_date)

st.markdown(
    f"<div style='text-align:center; font-size:20px; font-weight:600; "
    f"color:{sig_color}; padding:8px 0 4px;'>{sig_text}</div>",
    unsafe_allow_html=True
)

# ── 사이드바 옵션 ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    show_spx = st.checkbox("S&P500 Oscillator 표시", value=False)
    show_sp500 = st.checkbox("S&P500 지수 표시", value=False)
    months = st.slider("기간 (개월)", 1, 24, 6)
    fast_span   = st.selectbox("MACD Fast EMA", [8, 12, 16], index=1)
    slow_span   = st.selectbox("MACD Slow EMA", [21, 26, 30], index=1)
    signal_span = st.selectbox("MACD Signal",   [7,  9, 12],  index=1)
    st.markdown("---")
    st.caption("데이터: Yahoo Finance\n알고리즘: 125MA + RSI(10) + Bond Spread + VIX + Risk Appetite(HYG/IEF) → MACD")

# 기간 재적용
cutoff2 = data.index.max() - pd.DateOffset(months=months)
view    = data[data.index >= cutoff2].dropna(subset=["NDX_Osc"])

# MACD 옵션 변경 시 재계산
if (fast_span, slow_span, signal_span) != (12, 26, 9):
    view["NDX_Osc"] = calc_macd_hist(view["NDX_FGI"], fast_span, slow_span, signal_span)
    view["SPX_Osc"] = calc_macd_hist(view["SPX_FGI"], fast_span, slow_span, signal_span)

# ── 차트 ─────────────────────────────────────────────────────
fig = make_subplots(specs=[[{"secondary_y": True}]])

# 오실레이터 (NDX)
fig.add_trace(go.Scatter(
    x=view.index, y=view["NDX_Osc"],
    name="NDX Oscillator",
    line=dict(color="#4c7efe", width=1.8),
    hovertemplate="%{x|%Y-%m-%d}<br>Osc: %{y:.4f}<extra></extra>"
), secondary_y=False)

# 오실레이터 (SPX) - 옵션
if show_spx:
    fig.add_trace(go.Scatter(
        x=view.index, y=view["SPX_Osc"],
        name="SPX Oscillator",
        line=dict(color="#b05ce6", width=1.2, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>SPX Osc: %{y:.4f}<extra></extra>"
    ), secondary_y=False)

# NASDAQ 지수
fig.add_trace(go.Scatter(
    x=view.index, y=view["NASDAQ"],
    name="NASDAQ",
    line=dict(color="#2ea84c", width=1.8),
    hovertemplate="%{x|%Y-%m-%d}<br>NASDAQ: %{y:,.0f}<extra></extra>"
), secondary_y=True)

# S&P500 지수 - 옵션
if show_sp500:
    fig.add_trace(go.Scatter(
        x=view.index, y=view["S&P500"],
        name="S&P500",
        line=dict(color="#e8704b", width=1.2, dash="dot"),
        hovertemplate="%{x|%Y-%m-%d}<br>S&P500: %{y:,.0f}<extra></extra>"
    ), secondary_y=True)

# 제로라인
fig.add_hline(y=0, line_dash="dot", line_color="rgba(200,200,200,0.4)",
              line_width=1, secondary_y=False)

fig.update_layout(
    title=dict(
        text=f"Fear & Greed Oscillator — Last {months} Months",
        font=dict(size=14)
    ),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
    height=420,
    margin=dict(l=0, r=0, t=50, b=30),
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(20,20,35,0.9)",
)
fig.update_yaxes(title_text="Oscillator", title_font=dict(color="#4c7efe"),
                 tickfont=dict(color="#4c7efe"), secondary_y=False,
                 gridcolor="rgba(80,80,100,0.3)")
fig.update_yaxes(title_text="Index", title_font=dict(color="#2ea84c"),
                 tickfont=dict(color="#2ea84c"), secondary_y=True,
                 gridcolor="rgba(0,0,0,0)")
fig.update_xaxes(gridcolor="rgba(80,80,100,0.2)")

st.plotly_chart(fig, use_container_width=True)

# ── 구성요소 현황 ─────────────────────────────────────────────
st.subheader("📌 구성요소 현황 (최신)")
comp_cols = st.columns(5)
components = [
    ("Momentum\n(125MA)",     last["NDX_Momentum"],  "가격 위치"),
    ("RSI(10)",               last["NDX_RSI"],        "단기 모멘텀"),
    ("Bond Spread\n(10Y-5Y)", last["NDX_BondSpread"], "금리 곡선"),
    ("VIX\n(역방향)",         1 - last["NDX_VIX"],    "공포지수 반전"),
    ("Risk Appetite\n(HYG/IEF)", last["NDX_RiskApp"], "위험 선호도"),
]
for col, (name, val, desc) in zip(comp_cols, components):
    bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
    color = "#2ea84c" if val > 0.5 else "#4c7efe"
    col.markdown(
        f"<div style='text-align:center;font-size:11px;color:#aaa;'>{name}</div>"
        f"<div style='text-align:center;font-size:16px;font-weight:600;color:{color};'>{val:.2f}</div>"
        f"<div style='text-align:center;font-size:10px;color:#666;'>{bar}</div>",
        unsafe_allow_html=True
    )

# ── 해석 가이드 ───────────────────────────────────────────────
with st.expander("📖 오실레이터 해석 가이드"):
    st.markdown("""
    | 범위 | 신호 | 의미 |
    |---|---|---|
    | +0.015 이상 | 🔴 Extreme Greed | 심리 과열, 차익 실현 고려 |
    | 0 ~ +0.015 | 🟠 Greed | 심리 개선 중, 모멘텀 유지 |
    | -0.005 ~ +0.005 | ⚪ Neutral | 전환 구간, 방향 확인 필요 |
    | -0.015 ~ -0.005 | 🔵 Fear | 심리 악화 중, 주의 |
    | -0.015 이하 | 🟣 Extreme Fear | 공포 극단 → 역발상 매수 기회 검토 |

    **오실레이터 = FGI의 MACD Histogram**
    - FGI 구성: 125MA Momentum + RSI(10) + Bond Spread(10Y-5Y) + VIX + Risk Appetite(HYG/IEF)
    - 제로선 상향 돌파 → 심리 회복 신호
    - 가격 신고점 + 오실레이터 하락 → Bearish Divergence (현재 상태)
    """)

st.caption(f"마지막 업데이트: {last_date} | 데이터: Yahoo Finance | 새로고침하면 최신 데이터 반영")
