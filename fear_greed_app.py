"""
Fear & Greed Oscillator — 미국 + 한국 완전 자동화
pykrx 없이 FinanceDataReader + yfinance만 사용
"""
import warnings; warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

st.set_page_config(page_title="Fear & Greed Oscillator", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
  .main{padding:.5rem 1rem}
  .block-container{padding-top:1rem;max-width:1100px}
  h1{font-size:1.4rem!important}
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 1) 데이터 수집
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_us_global(start, end):
    import yfinance as yf
    tickers = {"VIX":"^VIX","10Y":"^TNX","5Y":"^FVX","HYG":"HYG","IEF":"IEF"}
    frames = {}
    for name, t in tickers.items():
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if not df.empty:
            frames[name] = df["Close"].squeeze().rename(name)
    if len(frames) < 5:
        st.error("미국 공통 지표 다운로드 실패. 잠시 후 새로고침하세요.")
        st.stop()
    return pd.concat(frames.values(), axis=1).ffill()


@st.cache_data(ttl=3600)
def load_us_index(ticker, start, end):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    return df["Close"].squeeze() if not df.empty else None


@st.cache_data(ttl=3600)
def load_korean_data(start, end):
    """
    pykrx 없이 FinanceDataReader + yfinance 만으로 한국 데이터 수집
    - KOSPI/KOSDAQ : FinanceDataReader
    - VKOSPI       : FinanceDataReader → 실패 시 실현변동성
    - Bond Spread  : yfinance 미국 금리 (글로벌 심리 대용)
    - Put/Call     : VKOSPI 상대값으로 근사
    """
    import FinanceDataReader as fdr
    import yfinance as yf

    result = {}
    warnings_list = []

    # ── KOSPI / KOSDAQ ─────────────────────────────────────
    for name, ticker in [("KOSPI","KS11"), ("KOSDAQ","KQ11")]:
        try:
            df = fdr.DataReader(ticker, start, end)
            col = "Close" if "Close" in df.columns else df.columns[0]
            result[name] = df[col].rename(name)
        except Exception as e:
            warnings_list.append(f"{name} 로드 실패: {e}")

    # ── VKOSPI ──────────────────────────────────────────────
    vkospi_ok = False
    for ticker in ["VKOSPI", "KRX:VKOSPI"]:
        try:
            df = fdr.DataReader(ticker, start, end)
            col = "Close" if "Close" in df.columns else df.columns[0]
            result["VKOSPI"] = df[col].rename("VKOSPI")
            vkospi_ok = True
            break
        except Exception:
            pass

    if not vkospi_ok and "KOSPI" in result:
        rv = result["KOSPI"].pct_change().rolling(20).std() * np.sqrt(252) * 100
        result["VKOSPI"] = rv.rename("VKOSPI")
        warnings_list.append("VKOSPI: 실현변동성(20일)으로 대체")

    # ── Bond Spread: yfinance 미국 금리 ─────────────────────
    try:
        t10 = yf.download("^TNX", start=start, end=end, auto_adjust=True,
                          progress=False)["Close"].squeeze()
        t5  = yf.download("^FVX", start=start, end=end, auto_adjust=True,
                          progress=False)["Close"].squeeze()
        result["BOND_10Y"] = t10.rename("BOND_10Y")
        result["BOND_5Y"]  = t5.rename("BOND_5Y")
    except Exception as e:
        warnings_list.append(f"Bond 금리 로드 실패: {e}")

    # ── Put/Call 근사: VKOSPI 상대값 ─────────────────────────
    if "VKOSPI" in result:
        ma60 = result["VKOSPI"].rolling(60, min_periods=20).mean()
        result["PCR"] = (result["VKOSPI"] / ma60.replace(0, np.nan)).rename("PCR")
        warnings_list.append("Put/Call: VKOSPI 60일 상대값으로 근사")

    result["_warnings"] = warnings_list
    return result


# ══════════════════════════════════════════════════════════════
# 2) 오실레이터 계산
# ══════════════════════════════════════════════════════════════

def calc_rsi(s, w=10):
    d = s.diff()
    g = d.clip(lower=0).rolling(w).mean()
    l = (-d.clip(upper=0)).rolling(w).mean()
    return 100 - (100 / (1 + g / l))


def calc_macd_hist(s, fast=12, slow=26, sig=9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    m  = ef - es
    return m - m.ewm(span=sig, adjust=False).mean()


def calc_us_oscillator(index_s, global_df):
    df = global_df.copy()
    df["IDX"]           = index_s
    df["Risk_Appetite"] = df["HYG"] / df["IEF"]
    df["MA125"]         = df["IDX"].rolling(125).mean()
    df["Momentum"]      = (df["IDX"] - df["MA125"]) / df["MA125"] * 100
    df["RSI"]           = calc_rsi(df["IDX"])
    df["BondSpread"]    = df["10Y"] - df["5Y"]
    sc   = MinMaxScaler()
    cols = ["Momentum","RSI","BondSpread","VIX","Risk_Appetite"]
    valid = df.dropna(subset=cols).index
    df.loc[valid, cols] = sc.fit_transform(df.loc[valid, cols])
    df["FGI"] = (df["Momentum"]*0.2 + df["Risk_Appetite"]*0.2 +
                 (1-df["VIX"])*0.2  + df["BondSpread"]*0.2 + df["RSI"]*0.2)
    df["Oscillator"] = calc_macd_hist(df["FGI"])
    return df.dropna(subset=["Oscillator"])


def calc_kr_oscillator(index_s, vkospi, bond_10y, bond_5y, pcr):
    """
    원본 스크립트와 동일한 공식:
    FGI = Momentum*0.2 + (1-PutCall)*0.2 + (1-Volatility)*0.2 + BondDiff*0.2 + RSI*0.2
    """
    df = pd.DataFrame({
        "IDX":      index_s,
        "VKOSPI":   vkospi,
        "BOND_10Y": bond_10y,
        "BOND_5Y":  bond_5y,
        "PCR":      pcr,
    }).ffill().dropna()

    df["MA125"]    = df["IDX"].rolling(125).mean()
    df["Momentum"] = (df["IDX"] - df["MA125"]) / df["MA125"] * 100
    df["RSI"]      = calc_rsi(df["IDX"])
    df["BondDiff"] = df["BOND_10Y"] - df["BOND_5Y"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    sc   = MinMaxScaler()
    cols = ["Momentum","PCR","VKOSPI","BondDiff","RSI"]
    valid = df.dropna(subset=cols).index
    if len(valid) == 0:
        return df
    df.loc[valid, cols] = sc.fit_transform(df.loc[valid, cols])
    df.loc[valid, "FGI"] = (
        df.loc[valid,"Momentum"]         * 0.2 +
        (1 - df.loc[valid,"PCR"])        * 0.2 +
        (1 - df.loc[valid,"VKOSPI"])     * 0.2 +
        df.loc[valid,"BondDiff"]         * 0.2 +
        df.loc[valid,"RSI"]              * 0.2
    )
    df["Oscillator"] = calc_macd_hist(df["FGI"])
    return df.dropna(subset=["Oscillator"])


# ══════════════════════════════════════════════════════════════
# 3) 시각화
# ══════════════════════════════════════════════════════════════

def signal(osc):
    if osc >  0.015: return "Extreme Greed", "#e05252"
    if osc >  0.005: return "Greed",         "#e8a04b"
    if osc > -0.005: return "Neutral",       "#888888"
    if osc > -0.015: return "Fear",          "#5b8fe8"
    return                   "Extreme Fear", "#9c36b5"


def make_chart(df, idx_label, idx_color, months):
    cut = df.index.max() - pd.DateOffset(months=months)
    v   = df[df.index >= cut]
    OSC_COL = "#6a5acd"
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=v.index, y=v["Oscillator"], name="Oscillator",
        line=dict(color=OSC_COL, width=2),
        hovertemplate="%{x|%Y-%m-%d}<br>Osc: %{y:.4f}<extra></extra>"),
        secondary_y=False)
    fig.add_trace(go.Scatter(x=v.index, y=v["IDX"], name=idx_label,
        line=dict(color=idx_color, width=1.8),
        hovertemplate=f"%{{x|%Y-%m-%d}}<br>{idx_label}: %{{y:,.2f}}<extra></extra>"),
        secondary_y=True)
    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(200,200,200,0.35)", line_width=1)
    fig.update_layout(hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=370, margin=dict(l=0,r=0,t=35,b=20),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,35,0.9)")
    fig.update_yaxes(title_text="Oscillator",
        tickfont=dict(color=OSC_COL), title_font=dict(color=OSC_COL),
        gridcolor="rgba(80,80,100,0.3)", secondary_y=False)
    fig.update_yaxes(title_text=idx_label,
        tickfont=dict(color=idx_color), title_font=dict(color=idx_color),
        gridcolor="rgba(0,0,0,0)", secondary_y=True)
    fig.update_xaxes(gridcolor="rgba(80,80,100,0.2)")
    return fig


# ══════════════════════════════════════════════════════════════
# 4) 메인
# ══════════════════════════════════════════════════════════════

st.title("📊 Fear & Greed Oscillator")

with st.sidebar:
    st.header("⚙️ 설정")
    months  = st.slider("기간 (개월)", 1, 24, 6)
    markets = st.multiselect("시장 선택",
        ["🇺🇸 NASDAQ","🇺🇸 S&P500","🇰🇷 KOSPI","🇰🇷 KOSDAQ"],
        default=["🇰🇷 KOSPI","🇰🇷 KOSDAQ"])
    st.markdown("---")
    st.caption(
        "데이터: FinanceDataReader + yfinance\n\n"
        "한국 공식: 125MA + (1-PCR) + (1-VKOSPI) + Bond Spread + RSI\n\n"
        "접속 시 자동 갱신 (캐시 1시간)"
    )

if not markets:
    st.warning("사이드바에서 시장을 선택해 주세요.")
    st.stop()

end_dt   = datetime.today()
start_dt = end_dt - timedelta(days=365*3)
start_s, end_s = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

MCONF = {
    "🇺🇸 NASDAQ": ("^IXIC","#2ea84c","NASDAQ","us"),
    "🇺🇸 S&P500":  ("^GSPC","#e8704b","S&P500","us"),
    "🇰🇷 KOSPI":   ("KS11", "#e8b04b","KOSPI", "kr"),
    "🇰🇷 KOSDAQ":  ("KQ11", "#a04be8","KOSDAQ","kr"),
}

us_selected = any(MCONF[m][3]=="us" for m in markets)
kr_selected = any(MCONF[m][3]=="kr" for m in markets)

us_global, kr_data = None, None

if us_selected:
    with st.spinner("미국 지표 로딩 중..."):
        us_global = load_us_global(start_s, end_s)

if kr_selected:
    with st.spinner("한국 데이터 수집 중..."):
        kr_data = load_korean_data(start_s, end_s)
    for w in kr_data.get("_warnings", []):
        st.info(f"ℹ️ {w}")

for mkt in markets:
    ticker, color, label, region = MCONF[mkt]
    st.markdown(f"---\n### {mkt}")

    try:
        if region == "us":
            with st.spinner(f"{label} 계산 중..."):
                idx_s = load_us_index(ticker, start_s, end_s)
                if idx_s is None:
                    st.error(f"❌ {label} 데이터 없음"); continue
                df = calc_us_oscillator(idx_s, us_global)
        else:
            with st.spinner(f"{label} 계산 중..."):
                if label not in kr_data:
                    st.error(f"❌ {label} 지수 데이터 없음"); continue
                df = calc_kr_oscillator(
                    kr_data[label],
                    kr_data.get("VKOSPI",  pd.Series(dtype=float)),
                    kr_data.get("BOND_10Y",pd.Series(dtype=float)),
                    kr_data.get("BOND_5Y", pd.Series(dtype=float)),
                    kr_data.get("PCR",     pd.Series(dtype=float)),
                )

        if df.empty:
            st.error(f"❌ {label} 계산 결과 없음"); continue

        last = df.iloc[-1]
        osc  = float(last["Oscillator"])
        sig_t, sig_c = signal(osc)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric(label,        f"{float(last['IDX']):,.2f}")
        c2.metric("Oscillator", f"{osc:.4f}")
        c3.metric("날짜",       df.index[-1].strftime("%Y-%m-%d"))
        c4.markdown(f"<div style='padding:8px 0;font-size:18px;"
                    f"font-weight:500;color:{sig_c}'>{sig_t}</div>",
                    unsafe_allow_html=True)

        st.plotly_chart(make_chart(df, label, color, months),
                        use_container_width=True)

    except Exception as e:
        st.error(f"❌ {label} 오류: {e}")

with st.expander("📖 오실레이터 해석 가이드"):
    st.markdown("""
| 범위 | 신호 | 의미 |
|---|---|---|
| +0.015 이상 | 🔴 Extreme Greed | 심리 과열 — 차익 실현 고려 |
| 0 ~ +0.015  | 🟠 Greed | 심리 개선, 모멘텀 유지 |
| ±0.005      | ⚪ Neutral | 전환 구간 |
| -0.015 ~ 0  | 🔵 Fear | 심리 악화, 주의 |
| -0.015 이하 | 🟣 Extreme Fear | 공포 극단 → 역발상 매수 검토 |
    """)

st.caption(f"업데이트: {datetime.today().strftime('%Y-%m-%d %H:%M')} | "
           f"FinanceDataReader + yfinance")
