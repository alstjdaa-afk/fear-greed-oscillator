"""
Fear & Greed Oscillator — 미국 + 한국 완전 자동화
US : NASDAQ / S&P500  (Yahoo Finance)
KR : KOSPI / KOSDAQ   (FinanceDataReader + pykrx)
"""

import warnings; warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# ── 페이지 설정 ──────────────────────────────────────────────
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
def load_us_global(start: str, end: str) -> pd.DataFrame:
    """미국 공통 지표: VIX, 금리, HYG/IEF"""
    import yfinance as yf
    tickers = {"VIX":"^VIX","10Y":"^TNX","5Y":"^FVX","HYG":"HYG","IEF":"IEF"}
    frames = {}
    for name, t in tickers.items():
        df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if not df.empty:
            frames[name] = df["Close"].rename(name)
    if len(frames) < 5:
        st.error("미국 공통 지표 다운로드 실패. 잠시 후 새로고침하세요.")
        st.stop()
    return pd.concat(frames.values(), axis=1).ffill()


@st.cache_data(ttl=3600)
def load_us_index(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    return df["Close"].squeeze() if not df.empty else None


@st.cache_data(ttl=3600)
def load_korean_data(start: str, end: str) -> dict:
    """
    한국 시장 자동 수집
    - 지수: FinanceDataReader
    - VKOSPI: pykrx stock
    - 국채금리: pykrx bond
    - Put/Call: pykrx derivatives → 전체 PCR 비율
    """
    import FinanceDataReader as fdr
    from pykrx import stock as pkstock, bond as pkbond

    s_krx = pd.Timestamp(start).strftime('%Y%m%d')
    e_krx = pd.Timestamp(end).strftime('%Y%m%d')

    result = {}
    errors = []

    # ── 1) KOSPI / KOSDAQ ──────────────────────────────────────
    for name, ticker in [('KOSPI', 'KS11'), ('KOSDAQ', 'KQ11')]:
        try:
            df = fdr.DataReader(ticker, start, end)
            result[name] = df['Close'].rename(name)
        except Exception as e:
            errors.append(f"{name}: {e}")

    # ── 2) VKOSPI ─────────────────────────────────────────────
    # pykrx VKOSPI 코드: KRX 공식 코드 'VKOSPI' 또는 숫자코드
    vkospi_loaded = False
    for code in ['VKOSPI', '147']:
        try:
            df = pkstock.get_index_ohlcv(s_krx, e_krx, code)
            if not df.empty and '종가' in df.columns:
                result['VKOSPI'] = df['종가'].rename('VKOSPI')
                vkospi_loaded = True
                break
        except Exception:
            pass

    if not vkospi_loaded:
        # Fallback: 실현 변동성으로 근사 (KOSPI 20일 롤링 변동성 × 100)
        if 'KOSPI' in result:
            rv = result['KOSPI'].pct_change().rolling(20).std() * np.sqrt(252) * 100
            result['VKOSPI'] = rv.rename('VKOSPI')
            errors.append("VKOSPI: pykrx 실패 → 실현변동성으로 대체")

    # ── 3) 국채 금리 (pykrx bond) ──────────────────────────────
    bond_loaded = False
    for issuer in ['국고채', '국채']:
        try:
            df = pkbond.get_otc_treasury_yields(s_krx, e_krx, issuer)
            if not df.empty:
                y5  = df['5년'] if '5년' in df.columns else df.iloc[:, 2]
                y10 = df['10년'] if '10년' in df.columns else df.iloc[:, 4]
                result['KR_5Y']  = y5.rename('KR_5Y')
                result['KR_10Y'] = y10.rename('KR_10Y')
                bond_loaded = True
                break
        except Exception:
            pass

    if not bond_loaded:
        # Fallback: yfinance 미국 금리 대용 (방향성 동일)
        try:
            import yfinance as yf
            for yk, rk in [('^FVX', 'KR_5Y'), ('^TNX', 'KR_10Y')]:
                df = yf.download(yk, start=start, end=end, auto_adjust=True, progress=False)
                if not df.empty:
                    result[rk] = df['Close'].squeeze().rename(rk)
            errors.append("한국 국채금리: pykrx 실패 → 미국 금리 대용")
        except Exception:
            pass

    # ── 4) Put/Call 비율 ───────────────────────────────────────
    # KOSPI200 옵션 총 거래량 PUT/CALL 비율
    pcr_loaded = False
    try:
        from pykrx import derivatives
        # KOSPI200 파생상품 전체 PUT/CALL 거래량
        call_vol_total = 0
        put_vol_total  = 0
        rows = []
        # 날짜 범위 내 각 영업일에 대해 계산 (최근 데이터만 가져와 효율화)
        dates = pd.bdate_range(start, end)
        for dt in dates[-60:]:  # 최근 60 영업일만
            d = dt.strftime('%Y%m%d')
            try:
                call_df = derivatives.get_derivatives_ohlcv_by_date(d, d, 'C')
                put_df  = derivatives.get_derivatives_ohlcv_by_date(d, d, 'P')
                c_vol = call_df['거래량'].sum() if '거래량' in call_df.columns else 0
                p_vol = put_df['거래량'].sum()  if '거래량' in put_df.columns  else 0
                if c_vol > 0:
                    rows.append({'date': dt, 'pcr': p_vol / c_vol})
            except Exception:
                pass

        if rows:
            pcr_series = pd.DataFrame(rows).set_index('date')['pcr']
            result['PCR'] = pcr_series.rename('PCR')
            pcr_loaded = True
    except Exception:
        pass

    if not pcr_loaded:
        # Fallback: VKOSPI 기반 대리 Put/Call (변동성 높을수록 PUT 우세)
        if 'VKOSPI' in result:
            # VKOSPI가 높으면 PUT이 콜보다 비쌈 → PCR 높음을 모사
            result['PCR'] = (result['VKOSPI'] / result['VKOSPI'].rolling(60).mean()).rename('PCR')
            errors.append("Put/Call: pykrx 실패 → VKOSPI 기반 대리값 사용")

    if errors:
        result['_warnings'] = errors

    return result


# ══════════════════════════════════════════════════════════════
# 2) 오실레이터 계산
# ══════════════════════════════════════════════════════════════

def calc_rsi(s: pd.Series, w=10) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(w).mean()
    l = (-d.clip(upper=0)).rolling(w).mean()
    return 100 - (100 / (1 + g / l))


def calc_macd_hist(s: pd.Series, fast=12, slow=26, sig=9) -> pd.Series:
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    m  = ef - es
    return m - m.ewm(span=sig, adjust=False).mean()


def calc_us_oscillator(index_s: pd.Series,
                       global_df: pd.DataFrame) -> pd.DataFrame:
    """미국 버전: HYG/IEF + US VIX + US Bond Spread"""
    df = global_df.copy()
    df['IDX']           = index_s
    df['Risk_Appetite'] = df['HYG'] / df['IEF']
    df['MA125']         = df['IDX'].rolling(125).mean()
    df['Momentum']      = (df['IDX'] - df['MA125']) / df['MA125'] * 100
    df['RSI']           = calc_rsi(df['IDX'])
    df['BondSpread']    = df['10Y'] - df['5Y']

    sc   = MinMaxScaler()
    cols = ['Momentum', 'RSI', 'BondSpread', 'VIX', 'Risk_Appetite']
    valid = df.dropna(subset=cols).index
    df.loc[valid, cols] = sc.fit_transform(df.loc[valid, cols])

    df['FGI'] = (df['Momentum']      * 0.2 +
                 df['Risk_Appetite'] * 0.2 +
                 (1 - df['VIX'])     * 0.2 +
                 df['BondSpread']    * 0.2 +
                 df['RSI']           * 0.2)
    df['Oscillator'] = calc_macd_hist(df['FGI'])
    return df.dropna(subset=['Oscillator'])


def calc_kr_oscillator(index_s: pd.Series,
                       vkospi:  pd.Series,
                       kr_5y:   pd.Series,
                       kr_10y:  pd.Series,
                       pcr:     pd.Series) -> pd.DataFrame:
    """
    한국 버전: 업로드 스크립트와 동일한 공식
    FGI = Momentum + (1-PutCall) + (1-Volatility) + BondDiff + RSI  (각 0.2)
    """
    df = pd.DataFrame({
        'IDX':    index_s,
        'VKOSPI': vkospi,
        'KR_5Y':  kr_5y,
        'KR_10Y': kr_10y,
        'PCR':    pcr
    }).dropna()

    df['MA125']    = df['IDX'].rolling(125).mean()
    df['Momentum'] = (df['IDX'] - df['MA125']) / df['MA125'] * 100
    df['RSI']      = calc_rsi(df['IDX'])
    df['BondDiff'] = df['KR_10Y'] - df['KR_5Y']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    sc   = MinMaxScaler()
    cols = ['Momentum', 'PCR', 'VKOSPI', 'BondDiff', 'RSI']
    valid = df.dropna(subset=cols).index
    if len(valid) == 0:
        return df
    df.loc[valid, cols] = sc.fit_transform(df.loc[valid, cols])

    df.loc[valid, 'FGI'] = (
        df.loc[valid, 'Momentum'] * 0.2 +
        (1 - df.loc[valid, 'PCR'])    * 0.2 +   # 원본: (1 - PutCall)
        (1 - df.loc[valid, 'VKOSPI']) * 0.2 +   # 원본: (1 - Volatility)
        df.loc[valid, 'BondDiff']     * 0.2 +
        df.loc[valid, 'RSI']          * 0.2
    )
    df['Oscillator'] = calc_macd_hist(df['FGI'])
    return df.dropna(subset=['Oscillator'])


# ══════════════════════════════════════════════════════════════
# 3) 시각화 헬퍼
# ══════════════════════════════════════════════════════════════

OSC_COLOR  = '#6a5acd'
GREED_COLS = {
    'Extreme Greed': '#e05252',
    'Greed':         '#e8a04b',
    'Neutral':       '#888888',
    'Fear':          '#5b8fe8',
    'Extreme Fear':  '#9c36b5',
}

def signal(osc):
    if osc >  0.015: return 'Extreme Greed', '#e05252'
    if osc >  0.005: return 'Greed',         '#e8a04b'
    if osc > -0.005: return 'Neutral',       '#888888'
    if osc > -0.015: return 'Fear',          '#5b8fe8'
    return                   'Extreme Fear', '#9c36b5'


def make_chart(df, idx_label, idx_color, months):
    cut = df.index.max() - pd.DateOffset(months=months)
    v   = df[df.index >= cut]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=v.index, y=v['Oscillator'], name='Oscillator',
                             line=dict(color=OSC_COLOR, width=2),
                             hovertemplate='%{x|%Y-%m-%d}<br>Osc: %{y:.4f}<extra></extra>'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=v.index, y=v['IDX'], name=idx_label,
                             line=dict(color=idx_color, width=1.8),
                             hovertemplate=f'%{{x|%Y-%m-%d}}<br>{idx_label}: %{{y:,.2f}}<extra></extra>'),
                  secondary_y=True)
    fig.add_hline(y=0, line_dash='dot', line_color='rgba(200,200,200,0.35)', line_width=1)
    fig.update_layout(hovermode='x unified',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02),
                      height=370, margin=dict(l=0, r=0, t=35, b=20),
                      template='plotly_dark',
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(20,20,35,0.9)')
    fig.update_yaxes(title_text='Oscillator', tickfont=dict(color=OSC_COLOR),
                     title_font=dict(color=OSC_COLOR),
                     gridcolor='rgba(80,80,100,0.3)', secondary_y=False)
    fig.update_yaxes(title_text=idx_label, tickfont=dict(color=idx_color),
                     title_font=dict(color=idx_color),
                     gridcolor='rgba(0,0,0,0)', secondary_y=True)
    fig.update_xaxes(gridcolor='rgba(80,80,100,0.2)')
    return fig


def show_metrics(df, label):
    last = df.iloc[-1]
    osc  = float(last['Oscillator'])
    idx  = float(last['IDX'])
    dt   = df.index[-1].strftime('%Y-%m-%d')
    st_t, st_c = signal(osc)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(label,        f"{idx:,.2f}")
    c2.metric('Oscillator', f"{osc:.4f}")
    c3.metric('날짜',       dt)
    c4.markdown(f"<div style='padding:8px 0;font-size:18px;font-weight:500;"
                f"color:{st_c}'>{st_t}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 4) 메인
# ══════════════════════════════════════════════════════════════

st.title("📊 Fear & Greed Oscillator")

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    months  = st.slider("기간 (개월)", 1, 24, 6)
    markets = st.multiselect(
        "시장 선택",
        ["🇺🇸 NASDAQ", "🇺🇸 S&P500", "🇰🇷 KOSPI", "🇰🇷 KOSDAQ"],
        default=["🇰🇷 KOSPI", "🇰🇷 KOSDAQ"]
    )
    st.markdown("---")
    st.caption(
        "🇺🇸 미국: Yahoo Finance\n\n"
        "🇰🇷 한국: FinanceDataReader + pykrx\n\n"
        "알고리즘 (한국): 125MA Momentum + (1-Put/Call) + (1-VKOSPI) + 국채금리 스프레드 + RSI(10) → MACD(12,26,9)\n\n"
        "접속 시마다 최신 데이터 자동 갱신 (캐시 1시간)"
    )

if not markets:
    st.warning("사이드바에서 시장을 선택해 주세요.")
    st.stop()

end_dt   = datetime.today()
start_dt = end_dt - timedelta(days=365 * 3)
start_s, end_s = start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')

MCONF = {
    "🇺🇸 NASDAQ": ("^IXIC", "#2ea84c", "NASDAQ", "us"),
    "🇺🇸 S&P500":  ("^GSPC", "#e8704b", "S&P500", "us"),
    "🇰🇷 KOSPI":   ("KS11",  "#e8b04b", "KOSPI",  "kr"),
    "🇰🇷 KOSDAQ":  ("KQ11",  "#a04be8", "KOSDAQ", "kr"),
}

# ── 미국 글로벌 데이터 (선택된 경우만) ──────────────────────
us_selected = any(MCONF[m][3] == 'us' for m in markets)
kr_selected = any(MCONF[m][3] == 'kr' for m in markets)

us_global = None
kr_data   = None

if us_selected:
    with st.spinner("미국 공통 지표 로딩 중..."):
        us_global = load_us_global(start_s, end_s)

if kr_selected:
    with st.spinner("한국 시장 데이터 수집 중... (pykrx + FinanceDataReader)"):
        kr_data = load_korean_data(start_s, end_s)
    if '_warnings' in kr_data:
        for w in kr_data['_warnings']:
            st.warning(f"⚠️ {w}")

# ── 각 시장 렌더링 ───────────────────────────────────────────
for mkt in markets:
    ticker, color, label, region = MCONF[mkt]
    st.markdown(f"---\n### {mkt}")

    try:
        if region == 'us':
            with st.spinner(f"{label} 계산 중..."):
                idx_s = load_us_index(ticker, start_s, end_s)
                if idx_s is None:
                    st.error(f"❌ {label} 지수 데이터 없음"); continue
                df = calc_us_oscillator(idx_s.squeeze(), us_global)

        else:  # kr
            with st.spinner(f"{label} 계산 중..."):
                if label not in kr_data:
                    st.error(f"❌ {label} 지수 데이터 없음"); continue
                idx_s   = kr_data[label]
                vkospi  = kr_data.get('VKOSPI', pd.Series(dtype=float))
                kr_5y   = kr_data.get('KR_5Y',  pd.Series(dtype=float))
                kr_10y  = kr_data.get('KR_10Y', pd.Series(dtype=float))
                pcr     = kr_data.get('PCR',    pd.Series(dtype=float))
                df = calc_kr_oscillator(idx_s, vkospi, kr_5y, kr_10y, pcr)

        if df.empty:
            st.error(f"❌ {label} 계산 결과 없음"); continue

        show_metrics(df, label)
        st.plotly_chart(make_chart(df, label, color, months), use_container_width=True)

        with st.expander(f"📌 {label} 구성요소"):
            last  = df.iloc[-1]
            items = [("Momentum", float(last.get('Momentum', 0))),
                     ("RSI(10)",  float(last.get('RSI', 0))),
                     ("Bond Diff",float(last.get('BondDiff', last.get('BondSpread', 0)))),
                     ("변동성(역)", 1 - float(last.get('VKOSPI', last.get('VIX', 0)))),
                     ("Put/Call 역" if region=='kr' else "Risk Appetite",
                      1 - float(last.get('PCR', 0)) if region=='kr'
                        else float(last.get('Risk_Appetite', 0)))]
            cols = st.columns(5)
            for col, (name, val) in zip(cols, items):
                val = max(0., min(1., val))
                bar = "█" * int(val*10) + "░" * (10 - int(val*10))
                clr = "#2ea84c" if val > 0.5 else "#6a5acd"
                col.markdown(
                    f"<div style='text-align:center;font-size:11px;color:#aaa'>{name}</div>"
                    f"<div style='text-align:center;font-size:18px;font-weight:500;color:{clr}'>{val:.2f}</div>"
                    f"<div style='text-align:center;font-size:10px;color:#555'>{bar}</div>",
                    unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ {label} 처리 오류: {e}")

# ── 해석 가이드 ──────────────────────────────────────────────
with st.expander("📖 오실레이터 해석 가이드"):
    st.markdown("""
| 범위 | 신호 | 의미 |
|---|---|---|
| +0.015 이상 | 🔴 Extreme Greed | 심리 과열 — 차익 실현 고려 |
| 0 ~ +0.015  | 🟠 Greed | 심리 개선, 모멘텀 유지 |
| ±0.005      | ⚪ Neutral | 전환 구간, 방향 확인 필요 |
| -0.015 ~ 0  | 🔵 Fear | 심리 악화, 주의 |
| -0.015 이하 | 🟣 Extreme Fear | 공포 극단 → 역발상 매수 검토 |

**한국 버전 공식:** 업로드 스크립트와 동일 — 125MA Momentum + (1-Put/Call) + (1-VKOSPI) + 국채금리스프레드(10Y-5Y) + RSI(10)
    """)

st.caption(f"마지막 업데이트: {datetime.today().strftime('%Y-%m-%d %H:%M')} | "
           f"접속 시 자동 갱신 | FinanceDataReader + pykrx + yfinance")
