"""
CAN SLIM 智能選股看板
=====================
William O'Neil 法則 | 台股上市・上櫃・美股
自動篩選前3名個股，追蹤技術指標，21日均線警示
"""

import streamlit as st

st.set_page_config(
    page_title="CAN SLIM 智能選股看板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pytz
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Tuple

warnings.filterwarnings("ignore")

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False


# ================================================================
# CSS 樣式
# ================================================================
st.markdown("""
<style>
  .gradient-title {
    background: linear-gradient(90deg,#00ff88,#00bfff,#bf00ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-size:2.1rem; font-weight:800; margin:0;
  }
  .header-bar {
    background:linear-gradient(135deg,#090f1e,#141e3a,#090f1e);
    border:1px solid #1e3060; border-radius:12px;
    padding:18px 28px; margin-bottom:16px;
  }
  .alert-ema {
    background:linear-gradient(135deg,#1a0035,#2c0055);
    border:2px solid #bf00ff; border-radius:8px;
    padding:10px 16px; margin:6px 0;
  }
  .alert-touch {
    background:linear-gradient(135deg,#2d1800,#4a2800);
    border:2px solid #FFD700; border-radius:8px;
    padding:10px 16px; margin:6px 0;
  }
  .rank-card {
    background:#141928; border:1px solid #1e3060;
    border-radius:10px; padding:10px 14px; margin-bottom:8px;
    text-align:center;
  }
  .stTabs [data-baseweb="tab"] {
    background:#141928; border:1px solid #1e3060;
    border-radius:8px; color:#999; padding:7px 18px;
  }
  .stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#003a1a,#005a28) !important;
    border-color:#00ff88 !important; color:#00ff88 !important;
  }
</style>
""", unsafe_allow_html=True)


# ================================================================
# 股票候選名單
# ================================================================
TWSE_UNIVERSE = [          # 台股上市
    "2330.TW",  # 台積電
    "2454.TW",  # 聯發科
    "2317.TW",  # 鴻海
    "2308.TW",  # 台達電
    "2382.TW",  # 廣達
    "2303.TW",  # 聯電
    "3034.TW",  # 聯詠
    "2379.TW",  # 瑞昱
    "2395.TW",  # 研華
    "3711.TW",  # 日月光投控
    "2357.TW",  # 華碩
    "2376.TW",  # 技嘉
    "2327.TW",  # 國巨
    "3008.TW",  # 大立光
    "2207.TW",  # 和泰車
    "2412.TW",  # 中華電
    "2912.TW",  # 統一超
    "2881.TW",  # 富邦金
    "2882.TW",  # 國泰金
    "2886.TW",  # 兆豐金
    "2474.TW",  # 可成
    "4904.TW",  # 遠傳
    "2385.TW",  # 群光
    "2388.TW",  # 威盛
]

TPEX_UNIVERSE = [          # 台股上櫃
    "5347.TWO",  # 世界先進
    "6669.TWO",  # 緯穎
    "8069.TWO",  # 元太
    "6533.TWO",  # 晶心科
    "4977.TWO",  # 廣錠
    "6278.TWO",  # 台表科
    "5269.TWO",  # 祥碩
    "6719.TWO",  # 力旺
    "3533.TWO",  # 嘉澤
    "6245.TWO",  # 立端
    "3231.TWO",  # 緯創資通
    "3673.TWO",  # TPK
    "3583.TWO",  # 辛耘
    "8299.TWO",  # 群聯
    "3261.TWO",  # 聖暉企業
    "6414.TWO",  # 樺漢
    "3037.TWO",  # 欣興
    "6187.TWO",  # 北極星藥業
]

US_UNIVERSE = [            # 美股
    "NVDA", "AAPL", "MSFT", "META", "GOOGL",
    "AMZN", "TSLA", "AMD",  "AVGO", "SMCI",
    "ARM",  "CRWD", "PANW", "PLTR", "CELH",
    "MELI", "SHOP", "UBER", "COIN", "TSM",
    "ASML", "LRCX", "KLAC", "AMAT", "MRVL",
    "ORCL", "CRM",  "NOW",  "ADBE", "DKNG",
]

BENCHMARKS = {"TWSE": "^TWII", "TPEX": "^TWII", "US": "^GSPC"}
EMA21_THRESHOLD = 0.015
TOP_N = 3


# ================================================================
# 資料抓取（含快取）
# ================================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    for attempt in range(3):
        try:
            tk = yf.Ticker(symbol)
            df = tk.history(period=period, auto_adjust=True)
            if df is None or df.empty or len(df) < 40:
                return None
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.dropna(subset=["Close"])
            return df
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_info(symbol: str) -> dict:
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_benchmark(market: str) -> Optional[pd.DataFrame]:
    return fetch_history(BENCHMARKS.get(market, "^GSPC"))


# ================================================================
# 技術指標計算
# ================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["Close"].astype(float)
    df["EMA21"]       = c.ewm(span=21, adjust=False).mean()
    df["MA50"]        = c.rolling(50).mean()
    df["MA200"]       = c.rolling(200).mean()
    ema12             = c.ewm(span=12, adjust=False).mean()
    ema26             = c.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    delta             = c.diff()
    gain              = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss              = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["RSI"]         = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    bb_mid            = c.rolling(20).mean()
    bb_std            = c.rolling(20).std()
    df["BB_Upper"]    = bb_mid + 2 * bb_std
    df["BB_Mid"]      = bb_mid
    df["BB_Lower"]    = bb_mid - 2 * bb_std
    df["Vol_MA20"]    = df["Volume"].astype(float).rolling(20).mean()
    return df


def ema21_alert(df: Optional[pd.DataFrame]) -> Optional[dict]:
    if df is None or len(df) < 22 or "EMA21" not in df.columns:
        return None
    last  = df.iloc[-1]
    close = float(last["Close"])
    ema   = float(last["EMA21"])
    if pd.isna(ema) or ema == 0:
        return None
    pct = (close - ema) / ema
    if abs(pct) <= EMA21_THRESHOLD:
        return {"pct": pct, "close": close, "ema21": ema,
                "side": "above" if pct >= 0 else "below"}
    return None


# ================================================================
# CAN SLIM 評分引擎
# ================================================================
def _safe(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except Exception:
        return default


def score_canslim(symbol, hist, info, bench) -> dict:
    # C — 當季EPS成長 (25分)
    c = 0
    eg = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")
    if eg is not None:
        eg = _safe(eg)
        if eg >= 1.0: c = 25
        elif eg >= 0.5: c = 22
        elif eg >= 0.25: c = 18
        elif eg >= 0.1: c = 10
        elif eg > 0: c = 5
    else:
        eps = _safe(info.get("trailingEps"))
        fwd = _safe(info.get("forwardEps"))
        if eps != 0 and fwd != 0:
            g = (fwd - eps) / abs(eps)
            if g >= 0.5: c = 22
            elif g >= 0.25: c = 18
            elif g >= 0.1: c = 10
            elif g > 0: c = 5
        elif hist is not None and len(hist) >= 63:
            q = _safe(hist["Close"].iloc[-1]) / max(_safe(hist["Close"].iloc[-63]), 1) - 1
            if q >= 0.3: c = 15
            elif q >= 0.15: c = 10
            elif q > 0: c = 5

    # A — 年度獲利成長 (20分)
    a = 0
    rg  = _safe(info.get("revenueGrowth"))
    roe = _safe(info.get("returnOnEquity"))
    pm  = _safe(info.get("profitMargins"))
    if rg:
        if rg >= 0.3: a += 9
        elif rg >= 0.15: a += 6
        elif rg >= 0.05: a += 3
        elif rg > 0: a += 1
    if roe:
        if roe >= 0.3: a += 7
        elif roe >= 0.17: a += 5
        elif roe >= 0.1: a += 3
    if pm:
        if pm >= 0.25: a += 4
        elif pm >= 0.15: a += 3
        elif pm >= 0.08: a += 2
    a = min(a, 20)

    # N — 創新高 (15分)
    n = 0
    if hist is not None and len(hist) >= 20:
        curr  = _safe(hist["Close"].iloc[-1])
        hi252 = _safe(hist["Close"].rolling(min(252, len(hist))).max().iloc[-1], curr)
        pf    = (curr / hi252 - 1) if hi252 > 0 else -1.0
        if pf >= -0.03: n = 15
        elif pf >= -0.07: n = 12
        elif pf >= -0.15: n = 8
        elif pf >= -0.25: n = 4
        elif pf >= -0.35: n = 2

    # S — 量能籌碼 (15分)
    s = 0
    if hist is not None and len(hist) >= 21:
        rec = hist.tail(21).copy()
        rec["chg"] = rec["Close"].pct_change()
        up   = float(rec.loc[rec["chg"] > 0, "Volume"].sum())
        down = float(rec.loc[rec["chg"] < 0, "Volume"].sum())
        tot  = up + down
        if tot > 0:
            r = up / tot
            if r >= 0.65: s = 15
            elif r >= 0.58: s = 12
            elif r >= 0.52: s = 8
            elif r >= 0.48: s = 4
        if len(hist) >= 25:
            avg20 = float(hist["Volume"].tail(25).iloc[:-5].mean())
            avg5  = float(hist["Volume"].tail(5).mean())
            if avg20 > 0 and avg5 / avg20 >= 1.5:
                s = min(s + 3, 15)

    # L — 相對強度 (15分)
    l = 0
    if hist is not None and bench is not None and len(hist) >= 63 and len(bench) >= 63:
        sr = _safe(hist["Close"].iloc[-1])  / max(_safe(hist["Close"].iloc[-63]),  1) - 1
        br = _safe(bench["Close"].iloc[-1]) / max(_safe(bench["Close"].iloc[-63]), 1) - 1
        rs = sr - br
        if rs >= 0.3: l = 15
        elif rs >= 0.2: l = 12
        elif rs >= 0.1: l = 8
        elif rs >= 0.04: l = 5
        elif rs >= 0: l = 3

    # I — 法人持股 (5分)
    inst = info.get("heldPercentInstitutions")
    if inst is not None:
        iv = _safe(inst)
        if iv >= 0.7: i = 5
        elif iv >= 0.5: i = 4
        elif iv >= 0.3: i = 3
        elif iv >= 0.1: i = 2
        else: i = 1
    else:
        i = 2

    # M — 市場方向 (5分)
    m = 0
    if bench is not None and len(bench) >= 50:
        bc    = bench["Close"].astype(float)
        cur   = _safe(bc.iloc[-1])
        ma50  = _safe(bc.rolling(50).mean().iloc[-1])
        ma200 = _safe(bc.rolling(min(200, len(bc))).mean().iloc[-1], cur)
        if cur > ma50:   m += 2
        if cur > ma200:  m += 2
        if ma50 > ma200: m += 1
    else:
        m = 2

    total  = c + a + n + s + l + i + m
    result = {
        "symbol": symbol, "C": c, "A": a, "N": n, "S": s, "L": l, "I": i, "M": m,
        "total": total,
        "name":   info.get("shortName") or info.get("longName") or symbol,
        "sector": info.get("sector", "—"),
        "mktcap": info.get("marketCap"),
        "price": 0.0, "chg1d": 0.0, "chg1m": 0.0, "chg3m": 0.0,
    }
    if hist is not None and len(hist) > 1:
        result["price"] = _safe(hist["Close"].iloc[-1])
        result["chg1d"] = _safe(hist["Close"].pct_change().iloc[-1])
        result["chg1m"] = _safe(hist["Close"].iloc[-1]) / max(_safe(hist["Close"].iloc[-21]), 1) - 1 if len(hist) > 21 else 0.0
        result["chg3m"] = _safe(hist["Close"].iloc[-1]) / max(_safe(hist["Close"].iloc[-63]), 1) - 1 if len(hist) > 63 else 0.0
    return result


# ================================================================
# 掃描市場（平行執行）
# ================================================================
def screen_market(universe, market, top_n=TOP_N):
    bench    = fetch_benchmark(market)
    scored   = []
    hist_map = {}

    def _process(sym):
        h = fetch_history(sym)
        if h is None: return None
        inf = fetch_info(sym)
        return score_canslim(sym, h, inf, bench), h

    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_process, s): s for s in universe}
        for fut in as_completed(futs):
            try:
                res = fut.result(timeout=45)
                if res:
                    sc, h = res
                    scored.append(sc)
                    hist_map[sc["symbol"]] = h
            except Exception:
                pass

    scored.sort(key=lambda x: x["total"], reverse=True)

    top = []
    for s in scored[:top_n]:
        s = s.copy()
        h = hist_map.get(s["symbol"])
        if h is not None:
            s["hist"] = add_indicators(h)
        top.append(s)

    return top, scored


# ================================================================
# 圖表建構
# ================================================================
_BG = "#0d1117"; _PLOT = "#131722"; _GRID = "rgba(255,255,255,0.06)"
_GREEN = "#26a69a"; _RED = "#ef5350"; _GOLD = "#FFD700"


def build_candle_chart(stock: dict) -> go.Figure:
    df  = stock.get("hist")
    sym = stock["symbol"]
    nm  = stock.get("name", sym)

    if df is None or len(df) < 20:
        fig = go.Figure()
        fig.add_annotation(text="資料不足", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="#555"))
        fig.update_layout(paper_bgcolor=_BG, plot_bgcolor=_PLOT, height=520)
        return fig

    d   = df.tail(130).copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.57, 0.22, 0.21], vertical_spacing=0.015)

    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name=sym,
        increasing=dict(line=dict(color=_GREEN, width=1), fillcolor=_GREEN),
        decreasing=dict(line=dict(color=_RED,   width=1), fillcolor=_RED),
        showlegend=False,
    ), row=1, col=1)

    if "BB_Upper" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_Upper"], mode="lines",
            line=dict(color="rgba(120,120,220,0.35)", width=1),
            showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_Lower"], mode="lines",
            line=dict(color="rgba(120,120,220,0.35)", width=1),
            fill="tonexty", fillcolor="rgba(100,100,200,0.06)",
            showlegend=False, hoverinfo="skip"), row=1, col=1)

    if "MA200" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["MA200"], mode="lines", name="MA200",
            line=dict(color="#ff69b4", width=1.4, dash="dot")), row=1, col=1)
    if "MA50" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["MA50"], mode="lines", name="MA50",
            line=dict(color="#00bfff", width=1.6)), row=1, col=1)
    if "EMA21" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["EMA21"], mode="lines", name="EMA21 ⚡",
            line=dict(color=_GOLD, width=2.5)), row=1, col=1)

    vcols = [_GREEN if float(c) >= float(o) else _RED for c, o in zip(d["Close"], d["Open"])]
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=vcols,
                         opacity=0.75, showlegend=False), row=2, col=1)
    if "Vol_MA20" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["Vol_MA20"], mode="lines",
            line=dict(color="orange", width=1.2), showlegend=False), row=2, col=1)

    if "MACD" in d.columns:
        hcols = [_GREEN if v >= 0 else _RED for v in d["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(x=d.index, y=d["MACD_Hist"], marker_color=hcols,
                             opacity=0.65, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], mode="lines",
            line=dict(color="#00bfff", width=1.4), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["MACD_Signal"], mode="lines",
            line=dict(color="#ff8c00", width=1.4), showlegend=False), row=3, col=1)

    p, c1d = stock.get("price", 0), stock.get("chg1d", 0)
    sgn, cc = ("▲", _GREEN) if c1d >= 0 else ("▼", _RED)
    fig.update_layout(
        title=dict(
            text=f"<b>{sym}</b>  {nm}  <span style='color:{cc}'>{p:,.2f}  {sgn}{abs(c1d*100):.2f}%</span>",
            font=dict(size=12, color="#ddd"), x=0),
        height=520, paper_bgcolor=_BG, plot_bgcolor=_PLOT,
        font=dict(color="#888", size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        xaxis_rangeslider_visible=False, margin=dict(l=40, r=14, t=42, b=6),
        hovermode="x unified",
    )
    for row in range(1, 4):
        fig.update_xaxes(row=row, col=1, gridcolor=_GRID, showgrid=True,
                         zeroline=False, showticklabels=(row == 3))
        fig.update_yaxes(row=row, col=1, gridcolor=_GRID, showgrid=True,
                         zeroline=False, side="right")
    return fig


def build_radar(stock: dict) -> go.Figure:
    cats = ["C 當季EPS", "A 年獲利", "N 創新高", "S 量能", "L 相對強度", "I 法人", "M 市場"]
    maxs = [25, 20, 15, 15, 15, 5, 5]
    vals = [stock.get(k, 0) for k in ["C","A","N","S","L","I","M"]]
    pcts = [v / m * 100 for v, m in zip(vals, maxs)]
    fig  = go.Figure(go.Scatterpolar(
        r=pcts + [pcts[0]], theta=cats + [cats[0]], fill="toself",
        fillcolor="rgba(0,255,136,0.12)", line=dict(color="#00ff88", width=2),
        hovertemplate="%{theta}: %{r:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(bgcolor="#131722",
            radialaxis=dict(visible=True, range=[0,100],
                gridcolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="#555", size=8), tickvals=[25,50,75,100]),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="#aaa", size=8))),
        paper_bgcolor=_BG, height=210,
        margin=dict(l=25, r=25, t=15, b=15), showlegend=False,
    )
    return fig


def _scolor(score, mx=100.0):
    r = score / mx if mx else 0
    return "#00ff88" if r >= 0.72 else ("#FFD700" if r >= 0.48 else "#ef5350")


# ================================================================
# 市場分頁渲染（詳細圖表）
# ================================================================
def render_market_tab(top: list, all_scores: list) -> None:
    if not top:
        st.warning("⚠️ 無法取得足夠資料，請稍後重試。")
        return

    cols = st.columns(len(top))
    for i, (col, stk) in enumerate(zip(cols, top)):
        with col:
            sym, nm = stk["symbol"], stk.get("name", stk["symbol"])
            total   = stk.get("total", 0)
            sc_c    = _scolor(float(total))
            st.markdown(f"""
            <div class="rank-card">
              <span style="font-size:1.05rem;font-weight:800;color:#eee">#{i+1} {sym}</span><br>
              <span style="font-size:.78rem;color:#777">{nm}</span><br>
              <span style="font-size:1.5rem;font-weight:900;color:{sc_c}">
                {total}<span style="font-size:.8rem;color:#555">/100</span>
              </span>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(build_candle_chart(stk), use_container_width=True,
                            key=f"main_{sym}_{i}")

            c1, c2 = st.columns([3, 2])
            with c1:
                st.plotly_chart(build_radar(stk), use_container_width=True,
                                key=f"radar_{sym}_{i}")
            with c2:
                st.markdown("**分項得分**")
                for k, lbl, mx in [
                    ("C","C 當季EPS",25),("A","A 年獲利",20),("N","N 創新高",15),
                    ("S","S 量能",15),  ("L","L 相對強度",15),("I","I 法人",5),("M","M 市場",5),
                ]:
                    v  = stk.get(k, 0)
                    cc = _scolor(float(v), float(mx))
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'font-size:.78rem;margin:3px 0;">'
                        f'<span style="color:#888">{lbl}</span>'
                        f'<span style="color:{cc};font-weight:700">{v}/{mx}</span></div>',
                        unsafe_allow_html=True)

    if len(all_scores) > TOP_N:
        with st.expander(f"📋 完整排行榜（共掃描 {len(all_scores)} 支）"):
            rows = []
            for r, s in enumerate(all_scores[:25], 1):
                rows.append({"名次":r, "代號":s["symbol"],
                    "名稱":s.get("name",s["symbol"])[:14], "總分":s.get("total",0),
                    "C":s.get("C",0),"A":s.get("A",0),"N":s.get("N",0),
                    "S":s.get("S",0),"L":s.get("L",0),
                    "日漲跌":f"{s.get('chg1d',0)*100:+.2f}%",
                    "月漲跌":f"{s.get('chg1m',0)*100:+.2f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ================================================================
# 主程式
# ================================================================
def main() -> None:
    if HAS_AUTOREFRESH:
        st_autorefresh(interval=15 * 60 * 1000, key="canslim_auto")

    now_tw = datetime.now(pytz.timezone("Asia/Taipei")).strftime("%Y-%m-%d %H:%M")
    now_ny = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d %H:%M")

    st.markdown(f"""
    <div class="header-bar">
      <p class="gradient-title">📈 CAN SLIM 智能選股看板</p>
      <p style="color:#666;margin:4px 0 0 0;font-size:.85rem;">
        William O'Neil 法則 ｜ 台股上市・上櫃・美股 ｜
        🕐 台北 <b style="color:#aaa">{now_tw}</b> ｜
        🗽 紐約 <b style="color:#aaa">{now_ny}</b> ｜
        {'🔄 每15分鐘自動更新' if HAS_AUTOREFRESH else '⚠️ 安裝 streamlit-autorefresh 以啟用自動更新'}
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── 掃描三市場 ──
    with st.spinner("🔍 CAN SLIM 掃描中，請稍候…"):
        prog = st.progress(0, text="掃描台股上市 (25支)…")
        twse_top, twse_all = screen_market(TWSE_UNIVERSE, "TWSE")
        prog.progress(34, text="掃描台股上櫃 (20支)…")
        tpex_top, tpex_all = screen_market(TPEX_UNIVERSE, "TPEX")
        prog.progress(67, text="掃描美股 (30支)…")
        us_top, us_all = screen_market(US_UNIVERSE, "US")
        prog.progress(100, text="✅ 掃描完成！")
        time.sleep(0.25)
        prog.empty()

    # 合併9名（供警示用）
    all9 = ([(s, "🇹🇼 上市") for s in twse_top] +
            [(s, "🏪 上櫃")  for s in tpex_top] +
            [(s, "🇺🇸 美股") for s in us_top])

    # ── 21日均線接觸警示 ──
    alerts = [(stk, mkt, ema21_alert(stk.get("hist")))
              for stk, mkt in all9 if ema21_alert(stk.get("hist")) is not None]
    if alerts:
        st.markdown("### ⚡ 21日均線接觸警示")
        for stk, mkt, al in alerts:
            sym, nm = stk["symbol"], stk.get("name", stk["symbol"])
            pct     = al["pct"] * 100
            side_zh = "上方" if al["side"] == "above" else "下方"
            arrow_c = "#26a69a" if al["side"] == "above" else "#ef5350"
            arrow   = "▲" if al["side"] == "above" else "▼"
            cls     = "alert-touch" if abs(pct) <= 0.5 else "alert-ema"
            st.markdown(
                f'<div class="{cls}">⚡ <b style="color:#FFD700">{mkt} {sym}</b>'
                f' <span style="color:#bbb">({nm})</span>'
                f' 現價 <b style="color:#fff">{al["close"]:,.2f}</b>'
                f' 位於21日EMA {side_zh}'
                f' <b style="color:{arrow_c}">{arrow} {abs(pct):.2f}%</b>'
                f' ｜ EMA21 = {al["ema21"]:,.2f}</div>',
                unsafe_allow_html=True)
        st.markdown("---")

    # ================================================================
    # 三市場各前3名總覽（共9名）
    # ================================================================
    twse_cnt = len(twse_top)
    tpex_cnt = len(tpex_top)
    us_cnt   = len(us_top)

    st.markdown("### 📊 CAN SLIM 入選個股 — 各市場前3名，合計共9名")

    # 市場統計卡片
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(
            f'<div style="background:#0d2a1a;border:1px solid #00ff88;border-radius:8px;'
            f'padding:10px;text-align:center;">'
            f'<span style="font-size:1.3rem">🇹🇼</span><br>'
            f'<b style="color:#00ff88;font-size:1rem">台股上市</b><br>'
            f'<span style="color:#aaa;font-size:.85rem">前 {twse_cnt} 名｜掃描 {len(twse_all)} 支</span>'
            f'</div>', unsafe_allow_html=True)
    with h2:
        st.markdown(
            f'<div style="background:#0d1a2a;border:1px solid #00bfff;border-radius:8px;'
            f'padding:10px;text-align:center;">'
            f'<span style="font-size:1.3rem">🏪</span><br>'
            f'<b style="color:#00bfff;font-size:1rem">台股上櫃</b><br>'
            f'<span style="color:#aaa;font-size:.85rem">前 {tpex_cnt} 名｜掃描 {len(tpex_all)} 支</span>'
            f'</div>', unsafe_allow_html=True)
    with h3:
        st.markdown(
            f'<div style="background:#1a0d2a;border:1px solid #bf00ff;border-radius:8px;'
            f'padding:10px;text-align:center;">'
            f'<span style="font-size:1.3rem">🇺🇸</span><br>'
            f'<b style="color:#bf00ff;font-size:1rem">美股</b><br>'
            f'<span style="color:#aaa;font-size:.85rem">前 {us_cnt} 名｜掃描 {len(us_all)} 支</span>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin:12px 0'></div>", unsafe_allow_html=True)

    def _market_table(top_stocks: list) -> None:
        if not top_stocks:
            st.warning("資料不足，請稍後重試。")
            return
        rows = []
        for rank, stk in enumerate(top_stocks, 1):
            rows.append({
                "名次": rank,
                "代號": stk["symbol"],
                "名稱": stk.get("name", stk["symbol"])[:12],
                "現價": f"{stk.get('price', 0):,.2f}",
                "日%":  f"{stk.get('chg1d', 0)*100:+.2f}%",
                "月%":  f"{stk.get('chg1m', 0)*100:+.2f}%",
                "C":    stk.get("C", 0),
                "A":    stk.get("A", 0),
                "N":    stk.get("N", 0),
                "S":    stk.get("S", 0),
                "L":    stk.get("L", 0),
                "總分": stk.get("total", 0),
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "C":   st.column_config.ProgressColumn("C", min_value=0, max_value=25,  format="%d"),
                "A":   st.column_config.ProgressColumn("A", min_value=0, max_value=20,  format="%d"),
                "N":   st.column_config.ProgressColumn("N", min_value=0, max_value=15,  format="%d"),
                "S":   st.column_config.ProgressColumn("S", min_value=0, max_value=15,  format="%d"),
                "L":   st.column_config.ProgressColumn("L", min_value=0, max_value=15,  format="%d"),
                "總分": st.column_config.ProgressColumn("總分/100", min_value=0, max_value=100, format="%d"),
            },
        )

    # 三欄並排
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<p style="color:#00ff88;font-weight:700;margin:0 0 6px 0">🇹🇼 台股上市 前3名</p>',
                    unsafe_allow_html=True)
        _market_table(twse_top)
    with col2:
        st.markdown('<p style="color:#00bfff;font-weight:700;margin:0 0 6px 0">🏪 台股上櫃 前3名</p>',
                    unsafe_allow_html=True)
        _market_table(tpex_top)
    with col3:
        st.markdown('<p style="color:#bf00ff;font-weight:700;margin:0 0 6px 0">🇺🇸 美股 前3名</p>',
                    unsafe_allow_html=True)
        _market_table(us_top)

    st.markdown("---")

    # ── 詳細技術分析分頁 ──
    st.markdown("### 📈 詳細技術分析（K線・EMA21・MACD・雷達圖）")
    tab1, tab2, tab3 = st.tabs([
        f"🇹🇼 台股上市 前{twse_cnt}名",
        f"🏪 台股上櫃 前{tpex_cnt}名",
        f"🇺🇸 美股 前{us_cnt}名",
    ])
    with tab1: render_market_tab(twse_top, twse_all)
    with tab2: render_market_tab(tpex_top, tpex_all)
    with tab3: render_market_tab(us_top, us_all)

    st.markdown("---")
    with st.expander("📖 CAN SLIM 評分說明"):
        st.markdown("""
| 指標 | 說明 | 滿分 |
|------|------|:----:|
| **C** Current Earnings | 當季EPS年增率（目標 ≥ 25%） | 25 |
| **A** Annual Earnings  | 年度獲利成長（ROE・利潤率・營收增速） | 20 |
| **N** New High         | 接近52週高點或突破創新高 | 15 |
| **S** Supply & Demand  | 21日上漲量能 vs 下跌量能（籌碼積累） | 15 |
| **L** Leader           | 相對大盤強弱（近63個交易日） | 15 |
| **I** Institutional    | 法人持股比例 | 5 |
| **M** Market Direction | 大盤位於MA50/MA200上方 | 5 |

**21日均線警示**：9支個股中任一收盤價距EMA21在 **±1.5%** 以內時，自動顯示警示框。

> ⚠️ 本看板僅供學習參考，不構成投資建議。資料來源：Yahoo Finance。
        """)

    st.markdown(
        f'<div style="text-align:center;color:#333;font-size:.72rem;padding:10px 0">'
        f"CAN SLIM © William J. O'Neil ｜ 資料: Yahoo Finance ｜ "
        f"更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
