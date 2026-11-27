import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from google import genai
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 頁面設定（改為寬版面，平板與電腦看多指標圖表更舒適）
st.set_page_config(page_title="全方位 AI 量化分析系統", page_icon="📈", layout="wide")

# 初始化 Session State
if "ticker_input" not in st.session_state:
    st.session_state["ticker_input"] = "2454"
if "ai_report" not in st.session_state:
    st.session_state["ai_report"] = ""
if "ai_report_meta" not in st.session_state:
    st.session_state["ai_report_meta"] = ""

def set_ticker(code: str):
    st.session_state["ticker_input"] = code
    st.session_state["ai_report"] = ""

# ==========================================
# 快取函式：抓取股價、基本面與新聞（快取 10 分鐘）
# ==========================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_data(raw_code: str, period: str):
    code_clean = raw_code.strip().upper()
    if code_clean.endswith(".TW") or code_clean.endswith(".TWO") or code_clean.startswith("^"):
        candidates = [code_clean]
    elif code_clean and code_clean[0].isdigit():
        candidates = [f"{code_clean}.TW", f"{code_clean}.TWO"]
    else:
        candidates = [code_clean]

    for code in candidates:
        ticker = yf.Ticker(code)
        df = ticker.history(period=period)
        if df is not None and not df.empty:
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            try:
                info = ticker.info or {}
            except Exception:
                info = {}
            return code, df, info
    return None, None, {}

@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(stock_id: str):
    news_list = []
    stock_name = stock_id

    # 台股優先抓 Yahoo 股市繁中新聞
    if stock_id.endswith(".TW") or stock_id.endswith(".TWO"):
        base_url = "https://tw.stock.yahoo.com"
        url = f"{base_url}/quote/{stock_id}/news"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            soup = BeautifulSoup(resp.text, "html.parser")
            if soup.title and "(" in soup.title.text:
                stock_name = soup.title.text.split("(")[0].strip()

            seen = set()
            for tag in soup.find_all("h3"):
                title = tag.get_text(strip=True)
                if len(title) > 8 and "個股相關新聞與公告" not in title and title not in seen:
                    link_tag = tag.find("a") or tag.find_parent("a")
                    href = link_tag.get("href", "") if link_tag else ""
                    full_url = urljoin(base_url, href) if href else ""
                    news_list.append((title, full_url))
                    seen.add(title)
                    if len(news_list) >= 5:
                        break
        except Exception:
            pass
    else:
        # 美股與指數改抓 yfinance 內建新聞
        try:
            ticker = yf.Ticker(stock_id)
            raw_news = ticker.news or []
            for item in raw_news[:5]:
                content = item.get("content", item)
                title = content.get("title", "")
                link = ""
                if isinstance(content.get("canonicalUrl"), dict):
                    link = content["canonicalUrl"].get("url", "")
                elif isinstance(content.get("clickThroughUrl"), dict):
                    link = content["clickThroughUrl"].get("url", "")
                else:
                    link = item.get("link", "")
                if title:
                    news_list.append((title, link))
        except Exception:
            pass

    return stock_name, news_list

# ==========================================
# 技術指標計算
# ==========================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 均線
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA60"] = df["Close"].rolling(window=60).mean()

    # 布林通道 (20, 2)
    std20 = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["MA20"] + 2 * std20
    df["BB_Lower"] = df["MA20"] - 2 * std20

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["DIF"] = ema12 - ema26
    df["DEA"] = df["DIF"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["DIF"] - df["DEA"]

    # KD (9, 3, 3)
    low_min = df["Low"].rolling(window=9).min()
    high_max = df["High"].rolling(window=9).max()
    rsv = ((df["Close"] - low_min) / (high_max - low_min).replace(0, pd.NA) * 100).fillna(50.0)
    k_vals, d_vals = [50.0], [50.0]
    for val in rsv.iloc[1:]:
        k = (2 / 3) * k_vals[-1] + (1 / 3) * float(val)
        d = (2 / 3) * d_vals[-1] + (1 / 3) * k
        k_vals.append(k)
        d_vals.append(d)
    df["K"] = k_vals
    df["D"] = d_vals
    return df

# ==========================================
# 簡易策略回測計算
# ==========================================
def run_backtest(df: pd.DataFrame, strategy_name: str):
    bt = calculate_indicators(df).dropna(subset=["MA20", "RSI"]).copy()
    if len(bt) < 20:
        return None

    positions = []
    pos = 0
    trade_returns = []
    entry_price = 0.0

    for i in range(len(bt)):
        close = bt["Close"].iloc[i]
        ma20 = bt["MA20"].iloc[i]
        rsi = bt["RSI"].iloc[i]

        if strategy_name == "MA20 均線突破策略":
            if pos == 0 and close > ma20:
                pos = 1
                entry_price = close
            elif pos == 1 and close < ma20:
                pos = 0
                trade_returns.append((close - entry_price) / entry_price)
        else:  # RSI 超買超賣策略
            if pos == 0 and rsi < 30:
                pos = 1
                entry_price = close
            elif pos == 1 and rsi > 70:
                pos = 0
                trade_returns.append((close - entry_price) / entry_price)

        positions.append(pos)

    # 若期末仍持有，以最後收盤價結算該筆未實現損益計入勝率
    if pos == 1 and entry_price > 0:
        trade_returns.append((bt["Close"].iloc[-1] - entry_price) / entry_price)

    bt["Position"] = pd.Series(positions, index=bt.index).shift(1).fillna(0)
    bt["Daily_Ret"] = bt["Close"].pct_change().fillna(0)
    bt["Strat_Ret"] = bt["Position"] * bt["Daily_Ret"]

    bt["BuyHold_Equity"] = (1 + bt["Daily_Ret"]).cumprod()
    bt["Strat_Equity"] = (1 + bt["Strat_Ret"]).cumprod()

    strat_total_ret = (bt["Strat_Equity"].iloc[-1] - 1) * 100
    bh_total_ret = (bt["BuyHold_Equity"].iloc[-1] - 1) * 100

    # 最大回撤 (MDD)
    rolling_max = bt["Strat_Equity"].cummax()
    drawdown = (bt["Strat_Equity"] - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    total_trades = len(trade_returns)
    win_rate = (sum(1 for r in trade_returns if r > 0) / total_trades * 100) if total_trades > 0 else 0.0

    return {
        "df": bt,
        "strat_ret": strat_total_ret,
        "bh_ret": bh_total_ret,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "trades": total_trades,
    }

# ==========================================
# 左側邊欄：自選股清單、時間區間、指標開關
# ==========================================
with st.sidebar:
    st.header("⚙️ 看盤設定")

    st.subheader("⭐ 快捷自選股")
    watchlist = [
        ("2330", "台積電"),
        ("2454", "聯發科"),
        ("0050", "台灣50"),
        ("^TWII", "加權指數"),
        ("^SOX", "費城半導體"),
        ("NVDA", "NVIDIA"),
        ("TSM", "台積電ADR"),
    ]
    cols = st.columns(2)
    for idx, (code, label) in enumerate(watchlist):
        cols[idx % 2].button(
            f"{code} {label}",
            key=f"wl_{code}",
            use_container_width=True,
            on_click=set_ticker,
            args=(code,),
        )

    st.divider()
    period_map = {
        "1 個月": "1mo",
        "3 個月": "3mo",
        "6 個月": "6mo",
        "1 年": "1y",
        "3 年": "3y",
    }
    selected_period_label = st.selectbox("📅 時間區間", list(period_map.keys()), index=2)
    selected_period = period_map[selected_period_label]

    st.divider()
    st.subheader("📊 主圖疊加指標")
    show_ma5 = st.checkbox("MA5 (週線)", value=True)
    show_ma20 = st.checkbox("MA20 (月線)", value=True)
    show_ma60 = st.checkbox("MA60 (季線)", value=False)
    show_bb = st.checkbox("布林通道 (BBands)", value=False)

    st.subheader("📉 副圖技術指標")
    show_rsi = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD (12,26,9)", value=False)
    show_kd = st.checkbox("KD (9,3,3)", value=False)

# ==========================================
# 主畫面標題與輸入列
# ==========================================
st.title("📈 全方位 AI 量化分析系統")
st.caption("支援台股上市櫃、美股與大盤指數｜多指標互動 K 線、基本面、策略回測、雙股對比與 Gemini AI 短評")

col_in, col_btn = st.columns([4, 1])
with col_in:
    raw_code = st.text_input(
        "輸入股票或指數代號（台股如 2330、0050；美股如 NVDA、TSM；指數如 ^TWII、^SOX）",
        key="ticker_input",
    )
with col_btn:
    st.write("")
    st.write("")
    if st.button("🔄 重新整理資料", use_container_width=True):
        fetch_stock_data.clear()
        fetch_news.clear()
        st.rerun()

# 抓取主標的資料
with st.spinner(f"正在載入 {raw_code} 市場資料..."):
    stock_id, raw_df, info = fetch_stock_data(raw_code, selected_period)

if raw_df is None:
    st.error(f"找不到代號「{raw_code}」的股價資料，請確認代號是否正確。")
    st.stop()

df = calculate_indicators(raw_df)
stock_name_news, news_list = fetch_news(stock_id)
display_name = stock_name_news if stock_name_news != stock_id else info.get("shortName", stock_id)

# 建立三大功能分頁
tab1, tab2, tab3 = st.tabs(["📊 綜合看盤與 AI 分析", "🧪 簡易策略回測", "⚖️ 雙股強弱對比"])

# ==========================================
# 分頁 1：綜合看盤、基本面、新聞與 AI 報告
# ==========================================
with tab1:
    latest_date = df.index[-1].strftime("%Y-%m-%d")
    latest_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2] if len(df) > 1 else latest_close
    price_diff = latest_close - prev_close
    pct_diff = (price_diff / prev_close * 100) if prev_close else 0.0
    latest_ma20 = df["MA20"].iloc[-1]
    latest_rsi = df["RSI"].iloc[-1]
    currency = info.get("currency", "TWD" if ".TW" in stock_id else "USD")

    st.subheader(f"{display_name} ({stock_id})｜最新資料日期：{latest_date}")

    # 第一排：價格與技術指標卡片
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最新收盤價", f"{latest_close:.2f} {currency}", f"{price_diff:+.2f} ({pct_diff:+.2f}%)")
    m2.metric("20 日均線 (MA20)", f"{latest_ma20:.2f}" if pd.notna(latest_ma20) else "N/A")
    m3.metric("RSI (14)", f"{latest_rsi:.2f}" if pd.notna(latest_rsi) else "N/A")
    m4.metric("最新 K / D 值", f"{df['K'].iloc[-1]:.1f} / {df['D'].iloc[-1]:.1f}")

    # 第二排：基本面數據卡
    pe_ratio = info.get("trailingPE")
    dy_raw = info.get("trailingAnnualDividendYield")
    if dy_raw is None:
        dy_raw = info.get("dividendYield")
    if dy_raw is not None:
        dy_str = f"{dy_raw:.2f}%" if dy_raw > 1 else f"{dy_raw * 100:.2f}%"
    else:
        dy_str = "N/A"
    high_52w = info.get("fiftyTwoWeekHigh")
    low_52w = info.get("fiftyTwoWeekLow")

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("本益比 (P/E)", f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A")
    f2.metric("殖利率", dy_str)
    f3.metric("52 週最高", f"{high_52w:.2f}" if isinstance(high_52w, (int, float)) else "N/A")
    f4.metric("52 週最低", f"{low_52w:.2f}" if isinstance(low_52w, (int, float)) else "N/A")

    # 動態組合互動式 Plotly 圖表列數
    sub_panels = []
    if show_rsi:
        sub_panels.append("RSI (14)")
    if show_macd:
        sub_panels.append("MACD (12,26,9)")
    if show_kd:
        sub_panels.append("KD (9,3,3)")

    total_rows = 2 + len(sub_panels)
    if len(sub_panels) == 0:
        row_heights = [0.78, 0.22]
        chart_height = 560
    elif len(sub_panels) == 1:
        row_heights = [0.58, 0.18, 0.24]
        chart_height = 700
    elif len(sub_panels) == 2:
        row_heights = [0.48, 0.14, 0.19, 0.19]
        chart_height = 840
    else:
        row_heights = [0.42, 0.13, 0.15, 0.15, 0.15]
        chart_height = 960

    subplot_titles = [f"{display_name} ({stock_id}) K 線圖", "成交量"] + sub_panels
    dates = df.index.strftime("%Y-%m-%d")
    vol_colors = ["#e74c3c" if c >= o else "#2ecc71" for c, o in zip(df["Close"], df["Open"])]

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Row 1: K 線與疊加指標
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="K 線",
            increasing_line_color="#e74c3c",
            increasing_fillcolor="#e74c3c",
            decreasing_line_color="#2ecc71",
            decreasing_fillcolor="#2ecc71",
        ),
        row=1, col=1,
    )

    if show_ma5:
        fig.add_trace(go.Scatter(x=dates, y=df["MA5"], mode="lines", name="MA5", line=dict(color="#3498db", width=1.4)), row=1, col=1)
    if show_ma20:
        fig.add_trace(go.Scatter(x=dates, y=df["MA20"], mode="lines", name="MA20", line=dict(color="#ff7f0e", width=1.8)), row=1, col=1)
    if show_ma60:
        fig.add_trace(go.Scatter(x=dates, y=df["MA60"], mode="lines", name="MA60", line=dict(color="#9b59b6", width=1.8)), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=dates, y=df["BB_Upper"], mode="lines", name="布林上軌", line=dict(color="rgba(128,128,128,0.6)", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=df["BB_Lower"], mode="lines", name="布林下軌", line=dict(color="rgba(128,128,128,0.6)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(128,128,128,0.08)"), row=1, col=1)

    # Row 2: 成交量
    fig.add_trace(
        go.Bar(x=dates, y=df["Volume"], name="成交量", marker_color=vol_colors, opacity=0.8),
        row=2, col=1,
    )

    # 副圖區
    current_row = 3
    if show_rsi:
        fig.add_trace(go.Scatter(x=dates, y=df["RSI"], mode="lines", name="RSI(14)", line=dict(color="#9467bd", width=1.6)), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="red", opacity=0.6, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.6, row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=current_row, col=1)
        current_row += 1

    if show_macd:
        macd_colors = ["#e74c3c" if v >= 0 else "#2ecc71" for v in df["MACD_Hist"]]
        fig.add_trace(go.Bar(x=dates, y=df["MACD_Hist"], name="MACD柱", marker_color=macd_colors, opacity=0.65), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=dates, y=df["DIF"], mode="lines", name="DIF", line=dict(color="#ff7f0e", width=1.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=dates, y=df["DEA"], mode="lines", name="DEA", line=dict(color="#3498db", width=1.5)), row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        current_row += 1

    if show_kd:
        fig.add_trace(go.Scatter(x=dates, y=df["K"], mode="lines", name="K 值", line=dict(color="#e67e22", width=1.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=dates, y=df["D"], mode="lines", name="D 值", line=dict(color="#2980b9", width=1.5)), row=current_row, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color="red", opacity=0.6, row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="green", opacity=0.6, row=current_row, col=1)
        fig.update_yaxes(title_text="KD", range=[0, 100], row=current_row, col=1)

    fig.update_layout(
        height=chart_height,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=55, b=10),
    )
    fig.update_xaxes(type="category", nticks=8)
    fig.update_yaxes(title_text=f"價格 ({currency})", row=1, col=1)
    fig.update_yaxes(title_text="量能", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # 近期相關新聞（可點擊）
    with st.expander("📰 查看近期相關財經新聞（點擊標題看原文）", expanded=True):
        if news_list:
            for i, (title, link) in enumerate(news_list, 1):
                if link:
                    st.markdown(f"{i}. [{title}]({link})")
                else:
                    st.write(f"{i}. {title}")
        else:
            st.write("暫時沒有抓到相關新聞標題，將改以技術面與基本面進行分析。")

    # AI 分析區（可選風格＋一鍵下載紀錄）
    st.divider()
    st.subheader("🤖 Gemini AI 量化分析報告")
    col_style, col_ai_btn = st.columns([3, 1])
    with col_style:
        ai_style = st.selectbox(
            "選擇分析視角",
            ["綜合基本面與消息面", "短線技術面操作視角", "波段趨勢與風險評估"],
        )
    with col_ai_btn:
        st.write("")
        st.write("")
        gen_ai_btn = st.button("✨ 產生 AI 短評", type="primary", use_container_width=True)

    if gen_ai_btn:
        with st.spinner("正在請 Gemini AI 撰寫量化短評..."):
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
                client = genai.Client(api_key=api_key)
                news_text = "\n".join(f"- {t}" for t, _ in news_list) if news_list else "（無最新新聞）"
                ma20_txt = f"{latest_ma20:.2f}" if pd.notna(latest_ma20) else "N/A"
                rsi_txt = f"{latest_rsi:.2f}" if pd.notna(latest_rsi) else "N/A"

                style_instruction = {
                    "綜合基本面與消息面": "請平衡整合技術面、本益比/殖利率等基本面與新聞面，給出約 180 字的客觀總評與觀察重點。",
                    "短線技術面操作視角": "請聚焦於均線排列、RSI、KD、MACD 動能與支撐壓力區，撰寫約 180 字的短線技術面重點觀察。",
                    "波段趨勢與風險評估": "請從中長線波段角度切入，重點分析目前位階（對照 52 週高低點）、潛在利空風險與防守點位，約 180 字。",
                }[ai_style]

                prompt = f"""
以下是 {stock_id} {display_name} 的最新資訊（資料日期：{latest_date}）：
- 最新收盤價：{latest_close:.2f} {currency}（單日漲跌：{pct_diff:+.2f}%）
- MA5：{df['MA5'].iloc[-1]:.2f} / MA20：{ma20_txt}
- RSI(14)：{rsi_txt} / KD：{df['K'].iloc[-1]:.1f}, {df['D'].iloc[-1]:.1f}
- 本益比：{pe_ratio if pe_ratio else 'N/A'} / 殖利率：{dy_str}
- 52 週最高 / 最低：{high_52w if high_52w else 'N/A'} / {low_52w if low_52w else 'N/A'}

近期相關新聞：
{news_text}

請扮演專業的量化分析師，依照以下風格撰寫繁體中文分析報告：
{style_instruction}
"""
                ai_response = client.models.generate_content(
                    model="models/gemini-3.1-flash-lite",
                    contents=prompt,
                )
                st.session_state["ai_report"] = ai_response.text
                st.session_state["ai_report_meta"] = f"{stock_id} ({display_name})｜{latest_date}｜視角：{ai_style}"
            except Exception as e:
                st.error(f"AI 分析發生錯誤：{e}")

    if st.session_state["ai_report"]:
        st.caption(st.session_state["ai_report_meta"])
        st.info(st.session_state["ai_report"])

        download_content = (
            f"標的：{display_name} ({stock_id})\n"
            f"資料日期：{latest_date}\n"
            f"收盤價：{latest_close:.2f} {currency} ({pct_diff:+.2f}%)\n"
            f"MA20：{latest_ma20:.2f}｜RSI(14)：{latest_rsi:.2f}\n"
            f"本益比：{pe_ratio if pe_ratio else 'N/A'}｜殖利率：{dy_str}\n"
            f"分析視角：{ai_style}\n"
            f"{'='*40}\n"
            f"{st.session_state['ai_report']}\n"
        )
        st.download_button(
            label="📥 下載本次觀察報告 (.txt)",
            data=download_content,
            file_name=f"{stock_id}_{latest_date}_AI報告.txt",
            mime="text/plain",
        )

# ==========================================
# 分頁 2：簡易策略回測
# ==========================================
with tab2:
    st.subheader(f"🧪 {display_name} ({stock_id}) 簡易量化策略回測")
    st.caption("以收盤訊號於次一交易日進出場模擬（未計入手續費與滑價），並與「買進持有 (Buy & Hold)」對照。")

    c_strat, c_bt_period = st.columns(2)
    with c_strat:
        strat_choice = st.selectbox("選擇回測策略", ["MA20 均線突破策略", "RSI 超買超賣策略 (30買 / 70賣)"])
    with c_bt_period:
        bt_period_map = {"近 6 個月": "6mo", "近 1 年": "1y", "近 3 年": "3y"}
        bt_period_label = st.selectbox("回測區間", list(bt_period_map.keys()), index=1)

    _, bt_raw_df, _ = fetch_stock_data(stock_id, bt_period_map[bt_period_label])
    bt_res = run_backtest(bt_raw_df, strat_choice) if bt_raw_df is not None else None

    if bt_res is None:
        st.warning("歷史資料筆數不足，無法進行回測，請嘗試切換較長區間。")
    else:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("策略累積報酬率", f"{bt_res['strat_ret']:+.2f}%")
        b2.metric("買進持有報酬率", f"{bt_res['bh_ret']:+.2f}%")
        b3.metric("交易勝率 (次數)", f"{bt_res['win_rate']:.1f}% ({bt_res['trades']} 次)")
        b4.metric("策略最大回撤 (MDD)", f"{bt_res['max_dd']:.2f}%")

        bt_df = bt_res["df"]
        bt_dates = bt_df.index.strftime("%Y-%m-%d")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_dates, y=(bt_df["Strat_Equity"] - 1) * 100, mode="lines", name=strat_choice, line=dict(color="#e74c3c", width=2.2)))
        fig_bt.add_trace(go.Scatter(x=bt_dates, y=(bt_df["BuyHold_Equity"] - 1) * 100, mode="lines", name="買進持有 (Buy & Hold)", line=dict(color="#7f8c8d", width=1.8, dash="dash")))
        fig_bt.update_layout(
            title=f"{stock_id} 累積報酬率走勢對比 (%)",
            height=450,
            hovermode="x unified",
            yaxis_title="累積報酬率 (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        fig_bt.update_xaxes(type="category", nticks=8)
        st.plotly_chart(fig_bt, use_container_width=True)

# ==========================================
# 分頁 3：雙股強弱對比圖
# ==========================================
with tab3:
    st.subheader("⚖️ 雙股相對強弱對比（起點歸零漲跌幅 %）")
    st.caption("可輸入台股對照大盤（如 2330 vs 0050），或跨市場對照美台連動（如 2330 vs TSM、^TWII vs ^SOX）。")

    cc1, cc2, cc3 = st.columns([2, 2, 1.5])
    with cc1:
        comp_a = st.text_input("標的 A", value=raw_code).strip().upper()
    with cc2:
        default_b = "0050" if raw_code != "0050" else "2330"
        comp_b = st.text_input("標的 B", value=default_b).strip().upper()
    with cc3:
        comp_period_map = {"3 個月": "3mo", "6 個月": "6mo", "1 年": "1y", "3 年": "3y"}
        comp_period_label = st.selectbox("對比區間", list(comp_period_map.keys()), index=1)

    if comp_a and comp_b:
        id_a, df_a, _ = fetch_stock_data(comp_a, comp_period_map[comp_period_label])
        id_b, df_b, _ = fetch_stock_data(comp_b, comp_period_map[comp_period_label])

        if df_a is None or df_b is None:
            st.error("其中一檔標的找不到資料，請檢查代號是否正確。")
        else:
            # 依日期對齊（跨市場休市日自動以前值填補）
            merged = pd.DataFrame({
                id_a: df_a["Close"],
                id_b: df_b["Close"],
            }).sort_index().ffill().dropna()

            if len(merged) < 2:
                st.warning("兩檔標的沒有足夠的重疊交易日資料。")
            else:
                norm_a = (merged[id_a] / merged[id_a].iloc[0] - 1) * 100
                norm_b = (merged[id_b] / merged[id_b].iloc[0] - 1) * 100
                spread = norm_a.iloc[-1] - norm_b.iloc[-1]

                k1, k2, k3 = st.columns(3)
                k1.metric(f"{id_a} 區間漲跌幅", f"{norm_a.iloc[-1]:+.2f}%")
                k2.metric(f"{id_b} 區間漲跌幅", f"{norm_b.iloc[-1]:+.2f}%")
                k3.metric("相對強弱差 (A - B)", f"{spread:+.2f}%")

                m_dates = merged.index.strftime("%Y-%m-%d")
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=m_dates, y=norm_a, mode="lines", name=id_a, line=dict(color="#e74c3c", width=2.2)))
                fig_comp.add_trace(go.Scatter(x=m_dates, y=norm_b, mode="lines", name=id_b, line=dict(color="#3498db", width=2.2)))
                fig_comp.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.6)
                fig_comp.update_layout(
                    title=f"{id_a} vs {id_b} 相對強弱走勢 (%)",
                    height=460,
                    hovermode="x unified",
                    yaxis_title="累計漲跌幅 (%)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                fig_comp.update_xaxes(type="category", nticks=8)
                st.plotly_chart(fig_comp, use_container_width=True)
