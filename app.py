import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from google import genai
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="台股 AI 量化分析系統", page_icon="📈", layout="centered")
st.title("📈 台股 AI 量化分析系統")
st.caption("結合互動式 K 線圖（MA20、成交量、RSI）、財經新聞爬蟲與 Gemini AI 短評")

# 1. 使用者輸入介面
col_input, col_btn = st.columns([3, 1])
with col_input:
    raw_code = st.text_input("輸入台股代號（例如 2454、2330、0050）", value="2454").strip().upper()
with col_btn:
    st.write("")
    st.write("")
    run_btn = st.button("開始分析", type="primary", use_container_width=True)

if run_btn and raw_code:
    with st.spinner(f"正在抓取 {raw_code} 股價與最新新聞..."):
        # 自動判斷上市 (.TW) 或上櫃 (.TWO)
        if raw_code.endswith(".TW") or raw_code.endswith(".TWO"):
            candidates = [raw_code]
        else:
            candidates = [f"{raw_code}.TW", f"{raw_code}.TWO"]

        stock_id, df = None, None
        for code in candidates:
            temp_df = yf.Ticker(code).history(period="6mo")
            if not temp_df.empty:
                stock_id, df = code, temp_df
                break

        if df is None:
            st.error(f"找不到代號「{raw_code}」的股價資料，請確認代號是否正確。")
            st.stop()

        df.index = df.index.tz_localize(None)

        # 2. 計算技術指標
        df["MA20"] = df["Close"].rolling(window=20).mean()
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        latest_date = df.index[-1].strftime("%Y-%m-%d")
        latest_close = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2] if len(df) > 1 else latest_close
        price_diff = latest_close - prev_close
        latest_ma20 = df["MA20"].iloc[-1]
        latest_rsi = df["RSI"].iloc[-1]

        # 3. 抓取 Yahoo 股市新聞（含原文連結）與中文名稱
        base_url = "https://tw.stock.yahoo.com"
        url = f"{base_url}/quote/{stock_id}/news"
        headers = {"User-Agent": "Mozilla/5.0"}
        soup = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")

        stock_name = stock_id
        if soup.title and "(" in soup.title.text:
            stock_name = soup.title.text.split("(")[0].strip()

        news_list = []
        seen_titles = set()
        for tag in soup.find_all("h3"):
            title = tag.get_text(strip=True)
            if len(title) > 8 and "個股相關新聞與公告" not in title and title not in seen_titles:
                link_tag = tag.find("a") or tag.find_parent("a")
                href = link_tag.get("href", "") if link_tag else ""
                full_url = urljoin(base_url, href) if href else ""
                news_list.append((title, full_url))
                seen_titles.add(title)
                if len(news_list) >= 5:
                    break

    # 4. 顯示數據卡片
    st.subheader(f"{stock_name} ({stock_id})｜資料日期：{latest_date}")
    m1, m2, m3 = st.columns(3)
    m1.metric("最新收盤價", f"{latest_close:.2f}", f"{price_diff:+.2f}")
    m2.metric("20 日均線 (MA20)", f"{latest_ma20:.2f}")
    m3.metric("RSI (14)", f"{latest_rsi:.2f}")

    # 5. 繪製互動式三層 K 線圖（Plotly）
    dates = df.index.strftime("%Y-%m-%d")
    vol_colors = ["#e74c3c" if c >= o else "#2ecc71" for c, o in zip(df["Close"], df["Open"])]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.18, 0.24],
        subplot_titles=(f"{stock_name} ({stock_id}) K 線與 20 日均線", "成交量", "RSI (14)")
    )

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
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["MA20"],
            mode="lines",
            name="MA20 (月線)",
            line=dict(color="#ff7f0e", width=1.8),
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=dates,
            y=df["Volume"],
            name="成交量",
            marker_color=vol_colors,
            opacity=0.8,
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["RSI"],
            mode="lines",
            name="RSI (14)",
            line=dict(color="#9467bd", width=1.6),
        ),
        row=3, col=1
    )
    fig.add_hline(y=70, line_dash="dot", line_color="red", opacity=0.7, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.7, row=3, col=1)

    fig.update_layout(
        height=720,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=55, b=10),
    )
    fig.update_xaxes(type="category", nticks=8)
    fig.update_yaxes(title_text="價格 (TWD)", row=1, col=1)
    fig.update_yaxes(title_text="量能", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # 6. 顯示近期新聞（可點擊連結）
    with st.expander("📰 查看近期相關財經新聞（點擊標題看原文）", expanded=True):
        if news_list:
            for i, (title, link) in enumerate(news_list, 1):
                if link:
                    st.markdown(f"{i}. [{title}]({link})")
                else:
                    st.write(f"{i}. {title}")
        else:
            st.write("暫時沒有抓到新聞標題，將改以純技術面進行分析。")

    # 7. 呼叫 Gemini AI 產出報告
    with st.spinner("正在請 Gemini AI 撰寫量化短評..."):
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            client = genai.Client(api_key=api_key)
            news_text = "\n".join(f"- {title}" for title, _ in news_list) if news_list else "（無最新新聞）"
            prompt = f"""
以下是 {stock_id} {stock_name} 的最新資訊（資料日期：{latest_date}）：
- 最新收盤價：{latest_close:.2f}
- 20 日均線 (MA20)：{latest_ma20:.2f}
- 相對強弱指標 (RSI)：{latest_rsi:.2f}

近期相關新聞：
{news_text}

請扮演專業的台股量化分析師，綜合上述「技術面數據」與「新聞面消息」，撰寫約 150 字的重點短評：
1. 說明目前股價與均線、RSI 所呈現的多空狀態。
2. 統整近期新聞帶來的潛在利多或利空。
3. 給出後續客觀的觀察重點。
"""
            ai_response = client.models.generate_content(
                model="models/gemini-3.1-flash-lite",
                contents=prompt
            )
            st.subheader("🤖 AI 量化分析報告")
            st.info(ai_response.text)
        except Exception as e:
            st.error(f"AI 分析發生錯誤：{e}")
