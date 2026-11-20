import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from google import genai
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="台股 AI 量化分析系統", page_icon="📈", layout="centered")
st.title("📈 台股 AI 量化分析系統")
st.caption("結合即時技術指標（K 線、MA20、RSI）、財經新聞爬蟲與 Gemini AI 短評")

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

        # 3. 抓取 Yahoo 股市新聞與中文名稱
        url = f"https://tw.stock.yahoo.com/quote/{stock_id}/news"
        headers = {"User-Agent": "Mozilla/5.0"}
        soup = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")

        stock_name = stock_id
        if soup.title and "(" in soup.title.text:
            stock_name = soup.title.text.split("(")[0].strip()

        news_list = []
        for tag in soup.find_all("h3"):
            title = tag.get_text(strip=True)
            if len(title) > 8 and title not in news_list:
                news_list.append(title)
                if len(news_list) >= 5:
                    break

    # 4. 顯示數據卡片
    st.subheader(f"{stock_name} ({stock_id})｜資料日期：{latest_date}")
    m1, m2, m3 = st.columns(3)
    m1.metric("最新收盤價", f"{latest_close:.2f}", f"{price_diff:+.2f}")
    m2.metric("20 日均線 (MA20)", f"{latest_ma20:.2f}")
    m3.metric("RSI (14)", f"{latest_rsi:.2f}")

    # 5. 繪製 K 線、成交量與 RSI 圖表
    mc = mpf.make_marketcolors(up="#e74c3c", down="#2ecc71", edge="inherit", wick="inherit", volume="inherit")
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", gridcolor="#cccccc", y_on_right=False)
    apds = [
        mpf.make_addplot(df["MA20"], panel=0, color="#ff7f0e", width=1.6, secondary_y=False),
        mpf.make_addplot(df["RSI"], panel=2, color="#9467bd", width=1.5, ylabel="RSI", ylim=(0, 100), secondary_y=False),
        mpf.make_addplot(pd.Series(70, index=df.index), panel=2, color="red", linestyle=":", width=1, secondary_y=False),
        mpf.make_addplot(pd.Series(30, index=df.index), panel=2, color="green", linestyle=":", width=1, secondary_y=False),
    ]
    fig, _ = mpf.plot(
        df,
        type="candle",
        style=style,
        volume=True,
        addplot=apds,
        panel_ratios=(4, 1.2, 1.5),
        title=f"\n{stock_id} - K-Line, MA20 (Orange) & RSI",
        ylabel="Price (TWD)",
        ylabel_lower="Volume",
        figsize=(10, 7),
        tight_layout=True,
        returnfig=True,
    )
    st.pyplot(fig)
    plt.close(fig)

    # 6. 顯示近期新聞
    with st.expander("📰 查看近期相關財經新聞", expanded=True):
        if news_list:
            for i, title in enumerate(news_list, 1):
                st.write(f"{i}. {title}")
        else:
            st.write("暫時沒有抓到新聞標題，將改以純技術面進行分析。")

    # 7. 呼叫 Gemini AI 產出報告
    with st.spinner("正在請 Gemini AI 撰寫量化短評..."):
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            client = genai.Client(api_key=api_key)
            news_text = "\n".join(f"- {t}" for t in news_list) if news_list else "（無最新新聞）"
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
