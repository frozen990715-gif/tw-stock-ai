import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from google import genai
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================= 頁面設定 =================
st.set_page_config(page_title="全方位 AI 量化分析系統", page_icon="📈", layout="wide")

# 初始化 Session State
if "stock_code" not in st.session_state:
    st.session_state.stock_code = "2330"
if "ai_reports" not in st.session_state:
    st.session_state.ai_reports = {}

# ================= 資料抓取與快取函數 =================
PERIOD_MAP = {
    "1個月": ("6mo", 22),
    "3個月": ("1y", 65),
    "6個月": ("1y", 130),
    "1年": ("2y", 250),
    "3年": ("5y", 750),
}

@st.cache_data(ttl=600, show_spinner=False)
def fetch_stock_data(raw_code: str, fetch_period: str):
    code_clean = raw_code.strip().upper()
    # 判斷台股（數字開頭）或美股／大盤指數
    if code_clean.startswith("^") or "." in code_clean or not code_clean[0].isdigit():
        candidates = [code_clean]
    else:
        candidates = [f"{code_clean}.TW", f"{code_clean}.TWO"]

    for code in candidates:
        ticker = yf.Ticker(code)
        df = ticker.history(period=fetch_period)
        if not df.empty:
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

    # 台股優先抓取 Yahoo 奇摩股市新聞與中文名稱
    if stock_id.endswith(".TW") or stock_id.endswith(".TWO"):
        try:
            base_url = "https://tw.stock.yahoo.com"
            url = f"{base_url}/quote/{stock_id}/news"
            headers = {"User-Agent": "Mozilla/5.0"}
            soup = BeautifulSoup(requests.get(url, headers=headers, timeout=8).text, "html.parser")

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

    # 美股／指數或 Yahoo 台股無資料時，改抓 yfinance 內建新聞
    if not news_list:
        try:
            yf_news = yf.Ticker(stock_id).news or []
            for item in yf_news[:5]:
                content = item.get("content", item)
                title = content.get("title")
                link_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
                link = link_obj.get("url") if isinstance(link_obj, dict) else content.get("link", "")
                if title:
                    news_list.append((title, link or ""))
        except Exception:
            pass

    return stock_name, news_list

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    # 均線
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA60"] = data["Close"].rolling(60).mean()

    # 布林通道 (20, 2)
    std20 = data["Close"].rolling(20).std()
    data["BB_UPPER"] = data["MA20"] + 2 * std20
    data["BB_LOWER"] = data["MA20"] - 2 * std20

    # RSI (14)
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["DIF"] = ema12 - ema26
    data["MACD_SIG"] = data["DIF"].ewm(span=9, adjust=False).mean()
    data["MACD_HIST"] = data["DIF"] - data["MACD_SIG"]

    # KD (9, 3, 3)
    low9 = data["Low"].rolling(9).min()
    high9 = data["High"].rolling(9).max()
    rsv = (data["Close"] - low9) / (high9 - low9).replace(0, np.nan) * 100
    k_vals, d_vals = [], []
    k, d = 50.0, 50.0
    for val in rsv:
        if pd.isna(val):
            k_vals.append(np.nan)
            d_vals.append(np.nan)
        else:
            k = (2 / 3) * k + (1 / 3) * val
            d = (2 / 3) * d + (1 / 3) * k
            k_
