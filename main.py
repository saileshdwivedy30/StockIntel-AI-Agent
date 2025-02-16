import asyncio
import streamlit as st
import numpy as np
import os
from typing import TypedDict, List
from dotenv import load_dotenv
import yfinance as yf
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ModelSettings
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()

# Fetch API Key from .env
api_key = os.getenv("OPENAI_API_KEY")

# Ensure API key is set
if not api_key:
    st.error("OPENAI_API_KEY is missing. Please check your .env file.")
    st.stop()


# Data structure for Moving Averages
class TechnicalIndicators(TypedDict):
    ma_20: List[float]
    ma_50: List[float]
    ma_200: List[float]


# AI Agent
agent = Agent(
    "openai:gpt-4o",
    deps_type=str,
    model_settings=ModelSettings(temperature=0, api_key=api_key),
    system_prompt="""
    You are a sophisticated AI-powered stock rating agent designed to assess financial reports, historical price movements, and key technical indicators. Your primary objective is to evaluate stocks and assign a rating on a scale from Strong Buy (A) to Strong Sell (E), along with a clear and well-reasoned explanation.

    When analyzing a stock, consider the following key factors:
    - Revenue growth  
    - Profitability  
    - Historical price trends  
    - Technical indicators  

    Based on your assessment, assign one of the following ratings and provide a detailed justification:

    A - **Strong Buy**: The stock appears undervalued with strong growth potential, robust financials, and positive market momentum.  
    B - **Buy**: The stock has solid fundamentals and favorable technical indicators but may present some risks or uncertainties.  
    C - **Hold**: The stock is fairly valued, with a mix of positive and negative signals from both fundamental and technical analysis. Holding is recommended unless significant catalysts emerge.  
    D - **Sell**: Weak fundamentals, downward trends, or negative market sentiment suggest potential downside risk.  
    E - **Strong Sell**: The stock faces serious financial or structural issues, high downside risk, or bearish trends indicating further losses.  
    """,
)


# Search for the stock ticker using DuckDuckGo
def get_stock_symbol(company_name: str) -> str:
    """Searches DuckDuckGo for a stock ticker based on the company name."""
    query = f"{company_name} stock symbol site:finance.yahoo.com"

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    for result in results:
        if "finance.yahoo.com" in result["href"]:
            # Extract stock symbol from the Yahoo Finance URL
            url_parts = result["href"].split("/")
            for part in url_parts:
                if part.isupper() and len(part) <= 5:  # Stock symbols are uppercase, usually 1-5 characters
                    return part

    return None  # Return None if no valid symbol is found



# Stock Data Fetching
@agent.tool
def fetch_stock_info(ctx: RunContext[str]):
    stock = yf.Ticker(ctx.deps)
    info = stock.info

    return {
        "long_name": info.get("longName", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "sector": info.get("sector", "N/A"),
    }


@agent.tool
def fetch_quarterly_financials(ctx: RunContext[str]):
    stock = yf.Ticker(ctx.deps)
    return stock.quarterly_financials.T[["Total Revenue", "Net Income"]].to_csv()


@agent.tool
def fetch_annual_financials(ctx: RunContext[str]):
    stock = yf.Ticker(ctx.deps)
    return stock.financials.T[["Total Revenue", "Net Income"]].to_csv()


@agent.tool
def fetch_weekly_price_history(ctx: RunContext[str]):
    stock = yf.Ticker(ctx.deps)
    return stock.history(period="1y", interval="1wk").to_csv()


@agent.tool
def calculate_rsi_weekly(ctx: RunContext[str]):
    stock = yf.Ticker(ctx.deps)
    data = stock.history(period="1y", interval="1wk")

    if data.empty:
        return st.error(f"Failed to fetch data for {ctx.deps}")

    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


# Moving Averages Function
@agent.tool
def calculate_moving_averages(ctx: RunContext[str]):
    stock = yf.Ticker(ctx.deps)
    data = stock.history(period="1y", interval="1d")  # Fetch daily price data for 1 year

    if data.empty:
        return st.error(f"Failed to fetch data for {ctx.deps}")

    closing_prices = data["Close"].values  # Extract closing prices

    ma_20 = np.convolve(closing_prices, np.ones(20) / 20, mode="valid").tolist()
    ma_50 = np.convolve(closing_prices, np.ones(50) / 50, mode="valid").tolist()
    ma_200 = np.convolve(closing_prices, np.ones(200) / 200, mode="valid").tolist()

    return TechnicalIndicators(
        ma_20=ma_20,
        ma_50=ma_50,
        ma_200=ma_200,
    )

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


st.title("📈 StockIntel AI - Stock Analysis AI Agent")

st.write("Enter a **company name** (e.g., Apple, Tesla) or a **stock ticker** (e.g., AAPL, TSLA).")

# User input field
user_input = st.text_input("Enter company name or stock ticker:")

# Validate input and fetch stock symbol if a company name is entered
if user_input:
    user_input = user_input.strip()

    # Check if the input is already a valid stock symbol
    if len(user_input) <= 5 and user_input.isupper():
        selected_symbol = user_input  # Assume it's a ticker
    else:
        st.write(f"🔍 Searching for stock ticker for **{user_input}**...")
        selected_symbol = get_stock_symbol(user_input)

    # When no valid symbol is found
    if not selected_symbol:
        st.error(f"Unable to find a stock symbol for '{user_input}'. Please try again.")
    else:
        st.success(f"Found stock symbol: **{selected_symbol}**")
        result = agent.run_sync(f"Analyze this stock", deps=selected_symbol)
        st.markdown(result.data)

