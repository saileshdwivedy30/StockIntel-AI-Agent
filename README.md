

# 📈 StockIntel AI - Stock Analysis AI Agent  

An **AI-powered stock analysis agent** that allows users to enter a **company name** or **stock ticker** and get a comprehensive analysis based on **real-time financial data, technical indicators, and AI-driven insights**. The AI agent fetches stock data, performs **technical analysis**, and assigns a rating based on financial health and market trends.  

---

## 🎥 **Demo**  
Video Link: https://www.youtube.com/watch?v=oYHgGF4P0tQ

---

## 🤖 **Tech Stack**  
- **Python**  
- **Pydantic AI** (Python agent framework)
- **Streamlit** (UI Framework)  
- **Yahoo Finance (`yfinance`)** (Stock Data)  
- **DuckDuckGo Search (`duckduckgo-search`)** (Symbol Lookup)  
- **OpenAI GPT-4o** (Stock Analysis AI)  

---

## 🚀 **Features**  
- **AI-Powered Stock Analysis**: Uses **GPT-4o** to provide insightful ratings and explanations.  
- **Company Name to Stock Symbol Conversion**: Enter a company name (e.g., *Apple*), and the AI agent finds the corresponding stock symbol using **DuckDuckGo Search**.  
- **Stock Ticker Lookup**: Directly enter a stock ticker (e.g., *AAPL*).  
- **Real-Time Financial Data**: Fetches stock data using **Yahoo Finance (`yfinance`)**.  
- **Technical Analysis**:
  - Moving Averages (**20-day, 50-day, 200-day**)  
  - **RSI (Relative Strength Index)**  
  - **Quarterly & Annual Financial Reports**  
- **AI-Driven Stock Rating**: Evaluates stocks based on financial and technical factors and assigns a rating.  

---

## 📦 **Installation**  

#### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/saileshdwivedy30/stockintel-ai.git
cd stockintel-ai
```

#### 2️⃣ **Create a Virtual Environment & Activate it**  
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

#### 3️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

#### 4️⃣ **Set Up Environment Variables**  
Create a `.env` file in the project root and add:  
```env
OPENAI_API_KEY=your_openai_api_key_here
```

#### 5️⃣ **Run the Application**  
```bash
streamlit run main.py
```
Then, open the **localhost link** in your browser.

---

## 🛡 **License**  
This project is licensed under the **MIT License**. 

---

## 👨‍💻 **Author**  
🔹 **Sailesh Dwivedy**  
🔹 [GitHub Profile](https://github.com/saileshdwivedy30)  
🔹 [LinkedIn Profile](https://www.linkedin.com/in/saileshdwivedy/)  

