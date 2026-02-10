# ⚡ QUANTUM AI TRADER

**Quantum AI Trader** is a high-performance algorithmic trading terminal that leverages Machine Learning (Gradient Boosting) to forecast price movements for EUR/USD and GOLD (XAU/USD). This project integrates quantitative analysis, financial data science, and a real-time interactive dashboard.

[Image of an algorithmic trading system architecture showing data input, feature engineering, ML model prediction, and signal output]

---

## 🚀 Key Features

* **🤖 Specialized Dual-Model System:** Individualized ML models for the Forex market and Precious Metals.
* **📊 22-Factor Technical Analysis:** Real-time processing of RSI, MACD, ATR, Bollinger Bands, and DXY (Dollar Index) correlation.
* **⏳ Optimized for Swing Trading:** Trained on the 1D (Daily) timeframe to provide high-probability forecasts for 24–48 hour windows.
* **🔴 Live Trading Mode:** Automated data fetching and prediction cycles every 5 seconds via Yahoo Finance API.
* **🛡️ Integrated Risk Management:** Dynamic calculation of Entry, Stop-Loss, and Take-Profit levels based on market volatility (ATR).
---

## 🛠 Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** Scikit-learn (Gradient Boosting Classifier)
* **Dashboard:** Streamlit
* **Data Source:** YFinance API
* **Visualization:** Plotly (Interactive Candlestick Charts)
* **Data Processing:** Pandas, NumPy

[Image of a data preprocessing pipeline for financial machine learning]

---

## 📂 Project Structure

* `app.py` — The primary interactive terminal and UI.
* `robust_model_*.pkl` — Serialized weights for the trained AI models.
* `robust_features_*.pkl` — Metadata for model-specific feature alignment.
* `requirements.txt` — Project dependencies.

---

## ⚙️ Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/quantum-ai-trader.git](https://github.com/your-username/quantum-ai-trader.git)
   cd quantum-ai-trader
