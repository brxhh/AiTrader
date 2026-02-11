import datetime

import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings

warnings.filterwarnings("ignore")

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="QUANT TERMINAL", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# --- CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #131722; color: #d1d4dc; font-family: -apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif; }
    h1, h2, h3 { color: #d1d4dc !important; }
    [data-testid="stSidebar"] { background-color: #1e222d; border-right: 1px solid #2a2e39; }
    div[data-testid="metric-container"] { background-color: #2a2e39; border: 1px solid #363a45; padding: 10px; border-radius: 4px; }
    [data-testid="stMetricLabel"] { color: #787b86 !important; }
    [data-testid="stMetricValue"] { color: #d1d4dc !important; }
    .stButton>button { background-color: #2962ff; color: white; border: none; border-radius: 4px; font-weight: 600; }
    .stButton>button:hover { background-color: #1e53e5; }

    /* Скрываем только лишнее, но оставляем HEADER (верхнюю полоску), чтобы была кнопка меню */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    </style>
""", unsafe_allow_html=True)


# --- ЛОГИКА СТАТУСА РЫНКА ---
def get_market_status():
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    weekday = now_utc.weekday()  # 0=Пн, 4=Пт, 5=Сб, 6=Вс
    hour = now_utc.hour

    # Рынки закрываются в пятницу в 22:00 UTC и открываются в воскресенье в 22:00 UTC
    if weekday == 5:  # Суббота
        return "CLOSED", "#ff5555"
    if weekday == 4 and hour >= 22:  # Пятница вечер
        return "CLOSING", "#ffa500"
    if weekday == 6 and hour < 22:  # Воскресенье день
        return "CLOSED", "#ff5555"

    return "OPEN", "#00ff00"


status_text, status_color = get_market_status()

# Обновленный заголовок
st.title("⚡ QUANTUM AI TRADER")
st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="color: {status_color}; font-weight: bold;">● MARKET {status_text}</span>
        <span style="color: #787b86;">| MODE: REAL-TIME</span>
        <span style="color: #787b86;">| UTC: {datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M')}</span>
    </div>
""", unsafe_allow_html=True)

if status_text == "CLOSED":
    st.warning(
        "⚠️ **MARKETS ARE CLOSED.** Данные на графике — это цены закрытия пятницы. Новые сигналы могут быть неточными до открытия торгов.")

st.divider()

# --- САЙДБАР ---
st.sidebar.title("⚙️ CONTROL PANEL")

ASSETS = {
    "EUR/USD": "EURUSD=X",
    "GOLD (XAU/USD)": "GC=F"
}

selected_name = st.sidebar.selectbox("ASSET CLASS", list(ASSETS.keys()), index=0)
TICKER = ASSETS[selected_name]
DXY_TICKER = "DX-Y.NYB"

st.sidebar.markdown("---")
st.sidebar.subheader("NEWS FEED")

surprise_val = st.sidebar.number_input(
    "Economic Surprise (USD):",
    min_value=-5.0, max_value=5.0, value=0.00, step=0.1, format="%.2f",
    help="Actual - Forecast"
)

st.sidebar.markdown("---")
run_live = st.sidebar.toggle("🔴 LIVE TRADING MODE", value=False)


# --- ФИЧИ ---
def add_ultimate_features(df, dxy_df):
    df = df.copy()

    df.index = df.index.tz_localize(None)
    if dxy_df is not None:
        dxy_df.index = dxy_df.index.tz_localize(None)
        dxy_aligned = dxy_df.reindex(df.index, method='ffill')
        df['Close_DXY'] = dxy_aligned['Close']
        df['Close_DXY'] = df['Close_DXY'].fillna(method='bfill')

    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'Lag_{lag}'] = df['Log_Ret'].shift(lag)

    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Dist_EMA'] = (df['Close'] - df['EMA_50']) / df['EMA_50']

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    high_low = df['High'] - df['Low']
    true_range = np.maximum(high_low, np.abs(df['High'] - df['Close'].shift()))
    df['ATR'] = true_range.rolling(14).mean() / df['Close']

    sma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = sma20 + 2 * std20
    df['BB_Lower'] = sma20 - 2 * std20
    df['BB_Pos'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    df['Body_Size'] = np.abs(df['Close'] - df['Open']) / df['Open']
    df['Shadow_Upper'] = (df['High'] - np.maximum(df['Close'], df['Open'])) / df['Open']
    df['Shadow_Lower'] = (np.minimum(df['Close'], df['Open']) - df['Low']) / df['Open']

    if dxy_df is not None:
        df['DXY_Ret'] = df['Close_DXY'].pct_change()
        df['Corr_DXY'] = df['Close'].rolling(20).corr(df['Close_DXY']).fillna(0)

    return df


# --- ГРАФИК ---
def plot_chart(df, ticker_name, unique_id):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=(f'{ticker_name}', 'Volume'),
                        row_width=[0.2, 0.7])

    # 1. Свечи
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350', name='Price'
    ), row=1, col=1)

    # 2. EMA
    if 'EMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', line=dict(color='#ff9800', width=1), name='EMA 50'),
            row=1, col=1)

    # 3. BB
    if 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(255, 255, 255, 0.2)', width=1),
                                 name='BB Upper', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(255, 255, 255, 0.2)', width=1),
                                 name='BB Lower', showlegend=False), row=1, col=1)

    # 4. Объемы
    colors = ['#26a69a' if row['Open'] - row['Close'] >= 0 else '#ef5350' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # Настройки
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#131722', plot_bgcolor='#131722',
        height=750, margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness=0.05,
        dragmode='pan',
        hovermode='x unified',

        # --- ФИКСАЦИЯ СОСТОЯНИЯ ---
        uirevision=unique_id,

        xaxis=dict(
            rangebreaks=[dict(bounds=["sat", "mon"])],
            range=[df.index[-90], df.index[-1]],
            showgrid=True, gridwidth=1, gridcolor="#2a2e39"
        ),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="#2a2e39")
    )

    config = {
        'scrollZoom': True,
        'displayModeBar': False,
        'displaylogo': False
    }

    chart_key = f"chart_{unique_id}"
    st.plotly_chart(fig, use_container_width=True, config=config, key=chart_key)


# --- ОСНОВНАЯ ЛОГИКА ---
if run_live or st.button("RUN ANALYSIS 🚀"):

    safe_name = TICKER.replace("=", "").replace("-", "")

    model_file = f"robust_model_{safe_name}.pkl"
    features_file = f"robust_features_{safe_name}.pkl"

    try:
        model = joblib.load(model_file)
        feature_names = joblib.load(features_file)
    except FileNotFoundError:
        st.error(f"⚠️ Модель для {selected_name} не найдена! ({model_file})")
        st.stop()

    with st.spinner('Thinking...' if not run_live else None):
        df = yf.download(TICKER, period="2y", interval="1d", progress=False)
        dxy = yf.download(DXY_TICKER, period="2y", interval="1d", progress=False)

    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if isinstance(dxy.columns, pd.MultiIndex): dxy.columns = dxy.columns.get_level_values(0)

    processed_data = add_ultimate_features(df, dxy)
    last_row = processed_data.iloc[-1].copy()
    last_row['surprise'] = surprise_val

    days = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
    try:
        current_day = days[int(last_row['DayOfWeek'])]
    except:
        current_day = "UNKNOWN"

    if run_live:
        st.info(f"📅 LIVE DATA | {current_day} | AUTO-REFRESH 5s")
    else:
        st.info(f"📅 TODAY IS: **{current_day}**")

    # ВЫЗЫВАЕМ ГРАФИК С УНИКАЛЬНЫМ ID АКТИВА
    plot_chart(df, selected_name, safe_name)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PRICE", f"{last_row['Close']:.4f}")
    c2.metric("RSI (14)", f"{last_row['RSI']:.1f}")
    c3.metric("ATR", f"{last_row['ATR']:.4f}")
    c4.metric("DXY CORR", f"{last_row['Corr_DXY']:.2f}")

    try:
        final_input = pd.DataFrame(index=[0])
        for col in feature_names:
            final_input[col] = last_row.get(col, 0.0)

        pred = model.predict(final_input)[0]
        prob = model.predict_proba(final_input)[0]
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    st.markdown("---")
    atr = last_row['ATR'] * last_row['Close']
    price = last_row['Close']

    col_sig, col_plan = st.columns([1, 2])

    with col_sig:
        if pred == 1:
            st.success("LONG / BUY")
            st.metric("CONFIDENCE", f"{prob[1] * 100:.2f}%")
            sl, tp = price - 2 * atr, price + 3 * atr
            bg, border = "#0f3d0f", "#00ff00"
        else:
            st.error("SHORT / SELL")
            st.metric("CONFIDENCE", f"{prob[0] * 100:.2f}%")
            sl, tp = price + 2 * atr, price - 3 * atr
            bg, border = "#3d0f0f", "#ff0000"

    with col_plan:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-around; background-color:{bg}; padding:15px; border-radius:10px; border:1px solid {border};">
            <div style="text-align:center;"><h4 style="color:#bbb; margin:0;">ENTRY</h4><h2 style="color:white; margin:0;">{price:.4f}</h2></div>
            <div style="text-align:center;"><h4 style="color:#ff5555; margin:0;">STOP</h4><h2 style="color:#ff5555; margin:0;">{sl:.4f}</h2></div>
            <div style="text-align:center;"><h4 style="color:#55ff55; margin:0;">TAKE</h4><h2 style="color:#55ff55; margin:0;">{tp:.4f}</h2></div>
        </div>""", unsafe_allow_html=True)

    if run_live:
        time.sleep(5)
        st.rerun()

else:
    st.info("👈 Нажми **RUN** или включи **LIVE MODE**.")