import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
# агрессивный тест

# --- 1. ЗАГРУЗКА ДАННЫХ (BIG DATA) ---
TICKER = "GC=F"
DXY_TICKER = "DX-Y.NYB"

print(f"📥 Скачиваем ИСТОРИЧЕСКИЕ данные (15 лет) для {TICKER} и {DXY_TICKER}...")
df = yf.download(TICKER, period="15y", interval="1d", progress=False)
dxy = yf.download(DXY_TICKER, period="15y", interval="1d", progress=False)

if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
if isinstance(dxy.columns, pd.MultiIndex): dxy.columns = dxy.columns.get_level_values(0)


# --- 2. ПРОФЕССИОНАЛЬНЫЕ ФИЧИ (НОВЫЙ НАБОР) ---
def add_ultimate_features(df, dxy_df):
    df = df.copy()

    # Синхронизация
    df.index = df.index.tz_localize(None)
    dxy_df.index = dxy_df.index.tz_localize(None)
    dxy_aligned = dxy_df.reindex(df.index, method='ffill')
    df['Close_DXY'] = dxy_aligned['Close'].bfill()

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

    df['DXY_Ret'] = df['Close_DXY'].pct_change()
    df['Corr_DXY'] = df['Close'].rolling(20).corr(df['Close_DXY']).fillna(0)

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df.dropna()


print("🧠 Генерируем расширенные фичи...")
data = add_ultimate_features(df, dxy)

# Убираем лишние колонки, оставляем только фичи для обучения
drop_cols = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Close_DXY', 'BB_Upper', 'BB_Lower']
feature_names = [col for col in data.columns if col not in drop_cols]

print(f"📊 Итого фич: {len(feature_names)}")
X = data[feature_names]
y = data['Target']

# Разделение
split = int(len(X) * 0.85)  # 85% на обучение, 15% на тест
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# --- 3. ПОИСК ЛУЧШИХ НАСТРОЕК (SUPER GRID) ---
print("🏋️‍♂️ Начинаем МАСШТАБНУЮ тренировку (Gradient Boosting)...")

param_dist = {
    'n_estimators': [100, 200, 300, 500],  # Больше деревьев
    'learning_rate': [0.005, 0.01, 0.05, 0.1],  # Тонкая настройка скорости
    'max_depth': [3, 4, 5, 7, 9],  # Разная глубина мышления
    'subsample': [0.7, 0.8, 0.9, 1.0],  # Защита от переобучения
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model = GradientBoostingClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,  # 50 попыток найти идеал (было 20)
    cv=5,  # 5 проверок на каждом шаге (было 3)
    verbose=1,
    n_jobs=-1,
    scoring='accuracy'
)

random_search.fit(X_train, y_train)

# --- 4. РЕЗУЛЬТАТЫ ---
best_model = random_search.best_estimator_
print(f"\n🏆 ЛУЧШИЕ ПАРАМЕТРЫ: {random_search.best_params_}")

y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"📊 Точность на тесте (новые данные): {acc * 100:.2f}%")
print(classification_report(y_test, y_pred))

# --- 5. СОХРАНЕНИЕ ---
safe_name = TICKER.replace("=", "").replace("-", "")
joblib.dump(best_model, f"robust_model_{safe_name}.pkl")
joblib.dump(feature_names, f"robust_features_{safe_name}.pkl")

print(f"✅ SUPER AI сохранен! Теперь ОБЯЗАТЕЛЬНО обнови app.py!")