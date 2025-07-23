# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta

# --- CONFIG ---
TANK_CAPACITY = 1000
ROOFTOP_AREA = 100
EFFICIENCY = 0.8
N_STEPS = 30
FORECAST_DAYS = 7

# --- HEADER ---
st.set_page_config(page_title="Smart Rainwater System", layout="centered")
st.title("💧 Smart Rainwater Harvesting Dashboard")
st.markdown("""
### 📌 Instructions:
Please upload a `.csv` file with these columns:
- **Date** – format: YYYY-MM-DD  
- **Rainfall_mm** – daily rainfall in millimeters
""")

# ✅ Download button for the correct sample file
try:
    with open("Date.csv", "rb") as file:
        st.download_button(
            label="📥 Click here to download the sample file (Date.csv)",
            data=file,
            file_name="Date.csv",  # keep the same name for upload convenience
            mime="text/csv"
        )
except FileNotFoundError:
    st.warning("⚠️ Sample file not found. Make sure 'Date.csv' is in the same GitHub folder.")




# --- LOAD CSV FILE ---
uploaded_file = st.file_uploader("📁 Upload your rainfall data (.csv)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- VALIDATE CSV FORMAT ---
    if 'Date' not in df.columns or 'Rainfall_mm' not in df.columns:
        st.error("CSV must have 'Date' and 'Rainfall_mm' columns.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Inflow'] = df['Rainfall_mm'] * ROOFTOP_AREA * EFFICIENCY
    df['Consumption'] = np.maximum(100, np.random.normal(250, 50, len(df)))

    # Simulate tank level
    tank_level = []
    level = TANK_CAPACITY * 0.5
    for i in range(len(df)):
        level += df['Inflow'].iloc[i] - df['Consumption'].iloc[i]
        level = min(max(0, level), TANK_CAPACITY)
        tank_level.append(level)
    df['Tank_Level'] = tank_level

    # --- PREPARE DATA FOR LSTM ---
    features = ['Rainfall_mm', 'Inflow', 'Consumption', 'Tank_Level']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    X, y_rain, y_level = [], [], []

    for i in range(N_STEPS, len(scaled)):
        X.append(scaled[i-N_STEPS:i])
        y_rain.append(scaled[i, 0])
        y_level.append(scaled[i, 3])

    X = np.array(X)

    # --- TRAIN MODELS ---
    def build_model():
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    with st.spinner("🧠 Training AI models..."):
        rain_model = build_model()
        rain_model.fit(X, np.array(y_rain), epochs=10, batch_size=32, verbose=0)

    # --- FORECAST RAIN ---
    last_seq = scaled[-N_STEPS:]
    input_data = last_seq.reshape(1, N_STEPS, len(features))
    forecast_rain = []

    for _ in range(FORECAST_DAYS):
        pred_scaled = rain_model.predict(input_data, verbose=0)[0, 0]
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = pred_scaled
        rain = scaler.inverse_transform(dummy)[0, 0]
        forecast_rain.append(max(0, rain))

    # --- GENETIC ALGORITHM ---
    avg_cons = df['Consumption'].mean()
    consumption_forecast = [avg_cons] * FORECAST_DAYS
    current_level = df['Tank_Level'].iloc[-1]

    def optimize_schedule(forecast_rain, consumption, level_start):
        best_score = -np.inf
        best_schedule = []

        for _ in range(200):
            schedule = [random.uniform(100, 300) for _ in range(FORECAST_DAYS)]
            score = 0
            level = level_start
            for i in range(FORECAST_DAYS):
                inflow = forecast_rain[i] * ROOFTOP_AREA * EFFICIENCY
                level += inflow - schedule[i]
                if level > TANK_CAPACITY:
                    score -= (level - TANK_CAPACITY) * 10
                    level = TANK_CAPACITY
                if level < 0:
                    score -= abs(level) * 10
                    level = 0
            if score > best_score:
                best_score = score
                best_schedule = schedule
        return best_schedule

    with st.spinner("⚙️ Optimizing tank usage..."):
        optimized_usage = optimize_schedule(forecast_rain, consumption_forecast, current_level)

    # --- DISPLAY RESULTS ---
    st.success("✅ Forecast and Optimization Complete")

    result_df = pd.DataFrame({
        'Date': pd.date_range(start=df.index[-1] + timedelta(days=1), periods=FORECAST_DAYS),
        'Forecasted_Rainfall_mm': forecast_rain,
        'Estimated_Consumption_L': consumption_forecast,
        'Optimized_Tank_Usage_L': optimized_usage
    })

    st.subheader("📅 Forecast & Usage Schedule")
    st.dataframe(result_df.style.format({"Forecasted_Rainfall_mm": "{:.2f}", "Estimated_Consumption_L": "{:.2f}", "Optimized_Tank_Usage_L": "{:.2f}"}))

    st.subheader("📊 Visual Comparison")
    st.line_chart(result_df.set_index("Date")[["Forecasted_Rainfall_mm", "Optimized_Tank_Usage_L", "Estimated_Consumption_L"]])

else:
    st.info("Upload a CSV file with 'Date' and 'Rainfall_mm' columns to start.")
