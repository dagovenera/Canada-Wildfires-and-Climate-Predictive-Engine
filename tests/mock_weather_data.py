import pandas as pd
import numpy as np

print("🛠️ Constructing synthetic 6-year climate matrix array (2021-2026)...")

# Generate continuous dates with every single day between Jan1/2021 and Dec31/2026
date_range = pd.date_range(start="2021-01-01", end="2026-12-31", freq="D")
total_days = len(date_range)

# Seed the generator so results are perfectly reproducible
np.random.seed(42)

# Simulate realistic weather variables using normal/binomial distributions
# Simulate summer temperature spikes to give ML model clear patterns to learn
months = date_range.month
base_temp = np.where((months >= 6) & (months <= 8), 28.5, 12.0) 
noise = np.random.normal(0, 4.5, total_days)
mean_temps = base_temp + noise

# Simulate precipitation (lots of dry summer days, rare heavy rainfall)
precip_chance = np.where((months >= 6) & (months <= 8), 0.15, 0.40)
has_precip = np.random.binomial(1, precip_chance)
precip_amount = has_precip * np.random.exponential(8.0, total_days)

# Simulate continuous sustained maximum wind gust speeds
wind_gusts = np.random.weibull(2.5, total_days) * 22.0

# Build the complete data frame
mock_df = pd.DataFrame({
    "Date/Time": date_range,
    "Station Name": "KELOWNA UBCO",
    "Latitude (y)": 49.939,
    "Longitude (x)": -119.395,
    "Mean Temp (°C)": np.round(mean_temps, 1),
    "Total Precip (mm)": np.round(precip_amount, 1),
    "Spd of Max Gust (km/h)": np.round(wind_gusts, 1)
})

# Export analysis-ready file with weather data frame
mock_df.to_csv("raw_eccc_weather.csv", index=False)
print(f"🎉 Success! Generated {total_days} clean rows.")
print("📁 Saved local file as: raw_eccc_weather.csv")
