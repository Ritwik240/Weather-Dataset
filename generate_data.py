import requests
import pandas as pd
from io import StringIO
import sys
import numpy as np
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
LAT, LON = 28.6139, 77.2090  # Delhi coordinates
START = "2015-01-01"

# --- DYNAMIC DATE LOGIC ---
yesterday = datetime.now() - timedelta(days=1)
END = yesterday.strftime('%Y-%m-%d')

print(f"--- Dynamic Data Generation ---")
print(f"Fetching data from {START} to {END}")

OUTPUT_FILENAME = "Unified_Weather_Dataset_Latest.json"

# --- 2. NASA POWER API ---
nasa_params = "ALLSKY_SFC_UV_INDEX,T2M,RH2M,WS10M,PRECTOTCORR"
nasa_url = (
    f"https://power.larc.nasa.gov/api/temporal/daily/point?"
    f"parameters={nasa_params}"
    f"&start={START.replace('-', '')}&end={END.replace('-', '')}"
    f"&latitude={LAT}&longitude={LON}"
    f"&community=RE&format=CSV"
)

print(f" Fetching NASA POWER data ({START} to {END})...")
try:
    nasa_resp = requests.get(nasa_url, timeout=180)
    if nasa_resp.status_code == 200:
        nasa_text = nasa_resp.text
        lines = nasa_text.splitlines()
        data_start = next(i for i, line in enumerate(lines) if line.startswith("YEAR"))
        df_nasa = pd.read_csv(StringIO("\n".join(lines[data_start:])))

        df_nasa.replace(-999, pd.NA, inplace=True)

        df_nasa.rename(columns={"YEAR": "Year", "MO": "Month", "DY": "Day"}, inplace=True)
        df_nasa["Date"] = pd.to_datetime(df_nasa[["Year", "Month", "Day"]])
        df_nasa = df_nasa[
            ["Date", "ALLSKY_SFC_UV_INDEX", "T2M", "RH2M", "WS10M", "PRECTOTCORR"]
        ].rename(
            columns={
                "ALLSKY_SFC_UV_INDEX": "UV_Index",
                "T2M": "Temp_C_NASA",
                "RH2M": "Humidity_%_NASA",
                "WS10M": "WindSpeed_m/s_NASA",
                "PRECTOTCORR": "Rainfall_mm_NASA",
            }
        )
        print("✅ NASA data cleaned and ready.")
    else:
        print(f"❌ NASA data fetch failed. Status: {nasa_resp.status_code}")
        df_nasa = pd.DataFrame()
except Exception as e:
    print(f"❌ NASA fetch error: {e}")
    df_nasa = pd.DataFrame()

# --- 3. Open-Meteo API ---
openmeteo_url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LAT}&longitude={LON}"
    f"&start_date={START}&end_date={END}"
    f"&daily=temperature_2m_mean,relative_humidity_2m_mean,"
    f"precipitation_sum,windspeed_10m_mean"
    f"&timezone=auto"
)

print(f" Fetching Open-Meteo weather data ({START} to {END})...")
try:
    om_resp = requests.get(openmeteo_url, timeout=180)
    if om_resp.status_code == 200:
        om_data = om_resp.json()
        df_om = pd.DataFrame(om_data["daily"])
        df_om["Date"] = pd.to_datetime(df_om["time"])
        df_om.rename(
            columns={
                "temperature_2m_mean": "Temp_C_OM",
                "relative_humidity_2m_mean": "Humidity_%_OM",
                "precipitation_sum": "Rainfall_mm_OM",
                "windspeed_10m_mean": "WindSpeed_m/s_OM",
            },
            inplace=True,
        )
        print("✅ Open-Meteo weather data fetched successfully.")
    else:
        print(f"❌ Open-Meteo weather fetch failed. Status: {om_resp.status_code}")
        df_om = pd.DataFrame()
except Exception as e:
    print(f"❌ Open-Meteo fetch error: {e}")
    df_om = pd.DataFrame()

# --- 4. Combine All Sources (No AQI) ---
print("\n Combining all datasets...")
if df_nasa.empty and df_om.empty:
    print("❌ Critical Error: Both NASA and Open-Meteo data fetching failed. Exiting.")
    sys.exit()

df_final = df_nasa.copy()

# Merge Open-Meteo
if not df_om.empty:
    df_final = pd.merge(df_final, df_om, on="Date", how="outer")
else:
    print("⚠️ Open-Meteo data is missing. Proceeding with NASA only.")

# --- 5. Unify and Clean ---
print(" Unifying data columns...")
df_final = df_final.replace({np.nan: pd.NA})

df_final["Temperature_C"] = df_final[["Temp_C_NASA", "Temp_C_OM"]].mean(axis=1, skipna=True)
df_final["Humidity_%"] = df_final[["Humidity_%_NASA", "Humidity_%_OM"]].mean(axis=1, skipna=True)
df_final["Rainfall_mm"] = df_final[["Rainfall_mm_NASA", "Rainfall_mm_OM"]].mean(axis=1, skipna=True)
df_final["WindSpeed_m/s"] = df_final[["WindSpeed_m/s_NASA", "WindSpeed_m/s_OM"]].mean(axis=1, skipna=True)

final_cols = ["Date", "UV_Index", "Temperature_C", "Humidity_%", "Rainfall_mm", "WindSpeed_m/s"]
df_final = df_final[final_cols]

df_final.sort_values("Date", inplace=True)

# --- FIX for interpolate() ---
df_final = df_final.set_index('Date')

for col in df_final.columns:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

df_final = df_final.interpolate(method='time', limit_direction='both')

df_final = df_final.bfill().ffill()
df_final = df_final.reset_index()

# --- 6. Save JSON ---
print(f"\n Saving final unified dataset as '{OUTPUT_FILENAME}'")
df_final.to_json(
    OUTPUT_FILENAME,
    orient="records",
    indent=4,
    date_format="iso"
)

print("\n🎉 Success! Final unified dataset saved.")
print(" Columns included:", list(df_final.columns))
print("\n Sample preview:")
print("--- First 5 Rows ---")
print(df_final.head())
print("\n--- Last 5 Rows ---")
print(df_final.tail())