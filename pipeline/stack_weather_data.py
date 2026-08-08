import pandas as pd

station_id = "1123939"
year_start = 2017
year_end = 2023
dfs = []

for year in range(year_start, year_end+1):
    df = pd.read_csv(f"Weather_Data/en_climate_daily_BC_{station_id}_{year}_P1D.csv")
    # Replace NaN and '<31' with 25 in Spd of Max Gust (Km/h)
    df["Spd of Max Gust (km/h)"] = df["Spd of Max Gust (km/h)"].replace('<31', 25)
    df["Spd of Max Gust (km/h)"] = df["Spd of Max Gust (km/h)"].fillna(25)

    #select data from July and August only
    df=df[(df["Month"] > 6) & (df["Month"] < 9)]

    dfs.append(df)

# Stack them vertically into one master dataframe
raw_eccc_weather = pd.concat(dfs, ignore_index=True)

# Save it as the uniform filename your pipeline expects
raw_eccc_weather.to_csv("raw_eccc_weather.csv", index=False)
print("Master weather dataset compiled successfully!")
