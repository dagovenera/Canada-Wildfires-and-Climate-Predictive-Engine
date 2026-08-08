import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# ==========================================
# STEP 1: INGEST & CLEAN ECCC WEATHER CSV
# ==========================================
print("🔄 Loading ECCC Weather Data...")
# Load raw daily observations
weather_df = pd.read_csv("raw_eccc_weather.csv")

# Ensure dates are clean datetime objects
weather_df["Date"] = pd.to_datetime(weather_df["Date/Time"])

# Drop columns with 100% missing values to optimize memory
weather_df = weather_df.dropna(axis=1, how="all")

# Select core features
weather_cols = [
    "Date",
    "Station Name",
    "Latitude (y)",
    "Longitude (x)",
    "Mean Temp (°C)",
    "Total Precip (mm)",
    "Spd of Max Gust (km/h)",
]
weather_clean = weather_df[weather_cols].dropna(subset=["Mean Temp (°C)"])


# ==========================================
# STEP 2: INGEST CNFDB WILDFIRE SHAPEFILE
# ==========================================
print("🔥 Loading CNFDB Wildfire Point Data...")
# Read the downloaded .shp file directly via GeoPandas
fire_gdf = gpd.read_file("NFDB_point/NFDB_point.shp")


# Clean dates and handle potential historical missing formatting
fire_gdf["Date"] = pd.to_datetime(fire_gdf["rep_date"], errors="coerce")
fire_gdf = fire_gdf.dropna(subset=["Date", "geometry"])

# Crucial: Reproject Lambert projection map coordinates into standard Lat/Long
if fire_gdf.crs != "EPSG:4326":
    print("🗺️ Reprojecting geospatial vectors to EPSG:4326...")
    fire_gdf = fire_gdf.to_crs("EPSG:4326")

# Create a binary classification label: This event represents an active fire
fire_gdf["Is_Fire_Event"] = 1


# ==========================================
# STEP 3: TEMPORAL MERGE (Match by Date)
# ==========================================
print("🔀 Executing temporal joins across data nodes...")
# We use a left outer join match weather conditions with specific fire days
merged_dataset = pd.merge(
    weather_clean,
    fire_gdf[["Date", "geometry", "Is_Fire_Event", "size_ha"]],
    on="Date",
    how="left"
)

# Replace NaN values on non-fire days with 0 for model training
merged_dataset["Is_Fire_Event"] = merged_dataset["Is_Fire_Event"].fillna(0)
merged_dataset["size_ha"] = merged_dataset["size_ha"].fillna(0.0)


# ==============================================
# STEP 4: GEOSPATIAL VALIDATION (Distance Check)
# ==============================================
print("📏 Filtering data points by geographic proximity...")


def check_proximity(row, max_distance_km=50):
    """Ensures weather station is close enough to the fire coordinate."""
    if row["Is_Fire_Event"] == 0:
        return False  # Keep normal baseline days intact

    station_loc = Point(row["Longitude (x)"], row["Latitude (y)"])
    fire_loc = row["geometry"]

    # Quick vector distance calculation (approximate degrees to km conversion)
    deg_distance = station_loc.distance(fire_loc)
    km_distance = deg_distance * 111  # 1 degree lat is roughly 111km

    return km_distance > max_distance_km


# Apply filter function across the dataframe matrix
is_valid_proximity = merged_dataset.apply(check_proximity, axis=1)
#final_mvp_dataset = merged_dataset[is_valid_proximity]
final_mvp_dataset = merged_dataset
final_mvp_dataset.loc[is_valid_proximity, 'Is_Fire_Event'] = 0
final_mvp_dataset.loc[is_valid_proximity, 'size_ha'] = 0.0



# ==========================================
# STEP 5: SAVE ANALYSIS-READY TARGET DATA
# ==========================================
# Drop the vector geometry object before pushing to standard storage
final_output = final_mvp_dataset.drop(columns=["geometry"])
final_output = final_output.drop_duplicates()
final_output.to_csv("clean_analysis_ready_data.csv", index=False)

print(f"🎉 Success! Processed {len(final_output)} rows.")
print("📁 Target file saved as: clean_analysis_ready_data.csv")
