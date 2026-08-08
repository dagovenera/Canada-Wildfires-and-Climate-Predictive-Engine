import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

print("🛠️ Generating mock wildfire vector files...")

# Define mock fire points inside British Columbia (near Kelowna coordinates)
# Kelowna sits roughly at Lat 49.9, Lon -119.4
data = {
    "rep_date": [
        "2024-07-05", "2024-07-12", "2024-08-01", 
        "2023-06-15", "2022-08-20", "2021-07-19"
    ],
    "size_ha": [150.5, 2400.0, 12.3, 450.2, 8900.0, 310.0],
    "geometry": [
        Point(-119.42, 49.91), # Fire very close to station
        Point(-119.38, 49.85), # Fire close to station
        Point(-119.10, 50.10), # Fire slightly further out
        Point(-120.50, 50.20), # Fire far away (should be filtered out by proximity check)
        Point(-119.40, 49.89), # Fire very close
        Point(-119.45, 49.93)  # Fire close
    ]
}

# Create a GeoDataFrame with standard geographic structures
df = pd.DataFrame(data)
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

# Generate folder infrastructure matching pipeline config
os.makedirs("NFDB_point", exist_ok=True)

# Export the required structural vector blocks (.shp, .shx, .dbf)
gdf.to_file("NFDB_point/nfdb_point.shp")

print("🎉 Complete! Mock spatial fire-event dataset saved to 'NFDB_point'")
