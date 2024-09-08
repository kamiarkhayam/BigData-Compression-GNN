import geopandas as gpd
import pandas as pd
import folium
from branca.colormap import LinearColormap

# Load the ZIP code boundaries and set CRS
zcta_gdf = gpd.read_file("C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\cb_2018_us_zcta510_500k\\cb_2018_us_zcta510_500k.shp")
zcta_gdf = zcta_gdf.to_crs("EPSG:4326")

# Filter for Illinois by ZIP code prefix
illinois_zips = zcta_gdf[zcta_gdf['GEOID10'].str.startswith(('60', '61', '62'))]

# Load energy consumption data
consumption_df = pd.read_csv("C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\Smart Meter Data\\9 digit\\cumulative_data_5digit_master.csv")
consumption_df['ZIP_CODE'] = consumption_df['ZIP_CODE'].astype(str)

# Calculate mean consumption per ZIP code
consumption_df['mean_consumption'] = consumption_df.iloc[:, 1:].mean(axis=1)
mean_consumption = consumption_df[['ZIP_CODE', 'mean_consumption']]

# Merge the mean consumption data with the ZIP code shapefile
illinois_zips = illinois_zips.merge(mean_consumption, left_on='GEOID10', right_on='ZIP_CODE', how='left')

# Initialize folium map centered on Illinois
m = folium.Map(
    location=[40.0, -89.0], 
    zoom_start=7,
    tiles=None
)

# Add Stamen Terrain Tile Layer with proper attribution
folium.TileLayer(
    'Stamen Terrain',
    attr='Map data © OpenStreetMap contributors, Imagery © Stamen Design',
    name='Stamen Terrain'
).add_to(m)

# Create a color map
colormap = LinearColormap(
    ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7'],
    vmin=min(illinois_zips['mean_consumption'].quantile([0.2, 0.4, 0.6, 0.8, 1]).tolist()),
    vmax=max(illinois_zips['mean_consumption'].quantile([0.2, 0.4, 0.6, 0.8, 1]).tolist()),
    caption='Average Energy Consumption'
)

# Add colored ZIP code areas
for idx, row in illinois_zips.iterrows():
    style_color = 'white' if pd.isna(row['mean_consumption']) else colormap(row['mean_consumption'])
    folium.GeoJson(
        row['geometry'],
        style_function=lambda feature, color=style_color: {'fillColor': color, 'color': 'black', 'weight': 0.7, 'fillOpacity': 0.7}
    ).add_to(m)

# Add the color map to the map
colormap.add_to(m)

# Layer control
folium.LayerControl().add_to(m)

# Save the map
m.save('illinois_energy_consumption_map.html')
