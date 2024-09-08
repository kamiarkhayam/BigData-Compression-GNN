import geopandas as gpd
import pandas as pd
import folium
from shapely.geometry import box

# Load the state shapefile and set CRS
gdf = gpd.read_file("C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\cb_2018_us_state_500k\\cb_2018_us_state_500k.shp")
us_mainland = gdf[gdf['STATEFP'].isin([str(i).zfill(2) for i in range(1, 57)])]  # Filter for US mainland (excluding territories)
us_mainland = us_mainland.to_crs("EPSG:4326")

# Load the dataset with latitude and longitude
data = pd.read_csv("C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\NREL PV Data\\all_states_features.csv")

# Initialize folium map centered on the US mainland
m = folium.Map(
    location=[39.8283, -98.5795],  # Center of the US mainland
    zoom_start=4,
    tiles='OpenStreetMap'
)

# Add several tile layers with proper attribution
folium.TileLayer(
    'Stamen Terrain', 
    attr='Map data © OpenStreetMap contributors, Imagery © Stamen Design'
).add_to(m)
folium.TileLayer(
    'Stamen Toner', 
    attr='Map data © OpenStreetMap contributors, Imagery © Stamen Design'
).add_to(m)
folium.TileLayer(
    'Stamen Watercolor', 
    attr='Map data © OpenStreetMap contributors, Imagery © Stamen Design'
).add_to(m)
folium.TileLayer(
    'CartoDB positron', 
    attr='Map data © OpenStreetMap contributors, CartoDB'
).add_to(m)
folium.TileLayer(
    'CartoDB dark_matter', 
    attr='Map data © OpenStreetMap contributors, CartoDB'
).add_to(m)

# Add a plain white mask outside the US mainland
world_bounds = box(-180, -90, 180, 90)
us_shape = us_mainland.unary_union
mask = world_bounds.difference(us_shape)

# Add the mask layer to hide everything outside the US mainland
folium.GeoJson(
    mask,
    style_function=lambda x: {
        'fillColor': 'white',
        'color': 'none',
        'fillOpacity': 1.0
    }
).add_to(m)

# Add state borders only
for _, row in us_mainland.iterrows():
    folium.GeoJson(
        row['geometry'].simplify(0.01), 
        style_function=lambda feature: {
            'fillColor': 'white',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.1
        },
        tooltip=row['NAME']
    ).add_to(m)

# Plot each point from the dataset with different colors based on PV type
for _, row in data.iterrows():
    color = 'navy' if row['PV Type'] == 'UPV' else 'darkred'  # Color based on PV Type
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=2,  # Smaller size for the point
        color=color,
        fill=True,
        fill_color=color
    ).add_to(m)

# Layer control to toggle between different tile layers
folium.LayerControl().add_to(m)

# Save the map
m.save('us_map_with_pv_data_points.html')
