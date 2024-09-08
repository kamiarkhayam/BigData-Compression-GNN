import geopandas as gpd
import folium
from shapely.geometry import box

# Load the county shapefile and set CRS
gdf = gpd.read_file("C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\cb_2018_us_county_500k\\cb_2018_us_county_500k.shp")
california = gdf[gdf['STATEFP'] == '06'].to_crs("EPSG:4326")

# Normalize county names
california['NAME'] = california['NAME'].str.strip().str.title()

# List of highlighted counties
highlighted_counties = ['Butte', 'Colusa', 'El Dorado', 'Glenn', 'Nevada', 'Placer', 
                        'Sacramento', 'Sierra', 'Sutter', 'Yolo', 'Yuba']
highlighted_geos = california[california['NAME'].isin([name.strip().title() for name in highlighted_counties])]

# Load and filter highways
highways = gpd.read_file("C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\SHN_Lines\\SHN_Lines.shp").to_crs("EPSG:4326")
major_highways = highways[highways['RouteType'].isin(['Interstate', 'US', 'State'])]

# Clip highways to the boundaries of the highlighted counties
clipped_highways = gpd.clip(major_highways, highlighted_geos)

# Get the bounding box of California
california_bounds = california.total_bounds  # [minx, miny, maxx, maxy]

# Map initialization with a base tile layer
m = folium.Map(
    location=[37.5, -119.5], 
    zoom_start=7,
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

# Add a plain white mask outside California
world_bounds = box(-180, -90, 180, 90)
california_shape = california.unary_union
mask = world_bounds.difference(california_shape)

# Add the mask layer to hide everything outside California
folium.GeoJson(
    mask,
    style_function=lambda x: {
        'fillColor': 'white',
        'color': 'none',
        'fillOpacity': 1.0
    }
).add_to(m)

# Add county borders
for _, row in california.iterrows():
    folium.GeoJson(
        row['geometry'].simplify(0.01), 
        style_function=lambda feature, color='midnightblue' if row['NAME'] in highlighted_counties else 'white': {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.3 if color == 'midnightblue' else 0.1
        },
        tooltip=row['NAME']
    ).add_to(m)

# Define color map for road types
road_color_map = {
    'Interstate': 'darkred',
    'US': 'navy',
    'State': 'darkgreen'
}

# Add highways with conditional styling for road types
for _, highway in clipped_highways.iterrows():
    style_color = road_color_map.get(highway['RouteType'], 'gray')  # Default to gray if type is unknown
    folium.GeoJson(
        highway['geometry'],
        style_function=lambda x, color=style_color: {
            'color': color,
            'weight': 2
        },
        tooltip=f"Route: {highway.get('Route', 'Unknown')}"
    ).add_to(m)

# Fit map bounds to California only
m.fit_bounds([[california_bounds[1], california_bounds[0]], [california_bounds[3], california_bounds[2]]])

# Layer control to toggle between different tile layers
folium.LayerControl().add_to(m)

# Save the map
m.save('california_map_with_clipped_surroundings.html')
