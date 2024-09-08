import fitz  # PyMuPDF
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

# Define the colors used in the Folium map as a continuous color map
colors = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7']
cmap = LinearSegmentedColormap.from_list("custom", colors)

# Define your specific data range based on quantiles
vmin = 62.51893660235708 * 2  # Replace with your actual minimum from quantile
vmax = 5190.964066776693 * 2  # Replace with your actual maximum from quantile

# Load the PDF file
pdf_file = "C:/Users/bmb2tn/OneDrive - University of Virginia/Ph.D. Projects/Big Data/smart_meter_zip_codes_map.pdf"
doc = fitz.open(pdf_file)

# Select the first page
page = doc[0]

# Define the area to crop
crop_rect = fitz.Rect(70, 70, 400, 580)  # Example coordinates, adjust as needed
page.set_cropbox(crop_rect)

# Convert the cropped PDF page to an image at a higher resolution
zoom = 4
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=mat)
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# Create a matplotlib figure
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)
ax.axis('off')  # Hide the axes

# Create a normalized colorbar based on your data range
norm = Normalize(vmin=vmin, vmax=vmax)
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.03)
cbar.set_label('Average Hourly Energy Consumption (kWh)', fontsize=16)
cbar.ax.tick_params(labelsize=14)
# Save the final image with the legend as a PNG
output_png = "C:/Users/bmb2tn/OneDrive - University of Virginia/Ph.D. Projects/Big Data/smart_meter_consumption_map.png"
plt.savefig(output_png, bbox_inches='tight', pad_inches=0, dpi=300)
print(f"Saved edited image with legend as {output_png}")

# Show the plot (optional)
plt.show()
