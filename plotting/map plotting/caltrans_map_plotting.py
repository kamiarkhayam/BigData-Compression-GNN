import fitz  # PyMuPDF
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import rcParams

# Set the font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

# Step 1: Load the PDF and Extract a Specific Part

# Load the PDF
pdf_file = "california_map_with_clipped_surroundings.pdf"
doc = fitz.open(pdf_file)

# Select the first page
page = doc[0]

# Define the area to crop (x0, y0, x1, y1) - adjust these values based on your needs
crop_rect = fitz.Rect(200, 100, 620, 550)  # Example coordinates, you can adjust them

# Crop the page to the defined rectangle
page.set_cropbox(crop_rect)

# Convert the cropped PDF page to an image (Pillow Image object) with higher resolution
zoom = 4  # Adjust the zoom factor to increase DPI (e.g., 2, 4, 8)
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=mat)
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# Step 2: Use Matplotlib to Add a Legend

# Create a matplotlib figure
fig, ax = plt.subplots(figsize=(10, 10))

# Display the image
ax.imshow(img)
ax.axis('off')  # Hide the axes

# Create legend items using matplotlib patches and lines

# Rectangle patch for Caltrans District 3 with transparency
caltrans_patch = mpatches.Patch(
    color='midnightblue', label='Caltrans District 3', alpha=0.3
)

# Line items for different road types
interstate_line = mlines.Line2D([], [], color='darkred', label='Interstate', linewidth=2)
us_route_line = mlines.Line2D([], [], color='navy', label='US Route', linewidth=2)
state_route_line = mlines.Line2D([], [], color='darkgreen', label='State Route', linewidth=2)

# Add the legend to the plot with a custom location using bbox_to_anchor
ax.legend(
    handles=[caltrans_patch, interstate_line, us_route_line, state_route_line],
    loc='upper left',  # Location anchor point (corner of the legend box)
    bbox_to_anchor=(0.55, 0.95),  # Custom coordinates (x, y)
    fontsize=18,
    frameon=True,
)

# Save the final image with the legend as a PNG
output_png = "caltrans_district3_map.png"
plt.savefig(output_png, bbox_inches='tight', pad_inches=0, dpi=600)

print(f"Saved edited image with legend as {output_png}")

# Show the plot (optional)
plt.show()
