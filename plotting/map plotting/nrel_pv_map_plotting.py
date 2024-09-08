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
pdf_file = "C:\\Users\\bmb2tn\\OneDrive - University of Virginia\\Ph.D. Projects\\Big Data\\nrel_pv_us_map.pdf"
doc = fitz.open(pdf_file)

# Select the first page
page = doc[0]

# Define the area to crop (x0, y0, x1, y1) - adjust these values based on your needs
crop_rect = fitz.Rect(100, 100, 1170, 700)  # Example coordinates, you can adjust them

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

# Circle patches for different PV types
upv_circle = mlines.Line2D([], [], color='navy', marker='o', linestyle='None', markersize=10, label='Utility PV')
dpv_circle = mlines.Line2D([], [], color='darkred', marker='o', linestyle='None', markersize=10, label='Distributed PV')

# Add the legend to the plot with a custom location using bbox_to_anchor
ax.legend(
    handles=[upv_circle, dpv_circle],
    loc='upper left',  # Location anchor point (corner of the legend box)
    bbox_to_anchor=(0.03, 0.3),  # Custom coordinates (x, y)
    fontsize=18,
    frameon=True,
)

# Save the final image with the legend as a PNG
output_png = "nrel_pv_us_map_.png"
plt.savefig(output_png, bbox_inches='tight', pad_inches=0, dpi=600)

print(f"Saved edited image with legend as {output_png}")

# Show the plot (optional)
plt.show()
