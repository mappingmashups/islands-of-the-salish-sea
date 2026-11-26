"""Quick test: fetch DEM and compute hillshade for Whidbey Island."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
from dem_hillshade import fetch_elevation, compute_hillshade, mask_to_polygon

gdf = gpd.read_file("Salish_Sea_Islands_Only/Salish_Sea_Islands_Only.shp")
whidbey = gdf[gdf["Name"] == "Whidbey Island"].iloc[0].geometry
print(f"Whidbey bbox: {whidbey.bounds}")

print("Fetching DEM tiles at zoom 12 …")
elev, x_coords, y_coords, cell_m = fetch_elevation(whidbey, zoom=12)
print(f"  Elevation array: {elev.shape}, cell size: {cell_m:.1f} m")
print(f"  Elevation range: {np.nanmin(elev):.1f} – {np.nanmax(elev):.1f} m")

hs = compute_hillshade(elev, cell_m)
hs_masked = mask_to_polygon(hs, x_coords, y_coords, whidbey)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
axes[0].imshow(elev, cmap="terrain", origin="upper")
axes[0].set_title("Elevation (UTM grid)")
axes[1].imshow(hs_masked, cmap="gray", vmin=0, vmax=1, origin="upper")
axes[1].set_title("Hillshade (masked to island)")
plt.tight_layout()
plt.savefig("test_dem_whidbey.png", dpi=150)
print("Saved test_dem_whidbey.png")
