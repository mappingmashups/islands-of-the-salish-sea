"""
dem_hillshade.py — DEM tile fetching and hillshade for draw_islands.py

Source: Mapterhorn Terrarium-encoded terrain tiles (512px WebP)
  https://tiles.mapterhorn.com/{z}/{x}/{y}.webp
  elevation_m = R*256 + G + B/256 - 32768

Tile coordinates follow the standard slippy-map (XYZ) convention in
Web Mercator (EPSG:3857).
"""

import math
import os
import io

import numpy as np
import requests
from PIL import Image
from pyproj import Transformer
import shapely

TILE_URL  = "https://tiles.mapterhorn.com/{z}/{x}/{y}.webp"
TILE_SIZE = 512   # pixels per tile
CACHE_DIR = ".dem_cache"


# ── tile math ─────────────────────────────────────────────────────────────────

def _lonlat_to_tile(lon, lat, zoom):
    """(lon, lat) → integer tile (tx, ty) at zoom level."""
    n = 2 ** zoom
    tx = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    ty = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return tx, ty


def _tile_nw_lonlat(tx, ty, zoom):
    """Return (lon, lat) of the NW (top-left) corner of tile (tx, ty)."""
    n = 2 ** zoom
    lon = tx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n))))
    return lon, lat


def _ground_m_per_pixel(lat_deg, zoom):
    """Approximate ground resolution in metres/pixel at given latitude."""
    return 40075016.686 * math.cos(math.radians(lat_deg)) / (2 ** zoom * TILE_SIZE)


# ── tile fetching ─────────────────────────────────────────────────────────────

def _fetch_tile(z, x, y):
    """Return tile as uint8 RGB array (TILE_SIZE × TILE_SIZE × 3), cached to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{z}_{x}_{y}.webp")
    if not os.path.exists(path):
        url = TILE_URL.format(z=z, x=x, y=y)
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
    with open(path, "rb") as f:
        return np.array(Image.open(io.BytesIO(f.read())).convert("RGB"))


def _terrarium_to_elevation(rgb):
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    return r * 256.0 + g + b / 256.0 - 32768.0


# ── main DEM fetch ─────────────────────────────────────────────────────────────

def fetch_elevation(geom_utm, crs="EPSG:26910", zoom=12, pad=0.20):
    """
    Download and reproject DEM tiles covering geom_utm.

    Returns
    -------
    elevation : 2D float64 array, shape (ny, nx)
        Row 0 = northern edge, row ny-1 = southern edge.
    x_coords  : 1D array of UTM eastings,  length nx  (west → east)
    y_coords  : 1D array of UTM northings, length ny  (north → south)
    cell_m    : approximate cell size in metres
    """
    b = geom_utm.bounds          # minx, miny, maxx, maxy in UTM
    w_m = b[2] - b[0]
    h_m = b[3] - b[1]
    margin = max(w_m, h_m) * pad
    ux0, uy0, ux1, uy1 = b[0]-margin, b[1]-margin, b[2]+margin, b[3]+margin

    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = to_wgs.transform([ux0, ux1], [uy0, uy1])
    lon_min, lon_max = sorted(lons)
    lat_min, lat_max = sorted(lats)
    lat_mid = (lat_min + lat_max) / 2

    # Tile range — add one-tile border for gradient accuracy at edges
    tx_min, ty_max = _lonlat_to_tile(lon_min, lat_min, zoom)
    tx_max, ty_min = _lonlat_to_tile(lon_max, lat_max, zoom)
    tx_min -= 1;  ty_min -= 1;  tx_max += 1;  ty_max += 1

    # Stitch tiles
    rows = []
    for ty in range(ty_min, ty_max + 1):
        row = [_fetch_tile(zoom, tx, ty) for tx in range(tx_min, tx_max + 1)]
        rows.append(np.concatenate(row, axis=1))
    stitched_rgb  = np.concatenate(rows, axis=0)
    stitched_elev = _terrarium_to_elevation(stitched_rgb)

    # Approximate output resolution
    cell_m = _ground_m_per_pixel(lat_mid, zoom)

    # Build UTM output grid (north → south, west → east)
    nx = max(32, int((ux1 - ux0) / cell_m))
    ny = max(32, int((uy1 - uy0) / cell_m))
    x_coords = np.linspace(ux0, ux1, nx)
    y_coords = np.linspace(uy1, uy0, ny)   # north → south

    xx, yy   = np.meshgrid(x_coords, y_coords)
    lons_g, lats_g = to_wgs.transform(xx.ravel(), yy.ravel())
    lons_g = np.asarray(lons_g).reshape(ny, nx)
    lats_g = np.asarray(lats_g).reshape(ny, nx)

    # Convert lon/lat to fractional pixel coords in the stitched array
    n = 2 ** zoom
    px = (lons_g + 180.0) / 360.0 * n * TILE_SIZE - tx_min * TILE_SIZE

    lat_r  = np.radians(lats_g)
    merc_y = np.log(np.tan(lat_r) + 1.0 / np.cos(lat_r))
    py = (1.0 - merc_y / math.pi) / 2.0 * n * TILE_SIZE - ty_min * TILE_SIZE

    # Bilinear interpolation
    h_s, w_s = stitched_elev.shape
    px = np.clip(px, 0, w_s - 1.001)
    py = np.clip(py, 0, h_s - 1.001)
    px0 = px.astype(int);  px1 = np.minimum(px0 + 1, w_s - 1)
    py0 = py.astype(int);  py1 = np.minimum(py0 + 1, h_s - 1)
    fx  = px - px0;        fy  = py - py0

    elevation = (stitched_elev[py0, px0] * (1 - fx) * (1 - fy) +
                 stitched_elev[py0, px1] *      fx  * (1 - fy) +
                 stitched_elev[py1, px0] * (1 - fx) *      fy  +
                 stitched_elev[py1, px1] *      fx  *      fy)

    return elevation, x_coords, y_coords, cell_m


# ── hillshade ─────────────────────────────────────────────────────────────────

def compute_hillshade(elevation, cell_m, azimuth=315.0, altitude=45.0):
    """
    Standard Lambert hillshade from a north-to-south elevation grid.

    elevation : 2D array, row 0 = north
    cell_m    : cell size in metres
    azimuth   : sun azimuth in degrees (0=N, 90=E, 180=S, 270=W)
    altitude  : sun altitude above horizon in degrees
    Returns hillshade in [0, 1].
    """
    alt_r = math.radians(altitude)
    az_r  = math.radians(azimuth)

    # np.gradient returns (d/drow, d/dcol); row increases southward
    dz_row, dz_col = np.gradient(elevation, cell_m)
    dz_dx =  dz_col          # east is positive col
    dz_dy = -dz_row          # north is negative row → flip sign

    slope  = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    aspect = np.arctan2(-dz_dy, dz_dx)   # 0=E, π/2=N, measured CCW from east

    hs = (np.sin(alt_r) * np.cos(slope) +
          np.cos(alt_r) * np.sin(slope) * np.cos(az_r - aspect))
    return np.clip(hs, 0.0, 1.0)


# ── mask ──────────────────────────────────────────────────────────────────────

def mask_to_polygon(arr, x_coords, y_coords, polygon_utm):
    """
    Return arr with NaN outside polygon_utm.  Uses shapely 2.x vectorized API.
    """
    xx, yy = np.meshgrid(x_coords, y_coords)
    pts    = shapely.points(xx.ravel(), yy.ravel())
    inside = shapely.contains(polygon_utm, pts).reshape(arr.shape)
    result = arr.copy()
    result[~inside] = np.nan
    return result


# ── page-space extent helper ───────────────────────────────────────────────────

def page_extent(x_coords, y_coords, geom_utm, final_geom):
    """
    Compute the matplotlib imshow extent (left, right, bottom, top) in page mm
    for a DEM grid that covers (x_coords, y_coords) in UTM.

    Anchors directly to final_geom.bounds (the placed polygon in page space),
    so it automatically accounts for compaction and row-centering shifts.

    Coordinate relationships (north_up rendering, y-down page axes):
      page_x = b_pg[0] + (utm_x - b_utm[0]) * mm_per_m
      page_y = b_pg[1] + (b_utm[3] - utm_y) * mm_per_m   (y-flipped)

    Returns (left_mm, right_mm, bottom_mm, top_mm) for ax.imshow(extent=...).
    """
    b_utm = geom_utm.bounds    # (ux0, uy0, ux1, uy1)  uy1=north, uy0=south
    b_pg  = final_geom.bounds  # (pg_xmin, pg_ymin, pg_xmax, pg_ymax)  pg_ymin=north edge

    mm_per_m = (b_pg[2] - b_pg[0]) / (b_utm[2] - b_utm[0])

    pg_x_left  = b_pg[0] + (x_coords[0]  - b_utm[0]) * mm_per_m   # west DEM edge
    pg_x_right = b_pg[0] + (x_coords[-1] - b_utm[0]) * mm_per_m   # east DEM edge

    # y_coords[0]  = northernmost DEM edge (uy1 + pad) → above island → smaller page_y
    # y_coords[-1] = southernmost DEM edge (uy0 - pad) → below island → larger page_y
    pg_y_top    = b_pg[1] + (b_utm[3] - y_coords[0])  * mm_per_m
    pg_y_bottom = b_pg[1] + (b_utm[3] - y_coords[-1]) * mm_per_m

    # imshow extent = [left, right, bottom, top]  (y-down axes: bottom > top)
    return pg_x_left, pg_x_right, pg_y_bottom, pg_y_top
