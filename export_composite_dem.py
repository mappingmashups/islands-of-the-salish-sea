"""
export_composite_dem.py — build a composite GeoTIFF for Relievo/Blender hillshade

Runs the exact same layout pipeline as draw_islands.py and stamps each island's
real DEM elevation into a single raster at the island's poster position.
Ocean (background) = 0 m.  Output is a Float32 GeoTIFF with a pixel size that
correctly encodes the physical scale, so Blender casts geographically-scaled shadows.

Usage (mirrors the draw_islands.py command you actually run):

  .venv/bin/python3 export_composite_dem.py \\
      --output composite_dem.tif \\
      --exclude "Harbor Island,Penn Islands,Copeland Islands,Secretary Islands" \\
      --break-after "fidalgo,galiano,anvil,jedediah" \\
      --row-buffers "12,8,4" \\
      --buffer 5 \\
      --row-paddings ",,0"
"""

import argparse
import sys
import numpy as np
import shapely
from scipy.ndimage import map_coordinates

# Import layout helpers from draw_islands.py (same directory)
from draw_islands import (
    split_rendezvous, normalize_geom, compute_layout,
    optimize_row1_nudge, compact_placed, center_rows,
    autoscale, place_geom, bbox_wh,
)
import dem_hillshade as _dem
import geopandas as gpd
from osgeo import gdal, osr


def build_composite(
    placed,
    final_geoms,
    page_w,
    page_h,
    mm_per_m,
    cell_m=None,
    zoom=12,
    output_path="composite_dem.tif",
    north_up=True,
):
    """
    Stamp each island's elevation into a poster-sized Float32 GeoTIFF.

    Parameters
    ----------
    placed       : layout entries (same list as draw_islands.py produces)
    final_geoms  : list of placed Shapely geoms, one per entry
    page_w/h     : page dimensions in mm
    mm_per_m     : poster scale factor
    cell_m       : composite pixel size in metres (default: DEM native ~12.7m at zoom 12)
    zoom         : DEM zoom level (must match what draw_islands.py uses)
    output_path  : destination GeoTIFF path
    north_up     : True (default) — matches draw_islands.py north_up orientation
    """
    if cell_m is None:
        # Zoom-12 ground resolution at mid-Salish latitude (~48.5°)
        import math
        cell_m = 40075016.686 * math.cos(math.radians(48.5)) / (2**zoom * 512)
    pixel_size_m = cell_m
    cell_mm = cell_m * mm_per_m           # mm per composite pixel
    nx = int(np.ceil(page_w / cell_mm))
    ny = int(np.ceil(page_h / cell_mm))

    print(f"Composite grid: {nx} × {ny} px  cell={cell_mm:.4f} mm = {pixel_size_m:.1f} m")

    composite = np.zeros((ny, nx), dtype=np.float32)

    for idx, entry in enumerate(placed):
        isl  = entry["island"]
        geom = final_geoms[idx]          # placed polygon in page-mm
        geom_utm = isl["geom_utm"]       # original UTM polygon

        print(f"  [{idx+1}/{len(placed)}] {isl['name']}  …", flush=True)

        # Fetch cached DEM tiles (already on disk)
        try:
            elev, xc, yc, cell_m_dem = _dem.fetch_elevation(geom_utm, zoom=zoom)
        except Exception as e:
            print(f"    DEM fetch failed: {e}")
            continue

        # Anchor = bbox CENTER of the UTM polygon — must match normalize_geom(),
        # which translates by (-bbox_cx, -bbox_cy), not by the polygon centroid.
        # Using polygon centroid here causes a shift for asymmetric islands
        # (e.g. Texada wider in the north), clipping one edge of the DEM.
        b_utm = geom_utm.bounds
        utm_cx = (b_utm[0] + b_utm[2]) / 2
        utm_cy = (b_utm[1] + b_utm[3]) / 2

        # Page-space bbox center of the placed polygon (correct counterpart)
        b = geom.bounds
        pg_cx = (b[0] + b[2]) / 2
        pg_cy = (b[1] + b[3]) / 2

        # Composite pixel bounding box for this island (with 1-pixel padding)
        col0 = max(0, int(np.floor(b[0] / cell_mm)) - 1)
        col1 = min(nx - 1, int(np.ceil( b[2] / cell_mm)) + 1)
        row0 = max(0, int(np.floor(b[1] / cell_mm)) - 1)
        row1 = min(ny - 1, int(np.ceil( b[3] / cell_mm)) + 1)

        # Build grid of composite pixel centres
        cols = np.arange(col0, col1 + 1)
        rows = np.arange(row0, row1 + 1)
        col_grid, row_grid = np.meshgrid(cols, rows)

        # Composite pixel → page-mm (pixel centres)
        px_mm = (col_grid + 0.5) * cell_mm
        py_mm = (row_grid + 0.5) * cell_mm

        # Page-mm → UTM
        # px_mm = pg_cx + (utm_x - utm_cx) * mm_per_m
        # py_mm = pg_cy - (utm_y - utm_cy) * mm_per_m   (y-flip because north_up)
        utm_x = utm_cx + (px_mm - pg_cx) / mm_per_m
        utm_y = utm_cy - (py_mm - pg_cy) / mm_per_m   # inverse y-flip

        # UTM → fractional DEM grid indices
        # x_coords increases west→east; y_coords[0]=north (decreases downward)
        dem_col = (utm_x - xc[0]) / cell_m_dem
        dem_row = (yc[0] - utm_y) / cell_m_dem

        h_s, w_s = elev.shape
        in_dem = (
            (dem_col >= 0) & (dem_col <= w_s - 1.001) &
            (dem_row >= 0) & (dem_row <= h_s - 1.001)
        )

        if not in_dem.any():
            print(f"    No valid DEM pixels — skipping")
            continue

        # Polygon containment check: only keep pixels whose UTM coords fall
        # inside the island polygon (excludes adjacent islands, mainland, etc.)
        pts = shapely.points(utm_x[in_dem], utm_y[in_dem])
        in_poly = shapely.contains(geom_utm, pts)

        # Build combined mask in original grid shape
        valid = np.zeros_like(in_dem)
        in_dem_indices = np.where(in_dem)
        valid[in_dem_indices[0][in_poly], in_dem_indices[1][in_poly]] = True

        if not valid.any():
            print(f"    No pixels inside polygon — skipping")
            continue

        # Bilinear interpolation (order=1) for masked pixels
        sampled_flat = map_coordinates(
            elev,
            [dem_row[valid], dem_col[valid]],
            order=1,
            mode="nearest",
        )

        # Write positive elevations (land) into composite
        target_rows = row_grid[valid]
        target_cols = col_grid[valid]
        land_mask = sampled_flat > 0.0

        composite[target_rows[land_mask], target_cols[land_mask]] = np.maximum(
            composite[target_rows[land_mask], target_cols[land_mask]],
            sampled_flat[land_mask],
        )

    # ── write GeoTIFF ──────────────────────────────────────────────────────────
    print(f"Writing {output_path} …")
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(
        output_path, nx, ny, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_NEEDED"],
    )
    # Affine: top-left origin at (0,0), x increases right, y increases down.
    # We store the pixel size as positive metres (X) and negative (Y for north-up).
    # Using a local CRS (EPSG:26910 coordinates repurposed as poster coords).
    ds.SetGeoTransform([0.0, pixel_size_m, 0.0, page_h / mm_per_m, 0.0, -pixel_size_m])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26910)   # UTM zone 10N — repurposed for scale only
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    band.WriteArray(composite)
    band.FlushCache()
    ds.FlushCache()
    ds = None

    w_m = nx * pixel_size_m / 1000
    h_m = ny * pixel_size_m / 1000
    elev_max = composite.max()
    print(f"Saved → {output_path}")
    print(f"  {nx} × {ny} px,  {w_m:.1f} km × {h_m:.1f} km,  max elev {elev_max:.0f} m")
    return output_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapefile",
                    default="Salish_Sea_Islands_Only/Salish_Sea_Islands_Only.shp")
    ap.add_argument("--top",         type=int,   default=100)
    ap.add_argument("--exclude",     default="")
    ap.add_argument("--break-after", default="", dest="break_after")
    ap.add_argument("--row-buffers", default="", dest="row_buffers")
    ap.add_argument("--buffer",      type=float, default=2.0)
    ap.add_argument("--row-paddings",default="", dest="row_paddings")
    ap.add_argument("--padding",     type=float, default=5.0)
    ap.add_argument("--margin",      type=float, default=14.0)
    ap.add_argument("--page-width",  type=float, default=420.0, dest="page_width")
    ap.add_argument("--page-height", type=float, default=594.0, dest="page_height")
    ap.add_argument("--scale",       type=int,   default=None)
    ap.add_argument("--simplify",    type=float, default=0)
    ap.add_argument("--label-size",  type=float, default=3.5,  dest="label_size")
    ap.add_argument("--wa-counties", default="WA_County_Boundaries/WA_County_Boundaries.shp",
                    dest="wa_counties")
    ap.add_argument("--bc-regional-districts",
                    default="ABMS_REGIONAL_DISTRICTS_SP/ABMS_RD_polygon.shp",
                    dest="bc_regional_districts")
    ap.add_argument("--compact-simplify", type=float, default=0.5,
                    dest="compact_simplify")
    ap.add_argument("--cell-m", type=float, default=None, dest="cell_m",
                    help="Composite pixel size in metres (default: DEM native ~12.7m at zoom 12)")
    ap.add_argument("--zoom",        type=int,   default=12)
    ap.add_argument("--output", default="composite_dem.tif")
    args = ap.parse_args()

    # ── load data (mirrors draw_islands.py main()) ────────────────────────────
    print(f"Reading {args.shapefile} …")
    gdf = gpd.read_file(args.shapefile)
    gdf = gdf[gdf["Name"].notna()]

    if args.exclude:
        exclude_names = {n.strip().lower() for n in args.exclude.split(",") if n.strip()}
        gdf = gdf[~gdf["Name"].str.lower().isin(exclude_names)]

    gdf = split_rendezvous(gdf)
    top = gdf.nlargest(args.top, "Area_SqKm").reset_index(drop=True)
    print(f"  {len(top)} islands selected")

    raw_geoms = top["geometry"].tolist()
    raw_names = top["Name"].tolist()
    break_after = [n.strip() for n in args.break_after.split(",") if n.strip()]

    row_paddings = {}
    if args.row_paddings.strip():
        for r, val in enumerate(args.row_paddings.split(",")):
            if val.strip():
                row_paddings[r] = float(val.strip())

    header_h = args.margin
    pts_per_mm = 72 / 25.4
    label_h_mm = (args.label_size / pts_per_mm) * 2.4 + 1.5

    if args.scale is None:
        print("Auto-scaling …")
        args.scale = autoscale(
            raw_geoms, args.page_width, args.page_height,
            args.margin, args.padding, header_h, label_h_mm,
            target_fill=0.90, simplify_m=args.simplify,
            names=raw_names, break_after=break_after,
            row_paddings=row_paddings,
        )
        print(f"  → 1:{args.scale:,}")

    mm_per_m = 1000.0 / args.scale

    # Build island dicts
    islands = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        geom = normalize_geom(row.geometry, mm_per_m, args.simplify)
        w, h = bbox_wh(geom)
        islands.append({
            "rank":     rank,
            "name":     row["Name"],
            "area":     row["Area_SqKm"],
            "geom":     geom,
            "geom_utm": row.geometry,
            "mm_per_m": mm_per_m,
            "w":        w,
            "geom_h":   h,
            "h":        h + label_h_mm,
        })

    # ── layout (mirrors draw_islands.py) ─────────────────────────────────────
    print("Packing layout …")
    placed, _ = compute_layout(
        islands, args.page_width, args.margin, args.padding, header_h,
        break_after=break_after,
        row_paddings=row_paddings,
    )

    print("Optimising row-1 nudge …")
    placed = optimize_row1_nudge(placed, args.margin, header_h, args.buffer)

    buffer_by_row = {}
    if args.row_buffers.strip():
        for r, val in enumerate(args.row_buffers.split(",")):
            val = val.strip()
            if val:
                buffer_by_row[r] = float(val)

    print("Compacting …")
    placed, final_geoms = compact_placed(
        placed, args.margin,
        buffer_mm=args.buffer,
        north_up=True,
        simplify_tol=args.compact_simplify,
        buffer_by_row=buffer_by_row or None,
    )

    print("Centering rows …")
    placed, final_geoms = center_rows(placed, final_geoms, args.page_width)

    # ── build composite DEM ───────────────────────────────────────────────────
    build_composite(
        placed, final_geoms,
        page_w=args.page_width,
        page_h=args.page_height,
        mm_per_m=mm_per_m,
        cell_m=args.cell_m,
        zoom=args.zoom,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
