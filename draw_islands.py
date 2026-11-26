#!/usr/bin/env python3
"""
draw_islands.py — Salish Sea islands poster

Draw the top-N Salish Sea islands by area at the same map scale, arranged in
reading order (largest → smallest) on a portrait page.

Inspired by David Garcia's "100 Largest Islands of the World" map.

Usage:
    python draw_islands.py [options]

Key options:
    --top N              Number of islands (default: 100)
    --exclude NAMES      Comma-separated island names to skip, e.g. 'Harbor Island'
    --scale S            1:S map scale (default: auto-fit to page)
    --page-width W       Page width mm  (default: 420 = A3)
    --page-height H      Page height mm (default: 594 = A3)
    --padding P          Shelf gap between islands mm (default: 5)
    --margin M           Page margin mm (default: 14)
    --row1 N             Force N islands onto row 1 (default: auto)
    --break-after NAMES  Comma-separated name fragments after which to force a
                         row break, e.g. 'fidalgo,galiano'
    --buffer MM          Min polygon-edge gap after kerning (default: 2)
    --row-buffers LIST   Per-row buffer overrides (1-indexed, comma-separated),
                         e.g. '12,8,4' — remaining rows use --buffer
    --fill-us COLOR      US island fill (default: #B8C9A3)
    --fill-ca COLOR      CA island fill (default: #C4CEC2)
    --output FILE        Output file; extension sets format pdf/png/svg
                         (default: islands.pdf)
    --dpi N              Raster resolution (default: 200)
"""

import argparse
import re
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import geopandas as gpd
from shapely.affinity import translate, scale as affine_scale, rotate as affine_rotate


# ── geometry helpers ──────────────────────────────────────────────────────────

def landscape_rotation_angle(geom):
    """
    Return the angle (degrees) to rotate the island so its longest axis is horizontal.
    Uses the minimum rotated bounding rectangle.
    """
    mrr = geom.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords[:4])
    dx1, dy1 = coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]
    dx2, dy2 = coords[2][0] - coords[1][0], coords[2][1] - coords[1][1]
    len1 = (dx1**2 + dy1**2) ** 0.5
    len2 = (dx2**2 + dy2**2) ** 0.5
    dx, dy = (dx1, dy1) if len1 >= len2 else (dx2, dy2)
    return -np.degrees(np.arctan2(dy, dx))


def normalize_geom(geom, mm_per_meter, simplify_m=0, rotate_landscape=False):
    """Center geometry at origin, optionally simplify/rotate, then scale meters → mm."""
    if simplify_m > 0:
        geom = geom.simplify(simplify_m, preserve_topology=True)
    if rotate_landscape:
        angle = landscape_rotation_angle(geom)
        geom = affine_rotate(geom, angle, origin=geom.centroid)
    b = geom.bounds
    cx = (b[0] + b[2]) / 2
    cy = (b[1] + b[3]) / 2
    g = translate(geom, -cx, -cy)
    return affine_scale(g, xfact=mm_per_meter, yfact=mm_per_meter, origin=(0, 0))


def bbox_wh(geom):
    b = geom.bounds
    return b[2] - b[0], b[3] - b[1]


def place_geom(geom, left_mm, top_mm, north_up=True):
    """
    Translate normalized geom so its bbox top-left corner = (left_mm, top_mm)
    in screen coordinates (Y increases downward).

    north_up=True (default): flip the Y-axis so geographic north is at the visual top.
    north_up=False: no flip (use when island was already rotate-to-fit oriented).
    """
    if north_up:
        # Flip Y so north (geog. max-y) appears at the visual top (screen min-y)
        geom = affine_scale(geom, xfact=1, yfact=-1, origin=(0, 0))
    b = geom.bounds
    return translate(geom, left_mm - b[0], top_mm - b[1])


def shapely_to_mpl_path(geom):
    """Convert Polygon or MultiPolygon to a single closed matplotlib Path."""
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    verts, codes = [], []
    for poly in polys:
        for ring in [poly.exterior] + list(poly.interiors):
            xy = np.array(ring.coords)
            if len(xy) < 3:
                continue
            n = len(xy)
            c = np.full(n, MplPath.LINETO, dtype=np.uint8)
            c[0] = MplPath.MOVETO
            c[-1] = MplPath.CLOSEPOLY
            verts.append(xy)
            codes.append(c)
    if not verts:
        return None
    return MplPath(np.vstack(verts), np.concatenate(codes))


# ── layout ────────────────────────────────────────────────────────────────────

def compute_layout(islands, page_w, margin, padding, header_h, row1_count=0,
                   break_after=None, row_paddings=None):
    """
    Shelf packing (left-to-right, top-to-bottom) in strict input order.

    Alignment per row:
      Row 1  → bottom-aligned: all islands share the same bottom edge, leaving
               open space in the upper-right for a title block.
      Row 2+ → midpoint-aligned: each island is centered on the first island's
               vertical midpoint, giving a relaxed, organic feel.

    row1_count  : if > 0, force exactly that many islands onto row 1, ignoring
                  the page-width overflow check for that row.
    break_after : set of lowercase name fragments; force a row break immediately
                  after any island whose name contains one of these strings.
    header_h    : reserved height at top of page (title area) in mm.
    Each island dict must have 'w', 'h' (packing height incl. label), 'geom_h'.
    Returns list of {'island':…, 'left':mm, 'top':mm} and content_bottom.
    """
    break_after = set(n.lower() for n in (break_after or []))

    def _forced_break(name):
        if not name or not break_after:
            return False
        ln = name.lower()
        return any(frag in ln for frag in break_after)

    usable_w = page_w - 2 * margin
    x = 0.0
    y = margin + header_h   # top of current shelf (screen y-down coords)
    shelf_h = 0.0
    placed = []
    shelf_entries = []
    row_num = 0             # 0 = first row

    def _align_shelf(entries, shelf_top, align):
        if not entries:
            return
        if align == "bottom":
            # All islands share the same bottom edge (tallest determines the row bottom).
            row_bottom = shelf_top + max(e["island"]["geom_h"] for e in entries)
            for entry in entries:
                entry["top"] = row_bottom - entry["island"]["geom_h"]
        else:  # "midpoint"
            if len(entries) < 2:
                return
            heights = sorted(e["island"]["geom_h"] for e in entries)
            median_h = heights[len(heights) // 2]
            anchor_mid = shelf_top + median_h / 2
            for entry in entries:
                gh = entry["island"]["geom_h"]
                entry["top"] = anchor_mid - gh / 2

    def _close_shelf():
        nonlocal x, y, shelf_h, shelf_entries, row_num
        align = "bottom" if row_num == 0 else "midpoint"
        _align_shelf(shelf_entries, y, align)
        gap = (row_paddings or {}).get(row_num, padding)
        y += shelf_h + gap
        x = 0.0
        shelf_h = 0.0
        shelf_entries = []
        row_num += 1

    for isl in islands:
        w, h = isl["w"], isl["h"]

        # Decide whether to start a new shelf before placing this island.
        # Row 1 with a forced count: keep adding until we hit row1_count,
        # then break on the very next island regardless of width.
        force_row1 = (row_num == 0 and row1_count > 0
                      and len(shelf_entries) < row1_count)
        width_overflow = placed and x + w > usable_w + 1e-6
        if width_overflow and not force_row1:
            _close_shelf()

        entry = {"island": isl, "left": margin + x, "top": y, "row": row_num}
        placed.append(entry)
        shelf_entries.append(entry)
        x += w + padding
        shelf_h = max(shelf_h, h)

        # Force a row break AFTER this island if its name is in break_after.
        if _forced_break(isl["name"]):
            _close_shelf()

    # Align and close the final shelf
    align = "bottom" if row_num == 0 else "midpoint"
    _align_shelf(shelf_entries, y, align)

    content_bottom = y + shelf_h
    return placed, content_bottom


def layout_height(islands_raw, scale_denom, page_w, margin, padding,
                  header_h, label_h_mm, simplify_m=0, names=None, break_after=None,
                  row_paddings=None):
    """Estimate total content height using shelf packing (matches compute_layout)."""
    mm = 1000.0 / scale_denom
    usable_w = page_w - 2 * margin
    x, y, sh = 0.0, margin + header_h, 0.0
    started = False
    row_num = 0
    break_frags = set(n.lower() for n in (break_after or []))

    def _forced_break(i):
        if not break_frags or not names:
            return False
        name = names[i]
        return bool(name) and any(frag in name.lower() for frag in break_frags)

    def _gap():
        return (row_paddings or {}).get(row_num, padding)

    for i, geom_raw in enumerate(islands_raw):
        b = geom_raw.bounds
        w = (b[2] - b[0]) * mm
        h = (b[3] - b[1]) * mm + label_h_mm
        if started and x + w > usable_w + 1e-6:
            y += sh + _gap()
            x, sh = 0.0, 0.0
            row_num += 1
        x += w + padding
        sh = max(sh, h)
        started = True
        if _forced_break(i):
            y += sh + _gap()
            x, sh = 0.0, 0.0
            row_num += 1
    return y + sh


def compact_placed(placed, margin, buffer_mm, north_up=True,
                   n_iter=22, simplify_tol=0, buffer_by_row=None):
    """
    Kerning pass: nudge each island left until the true polygon-to-polygon
    distance equals the target buffer for that island's row.

    buffer_by_row : optional dict {row_num: buffer_mm} — overrides buffer_mm
                    for specific rows. Rows not in the dict use buffer_mm.
    """
    min_x = margin

    final_geoms = []
    check_geoms = []

    for i, entry in enumerate(placed):
        isl  = entry["island"]
        cur_buf = (buffer_by_row.get(entry.get("row", 0), buffer_mm)
                   if buffer_by_row else buffer_mm)
        geom = place_geom(isl["geom"], entry["left"], entry["top"], north_up=north_up)

        if not final_geoms:
            final_geoms.append(geom)
            check_geoms.append(geom.simplify(simplify_tol) if simplify_tol else geom)
            continue

        b = geom.bounds
        max_left_approx = b[0] - min_x
        search_r = max_left_approx + (b[3] - b[1]) + cur_buf + 5

        def nearby(g):
            gb = g.bounds
            return (gb[2] > b[0] - search_r and gb[0] < b[2] + search_r and
                    gb[3] > b[1] - search_r and gb[1] < b[3] + search_r)

        cands = [cg for cg in check_geoms if nearby(cg)]

        if not cands:
            final_geoms.append(geom)
            check_geoms.append(geom.simplify(simplify_tol) if simplify_tol else geom)
            continue

        def min_dist_left(shift):
            return min(translate(geom, -shift, 0).distance(c) for c in cands)

        max_left = geom.bounds[0] - min_x
        if max_left > 0.05:
            lo, hi, best = 0.0, max_left, 0.0
            for _ in range(n_iter):
                mid = (lo + hi) / 2
                if min_dist_left(mid) >= cur_buf:
                    best = mid; lo = mid
                else:
                    hi = mid
                if hi - lo < 0.02:
                    break
            geom = translate(geom, -best, 0)
            entry["left"] -= best

        final_geoms.append(geom)
        check_geoms.append(geom.simplify(simplify_tol) if simplify_tol else geom)

        if (i + 1) % 10 == 0:
            print(f"  compacted {i+1}/{len(placed)} …", flush=True)

    return placed, final_geoms


def optimize_row1_nudge(placed, margin, header_h, buffer_mm, north_up=True,
                        n_y=60, n_iter=22):
    """
    Row-1 vertical-fit rule for the 2nd island (Texada):

    Slide Texada vertically across its full range (content-top → natural
    bottom-aligned position) and pick the y that lets it kern furthest left
    into the 1st island's (Whidbey's) concavity.

    After fixing Texada's y, bottom-align all remaining row-1 islands to
    Texada's bottom edge.  Whidbey keeps its original position and is allowed
    to descend below that shared baseline.
    """
    row1 = [e for e in placed if e.get("row", 0) == 0]
    if len(row1) < 2:
        return placed

    anchor = row1[0]   # Whidbey — stays put
    mover  = row1[1]   # Texada  — optimised

    whidbey_geom = place_geom(anchor["island"]["geom"],
                               anchor["left"], anchor["top"], north_up=north_up)

    y_min = margin + header_h   # highest allowed (top of content area)
    y_max = mover["top"]        # lowest  allowed (natural bottom-aligned top)

    best_shift = -1.0
    best_y     = y_max

    for k in range(n_y + 1):
        y_test = y_min + (y_max - y_min) * k / n_y
        geom   = place_geom(mover["island"]["geom"],
                             mover["left"], y_test, north_up=north_up)

        max_left = geom.bounds[0] - margin
        if max_left <= 0.05:
            shift = 0.0
        else:
            lo, hi, shift = 0.0, max_left, 0.0
            for _ in range(n_iter):
                mid = (lo + hi) / 2
                if translate(geom, -mid, 0).distance(whidbey_geom) >= buffer_mm:
                    shift = mid; lo = mid
                else:
                    hi = mid
                if hi - lo < 0.02:
                    break

        if shift > best_shift:
            best_shift = shift
            best_y     = y_test

    mover["top"] = best_y
    texada_bottom = best_y + mover["island"]["geom_h"]

    for entry in row1[2:]:
        entry["top"] = texada_bottom - entry["island"]["geom_h"]

    print(f"  {mover['island']['name']} nudged up {y_max - best_y:.1f} mm "
          f"→ baseline {texada_bottom:.1f} mm  (leftward gain: {best_shift:.1f} mm)")
    return placed


def center_rows(placed, final_geoms, page_w):
    """Shift each row so its content is horizontally centered on the page."""
    rows = {}
    for i, entry in enumerate(placed):
        r = entry.get("row", 0)
        rows.setdefault(r, []).append(i)

    for indices in rows.values():
        if final_geoms is not None:
            left_x  = min(final_geoms[i].bounds[0] for i in indices)
            right_x = max(final_geoms[i].bounds[2] for i in indices)
        else:
            left_x  = min(placed[i]["left"] for i in indices)
            right_x = max(placed[i]["left"] + placed[i]["island"]["w"] for i in indices)

        shift = page_w / 2 - (left_x + right_x) / 2
        for i in indices:
            placed[i]["left"] += shift
            if final_geoms is not None:
                final_geoms[i] = translate(final_geoms[i], shift, 0)

    return placed, final_geoms


def justify_rows(placed, final_geoms, page_w, margin, justify_from_row=0,
                 justify_margin=None, to_row=None):
    """
    Redistribute islands in rows [justify_from_row, to_row) (0-indexed) so they
    span exactly from justify_margin to (page_w - justify_margin), with equal gaps.
    to_row=None means all rows from justify_from_row onwards.
    justify_margin defaults to margin when not specified.
    """
    jm = justify_margin if justify_margin is not None else margin
    rows = {}
    for i, entry in enumerate(placed):
        r = entry.get("row", 0)
        if r >= justify_from_row and (to_row is None or r < to_row):
            rows.setdefault(r, []).append(i)

    available_w = page_w - 2 * jm

    for _r, idx_list in sorted(rows.items()):
        if len(idx_list) < 2:
            continue
        idx_list.sort(key=lambda i: final_geoms[i].bounds[0])
        widths = [final_geoms[i].bounds[2] - final_geoms[i].bounds[0] for i in idx_list]
        total_isl_w = sum(widths)
        if total_isl_w >= available_w:
            continue
        gap = (available_w - total_isl_w) / (len(idx_list) - 1)
        x = jm
        for j, i in enumerate(idx_list):
            shift = x - final_geoms[i].bounds[0]
            final_geoms[i] = translate(final_geoms[i], shift, 0)
            placed[i]["left"] += shift
            x += widths[j] + gap

    return placed, final_geoms


def expand_to_margins(placed, final_geoms, page_w, page_h, margin,
                      justify_from_row=None):
    """
    Scale the entire layout up uniformly, anchored at (page_w/2, top-of-content),
    until the widest non-justified row spans margin-to-margin, capped by the
    vertical constraint (bottom of last island ≤ page_h - margin).

    Returns (placed, final_geoms, f) where f is the actual scale factor applied.
    """
    justify_from = justify_from_row if justify_from_row is not None else len(placed)

    row_extents = {}
    for i, entry in enumerate(placed):
        r = entry.get("row", 0)
        if r >= justify_from:
            continue
        b = final_geoms[i].bounds
        lo, hi = row_extents.get(r, (b[0], b[2]))
        row_extents[r] = (min(lo, b[0]), max(hi, b[2]))

    if not row_extents:
        return placed, final_geoms, 1.0

    # Horizontal: widen until the widest centered row just touches both margins
    max_half_w = max(
        max(abs(hi - page_w / 2), abs(lo - page_w / 2))
        for lo, hi in row_extents.values()
    )
    f_horiz = (page_w / 2 - margin) / max_half_w if max_half_w > 0 else 1.0

    # Vertical: don't let the last island bleed past page_h - margin
    top_y = min(g.bounds[1] for g in final_geoms)
    bot_y = max(g.bounds[3] for g in final_geoms)
    f_vert = (page_h - margin - top_y) / (bot_y - top_y) if bot_y > top_y else 1.0

    f = min(f_horiz, f_vert)
    if f <= 1.001:
        return placed, final_geoms, 1.0

    ax, ay = page_w / 2, top_y
    for i in range(len(final_geoms)):
        final_geoms[i] = affine_scale(final_geoms[i], xfact=f, yfact=f, origin=(ax, ay))
        placed[i]["left"] = ax + f * (placed[i]["left"] - ax)
        placed[i]["top"]  = ay + f * (placed[i]["top"]  - ay)

    return placed, final_geoms, f


def autoscale(geoms, page_w, page_h, margin, padding, header_h, label_h_mm,
              target_fill=0.88, simplify_m=0, names=None, break_after=None,
              row_paddings=None):
    """Binary-search the 1:S scale denominator so content fills target_fill of page_h."""
    target = page_h * target_fill

    lo, hi = 50_000, 5_000_000
    for _ in range(50):
        mid = (lo + hi) // 2
        h = layout_height(geoms, mid, page_w, margin, padding,
                          header_h, label_h_mm, simplify_m,
                          names=names, break_after=break_after,
                          row_paddings=row_paddings)
        if h > target:
            lo = mid  # islands too big → increase denominator
        else:
            hi = mid
        if hi - lo <= 500:
            break
    return hi


# ── rendering ─────────────────────────────────────────────────────────────────

def _fix_svg_right_alignment(svg_path, title_x_mm, page_w_mm):
    """
    Post-process a matplotlib SVG to add text-anchor:end for right-aligned text.

    Matplotlib with svg.fonttype='none' pre-computes left-edge positions for
    ha='right' text using its own font metrics, but emits no text-anchor attribute.
    Any renderer that uses different fonts (Illustrator, Inkscape, rsvg-convert)
    will then mis-align the text. This function converts those elements to use
    text-anchor:end at the correct data-coordinate x, so alignment is font-independent.

    Only fixes the title (font-weight:700) and subtitle (opacity:0.6) elements —
    not the island labels or subregion labels.
    """
    import re
    with open(svg_path) as f:
        content = f.read()

    m = re.search(r'width="([\d.]+)pt"', content)
    if not m:
        return
    svg_w_pt = float(m.group(1))
    left_off  = (svg_w_pt - page_w_mm * (72 / 25.4)) / 2
    title_svg_x = left_off + title_x_mm * (72 / 25.4)

    def _anchor_end(match):
        style = match.group(1)
        y     = match.group(2)
        text  = match.group(3)
        return (f'<text style="{style}; text-anchor: end"'
                f' x="{title_svg_x:.3f}" y="{y}">{text}</text>')

    # Title: bold weight
    content = re.sub(
        r'<text style="(font-weight: 700;[^"]*)"'
        r' transform="translate\([\d.]+ ([\d.]+)\)">(.*?)</text>',
        _anchor_end, content, flags=re.DOTALL,
    )
    # Subtitle: opacity 0.6 (distinct from subregion labels at 0.5)
    content = re.sub(
        r'<text style="([^"]*opacity: 0\.6[^"]*)"'
        r' transform="translate\([\d.]+ ([\d.]+)\)">(.*?)</text>',
        _anchor_end, content, flags=re.DOTALL,
    )

    with open(svg_path, "w") as f:
        f.write(content)


def draw_map(placed, final_geoms, page_w, page_h, args, render_mode="all"):
    """
    placed     : list of entry dicts (with 'island' sub-dict)
    final_geoms: pre-computed screen-space geometries (one per entry, already
                 y-flipped and compacted); if None, compute from entry left/top.
    """
    fig_w_in = page_w / 25.4
    fig_h_in = page_h / 25.4

    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    # y-axis: 0=top, page_h=bottom (screen coords, natural for layout)
    ax.set_xlim(0, page_w)
    ax.set_ylim(page_h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    if render_mode == "text_only":
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(args.bg)
        ax.set_facecolor(args.bg)

    pts_per_mm = 72 / 25.4
    debug_rank  = getattr(args, "debug_rank",  False)
    debug_bbox  = getattr(args, "debug_bbox",  False)
    debug_grid  = getattr(args, "debug_grid",  False)

    if debug_grid and render_mode != "text_only":
        from matplotlib.patches import Rectangle as _GridRect
        # Margin outline
        ax.add_patch(_GridRect(
            (args.margin, args.margin),
            page_w - 2 * args.margin, page_h - 2 * args.margin,
            fill=False, edgecolor="#CC0000", linewidth=0.5, zorder=20, alpha=0.85,
        ))
        # 1-inch = 25.4 mm grid
        for _xg in np.arange(25.4, page_w, 25.4):
            ax.plot([_xg, _xg], [0, page_h], color="#0044CC",
                    linewidth=0.15, alpha=0.3, zorder=19)
        for _yg in np.arange(25.4, page_h, 25.4):
            ax.plot([0, page_w], [_yg, _yg], color="#0044CC",
                    linewidth=0.15, alpha=0.3, zorder=19)
    do_hillshade = getattr(args, "hillshade", False)
    if do_hillshade:
        import dem_hillshade as _dem

    # Full-page Relievo hillshade overlay (pre-rendered PNG from export_composite_dem.py)
    relievo_path = getattr(args, "hillshade_relievo", None)
    if relievo_path and render_mode != "text_only":
        from PIL import Image as _PIL_Image
        import numpy as _np_r
        _raw = _np_r.array(_PIL_Image.open(relievo_path))
        if _raw.dtype == _np_r.uint16:
            # Blender outputs 16-bit PNG; PIL.convert("L") clips to 255 (all white).
            # Normalise directly from the uint16 range actually present in the file.
            _hs_img = _raw.astype(float)
            _hs_img = (_hs_img - _hs_img.min()) / (_hs_img.max() - _hs_img.min())
        else:
            _hs_img = _raw.astype(float) / 255.0
        ax.imshow(_hs_img, cmap="gray", vmin=0, vmax=1, origin="upper",
                  extent=(0, page_w, page_h, 0),
                  alpha=args.hillshade_alpha, interpolation="bilinear", zorder=4)

    # Label sizing: scaled to each island's shorter bbox dimension, clamped.
    #   lfs = clamp(shorter_dim_mm * pts_per_mm * label_scale, label_min, label_max)
    label_scale = args.label_scale
    label_min   = args.label_size       # --label-size is the floor
    label_max   = args.label_max

    north_up = not args.rotate_to_fit

    # Pre-compute per-row label size: use the minimum lfs across islands in each
    # row so all labels on the same row share a consistent size.
    row_lfs = {}
    for idx, entry in enumerate(placed):
        isl = entry["island"]
        geom = final_geoms[idx] if final_geoms is not None else place_geom(
            isl["geom"], entry["left"], entry["top"], north_up=north_up)
        b = geom.bounds
        shorter_mm = min(b[2] - b[0], b[3] - b[1])
        lfs = float(np.clip(shorter_mm * pts_per_mm * label_scale, label_min, label_max))
        row = entry.get("row", 0)
        row_lfs[row] = min(row_lfs.get(row, lfs), lfs)

    for idx, entry in enumerate(placed):
        isl = entry["island"]
        # Use pre-compacted geometry if available, else recompute
        if final_geoms is not None:
            geom = final_geoms[idx]
        else:
            geom = place_geom(isl["geom"], entry["left"], entry["top"], north_up=north_up)

        path = shapely_to_mpl_path(geom)
        if path is None:
            continue

        country = isl.get("country", "unknown")
        if country == "US":
            fill = args.fill_us
        elif country == "CA":
            fill = args.fill_ca
        else:
            fill = args.fill

        if render_mode != "text_only":
            ax.add_patch(PathPatch(
                path,
                facecolor=fill,
                edgecolor=args.stroke,
                linewidth=args.linewidth,
                zorder=2,
            ))

        if do_hillshade and render_mode != "text_only":
            try:
                geom_utm    = isl["geom_utm"]
                mm_per_m    = isl["mm_per_m"]
                elev, xc, yc, cell_m = _dem.fetch_elevation(
                    geom_utm, zoom=args.hillshade_zoom)
                hs = _dem.compute_hillshade(
                    elev, cell_m,
                    azimuth=args.hillshade_azimuth,
                    altitude=args.hillshade_altitude)
                hs = _dem.mask_to_polygon(hs, xc, yc, geom_utm)
                extent = _dem.page_extent(xc, yc, geom_utm, geom)
                ax.imshow(hs, cmap="gray", vmin=0, vmax=1,
                          origin="upper", extent=extent,
                          alpha=args.hillshade_alpha,
                          interpolation="bilinear", zorder=3)
            except Exception as e:
                print(f"  Hillshade failed for {isl['name']}: {e}")

        b = geom.bounds  # minx miny maxx maxy
        cx  = (b[0] + b[2]) / 2
        cy  = (b[1] + b[3]) / 2
        w_mm = b[2] - b[0]
        h_mm = b[3] - b[1]

        lfs = row_lfs[entry.get("row", 0)]
        sfs = lfs * 0.78

        # Gap below island scales with font size (all in mm, data-space units)
        gap_mm    = max(0.8, lfs / pts_per_mm * 0.4)
        line_mm   = lfs / pts_per_mm * 1.25   # leading between name and area lines
        label_y   = b[3] + gap_mm

        if debug_bbox and render_mode == "all":
            from matplotlib.patches import Rectangle
            label_h_mm = isl["h"] - isl["geom_h"]
            # blue = geom bbox
            ax.add_patch(Rectangle((b[0], b[1]), w_mm, h_mm,
                fill=False, edgecolor="blue", linewidth=0.5, zorder=5, alpha=0.7))
            # red = full packing box (geom + label)
            ax.add_patch(Rectangle((b[0], b[1]), w_mm, h_mm + label_h_mm,
                fill=False, edgecolor="red", linewidth=0.5, zorder=5, alpha=0.5,
                linestyle="--"))

        # ── debug: large semi-transparent rank number at centroid ─────────────
        if debug_rank and render_mode == "all":
            rank_fs = max(4.0, shorter_mm * pts_per_mm * 0.5)
            ax.text(cx, cy, str(isl["rank"]),
                    ha="center", va="center",
                    fontsize=rank_fs, fontfamily=args.font,
                    color=args.label_color, alpha=0.30, zorder=4,
                    fontweight="bold")

        if render_mode != "no_text":
            name = isl["name"] or "Unnamed"
            short_name = re.sub(r" Island$", "", name)
            short_name = re.sub(r"^(North|South|East|West) ", r"\1\n", short_name)
            ax.text(cx, label_y, short_name,
                    ha="center", va="top",
                    fontsize=lfs, fontfamily=args.font,
                    color=args.label_color, zorder=3)
            subregion = isl.get("subregion")
            if subregion:
                subregion_label = re.sub(r" (County|Regional District)$", r"\n\1", subregion)
                subregion_label = re.sub(r"^Regional District of (.+)",
                                         r"Regional District\nof \1", subregion_label)
                subregion_label = "\n".join(
                    textwrap.fill(ln, 18) if len(ln) > 18 else ln
                    for ln in subregion_label.split("\n")
                )
                name_lines = short_name.count("\n") + 1
                ax.text(cx, label_y + line_mm * name_lines,
                        subregion_label,
                        ha="center", va="top",
                        fontsize=sfs * 0.85, fontfamily=args.font,
                        color=args.label_color, alpha=0.50, zorder=3)

    # ── title / subtitle / scale bars (text layer only) ──────────────────────
    if render_mode == "no_text":
        plt.tight_layout(pad=0)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved → {args.output}")
        return

    # ── title (upper-right) ───────────────────────────────────────────────────
    pts_per_mm = 72 / 25.4
    title_text = args.title or f"The {args.top} Largest Islands in the Salish Sea"
    title_x = page_w - args.margin
    title_y = args.margin

    title_wrapped = "\n".join(
        line
        for para in title_text.split("\n")
        for line in (textwrap.wrap(para, args.title_wrap) if para.strip() else [""])
    )
    n_title_lines = title_wrapped.count("\n") + 1
    title_line_h = args.title_size / pts_per_mm * 1.25
    ax.text(title_x, title_y,
            title_wrapped,
            ha="right", va="top",
            fontsize=args.title_size, fontfamily=args.font,
            fontweight="bold", color=args.label_color, zorder=3,
            linespacing=1.25)

    sub_size   = args.title_size * 0.22
    sub_bottom = None
    sub_cx     = title_x - (args.title_wrap * sub_size / pts_per_mm * 0.52) / 2
    if args.subtitle:
        sub_chars_auto = int(args.title_wrap * (args.title_size / sub_size))
        sub_chars = getattr(args, "subtitle_wrap", None) or sub_chars_auto
        sub_wrapped = "\n".join(
            line
            for para in args.subtitle.split("\n")
            for line in (textwrap.wrap(para, sub_chars) if para.strip() else [""])
        )
        sub_top     = title_y + n_title_lines * title_line_h + 3.0
        n_sub_lines = sub_wrapped.count("\n") + 1
        sub_line_h  = sub_size / pts_per_mm * 1.4
        sub_bottom  = sub_top + n_sub_lines * sub_line_h
        # Estimate subtitle center-x for scale bar alignment
        longest_chars = max(len(l) for l in sub_wrapped.split("\n"))
        char_w_mm = sub_size / pts_per_mm * 0.52
        sub_cx = title_x - longest_chars * char_w_mm / 2
        ax.text(title_x, sub_top,
                sub_wrapped,
                ha="right", va="top",
                fontsize=sub_size, fontfamily=args.font,
                color=args.label_color, alpha=0.6, zorder=3,
                linespacing=1.4)

    # ── scale bars (centered under subtitle) ──────────────────────────────────
    from matplotlib.patches import Rectangle as MplRect
    bar_km = _nice_km(args.scale)
    km_mm  = bar_km * 1000 * (1000.0 / args.scale)
    mi_mm  = 10 * 1609.344 * (1000.0 / args.scale)

    bar_h      = 0.5
    lbl_gap    = 1.0
    lbl_fs     = max(args.label_size, sub_size * 0.85)
    lbl_line_h = lbl_fs / pts_per_mm * 1.4
    bars_top   = (sub_bottom + 4.0) if sub_bottom is not None \
                 else (title_y + n_title_lines * title_line_h + 8.0)

    # miles bar (first), centered at sub_cx
    bx_mi = sub_cx - mi_mm / 2
    ax.add_patch(MplRect((bx_mi, bars_top), mi_mm, bar_h,
                 facecolor=args.label_color, edgecolor="none", zorder=3))
    ax.text(bx_mi, bars_top + bar_h + lbl_gap, "0",
            ha="center", va="top", fontsize=lbl_fs,
            fontfamily=args.font, color=args.label_color, zorder=3)
    ax.text(bx_mi + mi_mm, bars_top + bar_h + lbl_gap, "10 mi",
            ha="center", va="top", fontsize=lbl_fs,
            fontfamily=args.font, color=args.label_color, zorder=3)

    # km bar (below miles), centered at sub_cx
    km_top = bars_top + bar_h + lbl_gap + lbl_line_h + 2.0
    bx_km  = sub_cx - km_mm / 2
    ax.add_patch(MplRect((bx_km, km_top), km_mm, bar_h,
                 facecolor=args.label_color, edgecolor="none", zorder=3))
    ax.text(bx_km, km_top + bar_h + lbl_gap, "0",
            ha="center", va="top", fontsize=lbl_fs,
            fontfamily=args.font, color=args.label_color, zorder=3)
    ax.text(bx_km + km_mm, km_top + bar_h + lbl_gap, f"{bar_km} km",
            ha="center", va="top", fontsize=lbl_fs,
            fontfamily=args.font, color=args.label_color, zorder=3)

    plt.tight_layout(pad=0)
    if render_mode == "text_only":
        import matplotlib as _mpl
        _old_fonttype = _mpl.rcParams.get("svg.fonttype")
        _mpl.rcParams["svg.fonttype"] = "none"
        _text_path = getattr(args, "text_overlay", None) or args.output
        fig.savefig(_text_path, dpi=args.dpi, bbox_inches="tight",
                    facecolor="none", transparent=True)
        _mpl.rcParams["svg.fonttype"] = _old_fonttype
        plt.close(fig)
        if _text_path.endswith(".svg"):
            _fix_svg_right_alignment(_text_path,
                                     title_x_mm=page_w - args.margin,
                                     page_w_mm=page_w)
        print(f"Saved text overlay → {_text_path}")
    else:
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved → {args.output}")


def _nice_km(scale_denom):
    """Round km value that makes a ~25–40 mm scale bar."""
    for n in [1, 2, 5, 10, 20, 25, 50, 100, 150, 200, 500]:
        if n * 1000 * (1000.0 / scale_denom) >= 20:
            return n
    return 50


# ── data helpers ──────────────────────────────────────────────────────────────

def split_rendezvous(gdf):
    """
    Replace the 'Rendezvous Islands' MultiPolygon with individual named rows.

    The three main islands (> 0.1 km²) are named by centroid latitude:
      northernmost → North Rendezvous Island
      middle       → Middle Rendezvous Island
      southernmost → South Rendezvous Island

    The ~18 tiny satellite rocks (< 0.1 km²) are dropped.
    All three named parts are returned so nlargest() decides who makes the cut.
    """
    mask = gdf["Name"] == "Rendezvous Islands"
    if not mask.any():
        return gdf

    row = gdf[mask].iloc[0]
    geom = row.geometry
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]

    main_parts = [p for p in parts if p.area / 1e6 > 0.1]
    main_parts.sort(key=lambda p: p.centroid.y)   # south → north

    ns_names = ["South Rendezvous Island", "Middle Rendezvous Island",
                "North Rendezvous Island"]
    new_rows = []
    for name, poly in zip(ns_names, main_parts):
        new_row = row.copy()
        new_row["Name"] = name
        new_row["geometry"] = poly
        new_row["Area_SqKm"] = poly.area / 1e6
        new_rows.append(new_row)

    gdf = gdf[~mask].copy()
    new_gdf = gpd.GeoDataFrame(new_rows, crs=gdf.crs)
    gdf = gpd.GeoDataFrame(
        gpd.pd.concat([gdf, new_gdf], ignore_index=True), crs=gdf.crs
    )
    areas = ", ".join(f"{r['Name']} ({r['Area_SqKm']:.3f} km²)" for _, r in new_gdf.iterrows())
    print(f"  Split Rendezvous Islands → {areas}")
    return gdf


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--shapefile",
                    default="Salish_Sea_Islands_Only/Salish_Sea_Islands_Only.shp")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--scale", type=int, default=None,
                    help="Map scale denominator (default: auto-fit to page)")
    ap.add_argument("--page-width",  type=float, default=420, dest="page_width")
    ap.add_argument("--page-height", type=float, default=594, dest="page_height")
    ap.add_argument("--padding",  type=float, default=5)
    ap.add_argument("--margin",   type=float, default=14)
    ap.add_argument("--row1", type=int, default=0, dest="row1_count",
                    help="Force exactly N islands onto row 1, overriding the "
                         "page-width limit (0 = auto)")
    ap.add_argument("--break-after", default="", dest="break_after",
                    help="Comma-separated island name fragments after which to "
                         "force a row break, e.g. 'fidalgo,galiano'")
    ap.add_argument("--exclude", default="", dest="exclude",
                    help="Comma-separated island names to remove before selecting "
                         "top-N, e.g. 'Harbor Island'")
    ap.add_argument("--simplify", type=float, default=0,
                    help="Geometry simplification in meters (0=off)")
    ap.add_argument("--wa-counties", default="WA_County_Boundaries/WA_County_Boundaries.shp",
                    dest="wa_counties",
                    help="Path to WA county shapefile for US/CA tagging (set to '' to skip)")
    ap.add_argument("--bc-regional-districts",
                    default="ABMS_REGIONAL_DISTRICTS_SP/ABMS_RD_polygon.shp",
                    dest="bc_regional_districts",
                    help="Path to BC regional districts shapefile (set to '' to skip)")
    ap.add_argument("--fill",    default="#C8D0BC",
                    help="Fill color for untagged islands (default #C8D0BC)")
    ap.add_argument("--fill-us", default="#B8C9A3", dest="fill_us",
                    help="Fill color for US islands (default #B8C9A3, warmer green)")
    ap.add_argument("--fill-ca", default="#C4CEC2", dest="fill_ca",
                    help="Fill color for CA islands (default #C4CEC2, cooler grey-green)")
    ap.add_argument("--stroke",      default="#3A4430")
    ap.add_argument("--bg",          default="#F4F1EA")
    ap.add_argument("--label-color", default="#1E2820", dest="label_color")
    ap.add_argument("--linewidth",   type=float, default=0.35)
    ap.add_argument("--label-size",  type=float, default=3.5,  dest="label_size",
                    help="Minimum label size in pt (default 3.5)")
    ap.add_argument("--label-max",   type=float, default=14.0, dest="label_max",
                    help="Maximum label size in pt (default 14)")
    ap.add_argument("--label-scale", type=float, default=0.18, dest="label_scale",
                    help="label_pt = clamp(shorter_bbox_mm * pts_per_mm * scale, "
                         "min, max). Default 0.18.")
    ap.add_argument("--title-size",  type=float, default=56.0, dest="title_size")
    ap.add_argument("--title-wrap",  type=int,   default=12,   dest="title_wrap",
                    help="Characters per line for title wrapping (default 12)")
    ap.add_argument("--font", default="DejaVu Sans")
    ap.add_argument("--title", default=None)
    _default_subtitle = (
        "CC-BY 2026 Alan McConchie, mappingmashups.net\n"
        "Data from the WWU Salish Sea Institute and the Mapterhorn project."
    )
    ap.add_argument("--subtitle", default=_default_subtitle)
    ap.add_argument("--subtitle-wrap", type=int, default=None, dest="subtitle_wrap",
                    metavar="N",
                    help="Characters per line for subtitle wrapping (default: auto from "
                         "title-wrap and font-size ratio). Lower values → narrower block.")
    ap.add_argument("--subtitle-append", default="", dest="subtitle_append",
                    metavar="TEXT",
                    help="Text appended to the subtitle as a new paragraph.")
    ap.add_argument("--output", default="islands.pdf")
    ap.add_argument("--dpi",    type=int, default=200)
    ap.add_argument("--include-unnamed", action="store_true", dest="include_unnamed")
    ap.add_argument("--rotate-to-fit", action="store_true", dest="rotate_to_fit",
                    help="Rotate each island to its most compact (landscape) orientation "
                         "for better packing. Disables north-up orientation.")
    ap.add_argument("--buffer", type=float, default=2.0,
                    help="Minimum polygon-edge gap in mm after kerning (default 2). "
                         "Set to 0 to skip the compaction pass.")
    ap.add_argument("--compact-simplify", type=float, default=0.5,
                    dest="compact_simplify",
                    help="Simplification tolerance (mm) used only during compaction "
                         "distance checks, for speed (default 0.5).")
    ap.add_argument("--row-buffers", default="", dest="row_buffers",
                    help="Comma-separated buffer overrides per row (1-indexed), "
                         "e.g. '12,8,5' sets row1=12mm, row2=8mm, row3=5mm; "
                         "remaining rows use --buffer")
    ap.add_argument("--row-paddings", default="", dest="row_paddings",
                    help="Comma-separated vertical gap overrides after each row (1-indexed), "
                         "e.g. '5,5,2' sets gap-after-row1=5mm, after-row2=5mm, after-row3=2mm; "
                         "remaining rows use --padding")
    ap.add_argument("--hillshade", action="store_true",
                    help="Overlay DEM hillshade from Mapterhorn tiles")
    ap.add_argument("--hillshade-relievo", default=None, dest="hillshade_relievo",
                    metavar="PNG",
                    help="Full-page Relievo/Blender hillshade PNG to overlay (alternative to "
                         "--hillshade; produced by export_composite_dem.py + relievo)")
    ap.add_argument("--hillshade-zoom",     type=int,   default=12,   dest="hillshade_zoom")
    ap.add_argument("--hillshade-alpha",    type=float, default=0.45, dest="hillshade_alpha",
                    help="Opacity of hillshade overlay (default 0.45)")
    ap.add_argument("--hillshade-azimuth",  type=float, default=315.0, dest="hillshade_azimuth",
                    help="Sun azimuth in degrees (default 315 = NW)")
    ap.add_argument("--hillshade-altitude", type=float, default=45.0,  dest="hillshade_altitude",
                    help="Sun altitude in degrees (default 45)")
    ap.add_argument("--text-overlay", default=None, dest="text_overlay",
                    metavar="PATH.svg",
                    help="If set, render islands+hillshade to --output (no text), then render "
                         "all text (labels, title, scale bar) to this SVG with editable vectors "
                         "(svg.fonttype=none, transparent background).")
    ap.add_argument("--snap-top", action="store_true", dest="snap_top",
                    help="After layout, shift the entire layout up so the topmost island "
                         "polygon sits exactly at the top margin.")
    ap.add_argument("--expand-to-margins", action="store_true", dest="expand_to_margins",
                    help="After layout, scale the entire result up until the widest "
                         "non-justified row just touches the page margins (capped by "
                         "vertical fit). Updates --scale accordingly for scale bars.")
    ap.add_argument("--target-fill", type=float, default=0.90, dest="target_fill",
                    metavar="F",
                    help="Fraction of page height the auto-scale targets (default 0.90). "
                         "Increase (e.g. 0.97) to pack islands larger.")
    ap.add_argument("--justify-from-row", type=int, default=None, dest="justify_from_row",
                    metavar="N",
                    help="1-indexed row number from which to fully justify islands "
                         "margin-to-margin with equal spacing (e.g. 4 justifies rows 4–end)")
    ap.add_argument("--justify-margin", type=float, default=None, dest="justify_margin",
                    metavar="MM",
                    help="Left/right margin in mm used for justified rows (default: same as "
                         "--margin). Useful to pull the justified rows in tighter than the "
                         "rest of the layout.")
    ap.add_argument("--justify-row2-margin", type=float, default=None,
                    dest="justify_row2_margin", metavar="MM",
                    help="If set, partially justify row 2 to this margin (wider than "
                         "--justify-margin gives a subtler spread than the main rows).")
    ap.add_argument("--debug-grid", action="store_true", dest="debug_grid",
                    help="Overlay a red margin rectangle and a 0.15pt blue 1-inch grid")
    ap.add_argument("--debug-bbox", action="store_true", dest="debug_bbox",
                    help="Overlay packing bounding boxes (blue=geom, red=geom+label)")
    ap.add_argument("--debug-rank", action="store_true", dest="debug_rank",
                    help="Overlay a large semi-transparent rank number at each "
                         "island centroid, sized to 50%% of the shorter bbox dim.")
    args = ap.parse_args()

    if args.title:
        args.title = args.title.replace("\\n", "\n")
    if args.subtitle:
        args.subtitle = args.subtitle.replace("\\n", "\n")
    if args.subtitle_append:
        args.subtitle_append = args.subtitle_append.replace("\\n", "\n")
    if args.subtitle_append:
        args.subtitle = (args.subtitle or "").rstrip() + "\n" + args.subtitle_append

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"Reading {args.shapefile} …")
    gdf = gpd.read_file(args.shapefile)

    if not args.include_unnamed:
        gdf = gdf[gdf["Name"].notna()]

    if args.exclude:
        exclude_names = {n.strip().lower() for n in args.exclude.split(",") if n.strip()}
        excluded = gdf["Name"].str.lower().isin(exclude_names)
        n_excluded = excluded.sum()
        gdf = gdf[~excluded]
        print(f"  Excluded {n_excluded} island(s): {args.exclude}")

    # ── split grouped MultiPolygons into named individuals ────────────────────
    gdf = split_rendezvous(gdf)

    top = gdf.nlargest(args.top, "Area_SqKm").reset_index(drop=True)
    print(f"  {len(gdf)} islands in dataset, using top {len(top)}")
    print(f"  Largest : {top.iloc[0]['Name']} ({top.iloc[0]['Area_SqKm']:.1f} km²)")
    print(f"  Smallest: {top.iloc[-1]['Name']} ({top.iloc[-1]['Area_SqKm']:.1f} km²)")

    # ── US / Canada tagging ───────────────────────────────────────────────────
    if args.wa_counties and args.wa_counties.strip():
        print(f"Tagging US/CA using {args.wa_counties} …")
        wa = gpd.read_file(args.wa_counties).to_crs(top.crs)
        wa_union = wa.union_all()
        top["country"] = top["geometry"].apply(
            lambda g: "US" if g.intersects(wa_union) else "CA"
        )
        us = (top["country"] == "US").sum()
        print(f"  US: {us}  CA: {len(top) - us}")

        # Assign each US island to the county with the largest intersection area.
        top["subregion"] = None
        name_col = "JURISDIC_3" if "JURISDIC_3" in wa.columns else None
        if name_col is None:
            print("  Warning: county name column not found — subregion labels skipped")
        else:
            for idx in top[top["country"] == "US"].index:
                isl_geom = top.loc[idx, "geometry"]
                best_name, best_area = None, 0.0
                for _, county in wa.iterrows():
                    inter = isl_geom.intersection(county.geometry)
                    if not inter.is_empty and inter.area > best_area:
                        best_area = inter.area
                        best_name = county[name_col]
                top.loc[idx, "subregion"] = best_name
            assigned = top["subregion"].notna().sum()
            print(f"  County lookup: {assigned} US islands assigned")
            unassigned = top[(top["country"] == "US") & top["subregion"].isna()]
            if len(unassigned):
                print(f"  Warning: no county found for: "
                      f"{', '.join(unassigned['Name'].tolist())}")
    else:
        top["country"] = "unknown"
        top["subregion"] = None

    # ── BC regional district lookup (CA islands) ──────────────────────────────
    if args.bc_regional_districts and args.bc_regional_districts.strip():
        print(f"Looking up BC regional districts using {args.bc_regional_districts} …")
        rd = gpd.read_file(args.bc_regional_districts).to_crs(top.crs)
        name_col = "AA_NAME" if "AA_NAME" in rd.columns else None
        if name_col is None:
            print("  Warning: regional district name column not found — skipped")
        else:
            for idx in top[top["country"] == "CA"].index:
                isl_geom = top.loc[idx, "geometry"]
                best_name, best_area = None, 0.0
                for _, district in rd.iterrows():
                    inter = isl_geom.intersection(district.geometry)
                    if not inter.is_empty and inter.area > best_area:
                        best_area = inter.area
                        best_name = district[name_col]
                top.loc[idx, "subregion"] = best_name
            assigned = top[top["country"] == "CA"]["subregion"].notna().sum()
            print(f"  Regional district lookup: {assigned} CA islands assigned")
            unassigned = top[(top["country"] == "CA") & top["subregion"].isna()]
            if len(unassigned):
                print(f"  Warning: no district found for: "
                      f"{', '.join(unassigned['Name'].tolist())}")

    raw_geoms = top["geometry"].tolist()

    # ── determine scale ───────────────────────────────────────────────────────
    # Title is now in the upper-right empty space — no reserved header strip.
    header_h = args.margin
    pts_per_mm = 72 / 25.4
    label_h_mm = (args.label_size / pts_per_mm) * 2.4 + 1.5  # 2 lines + padding

    raw_names = top["Name"].tolist()
    break_after = [n.strip() for n in args.break_after.split(",") if n.strip()]

    row_paddings = {}
    if args.row_paddings.strip():
        for r, val in enumerate(args.row_paddings.split(",")):
            if val.strip():
                row_paddings[r] = float(val.strip())

    if args.scale is None:
        print("Auto-scaling to fit page …")
        args.scale = autoscale(
            raw_geoms, args.page_width, args.page_height,
            args.margin, args.padding, header_h, label_h_mm,
            target_fill=args.target_fill, simplify_m=args.simplify,
            names=raw_names, break_after=break_after,
            row_paddings=row_paddings,
        )
        print(f"  → 1:{args.scale:,}")
    else:
        print(f"Using scale 1:{args.scale:,}")

    mm_per_meter = 1000.0 / args.scale

    # ── build island dicts ────────────────────────────────────────────────────
    print("Normalizing geometries …")
    if args.rotate_to_fit:
        print("  (rotate-to-fit: landscape orientation, north-up disabled)")
    islands = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        geom = normalize_geom(row.geometry, mm_per_meter, args.simplify,
                              rotate_landscape=args.rotate_to_fit)
        w, h = bbox_wh(geom)
        islands.append({
            "rank":      rank,
            "name":      row["Name"],
            "area":      row["Area_SqKm"],
            "country":   row.get("country", "unknown"),
            "subregion": row.get("subregion"),
            "group":     row.get("Isl_Group"),
            "arch":      row.get("Archipelag"),
            "geom":      geom,
            "geom_utm":  row.geometry,   # original UTM geometry for DEM fetch
            "mm_per_m":  mm_per_meter,
            "w":         w,
            "geom_h":    h,
            "h":         h + label_h_mm,
        })

    # ── pack and render ───────────────────────────────────────────────────────
    print("Packing layout …")
    placed, content_bottom = compute_layout(
        islands, args.page_width, args.margin, args.padding, header_h,
        row1_count=args.row1_count,
        break_after=break_after,
        row_paddings=row_paddings,
    )

    pct = content_bottom / args.page_height * 100
    print(f"  Content bottom: {content_bottom:.0f} mm  ({pct:.0f}% of {args.page_height:.0f} mm page)")
    rows_map = {}
    for e in placed:
        rows_map.setdefault(e["row"], []).append(e["island"])
    for r in sorted(rows_map):
        names = ", ".join(f"{isl['rank']}. {isl['name']}" for isl in rows_map[r])
        print(f"  Row {r+1}: {names}")

    # ── kerning / compaction pass ─────────────────────────────────────────────
    north_up = not args.rotate_to_fit

    print("Optimising row-1 vertical nudge …")
    placed = optimize_row1_nudge(
        placed, args.margin, header_h, args.buffer, north_up=north_up,
    )

    buffer_by_row = {}
    if args.row_buffers.strip():
        for r, val in enumerate(args.row_buffers.split(",")):
            val = val.strip()
            if val:
                buffer_by_row[r] = float(val)  # 0-indexed internally

    final_geoms = None
    if args.buffer > 0:
        if buffer_by_row:
            desc = ", ".join(f"row{r+1}={v}mm" for r, v in sorted(buffer_by_row.items()))
            print(f"Compacting with per-row buffers ({desc}; default {args.buffer}mm) …")
        else:
            print(f"Compacting with {args.buffer} mm polygon buffer …")
        placed, final_geoms = compact_placed(
            placed, args.margin,
            buffer_mm=args.buffer,
            north_up=north_up,
            simplify_tol=args.compact_simplify,
            buffer_by_row=buffer_by_row or None,
        )
        new_bottom = max(g.bounds[3] for g in final_geoms) + args.margin
        print(f"  Content bottom after compaction: {new_bottom:.0f} mm")

    print("Centering rows …")
    placed, final_geoms = center_rows(placed, final_geoms, args.page_width)

    jfr = (args.justify_from_row - 1) if args.justify_from_row is not None else None

    jm = args.justify_margin  # may be None → falls back to margin inside justify_rows

    if jfr is not None:
        print(f"Justifying rows {args.justify_from_row}+ "
              f"(margin={jm if jm is not None else args.margin:.1f} mm) …")
        placed, final_geoms = justify_rows(
            placed, final_geoms, args.page_width, args.margin,
            justify_from_row=jfr, justify_margin=jm,
        )

    if args.expand_to_margins:
        print("Expanding layout to margins …")
        placed, final_geoms, expand_f = expand_to_margins(
            placed, final_geoms, args.page_width, args.page_height, args.margin,
            justify_from_row=jfr,
        )
        print(f"  Scale factor: {expand_f:.4f}  ({(expand_f - 1) * 100:+.1f}%)")
        args.scale = int(args.scale / expand_f)
        print(f"  Effective scale: 1:{args.scale:,}")
        if jfr is not None:
            placed, final_geoms = justify_rows(
                placed, final_geoms, args.page_width, args.margin,
                justify_from_row=jfr, justify_margin=jm,
            )

    if args.justify_row2_margin is not None:
        placed, final_geoms = justify_rows(
            placed, final_geoms, args.page_width, args.margin,
            justify_from_row=1, to_row=2,
            justify_margin=args.justify_row2_margin,
        )

    if args.snap_top:
        top_y = min(g.bounds[1] for g in final_geoms)
        shift = top_y - args.margin
        if abs(shift) > 0.01:
            print(f"Snapping to top margin: shifting layout up {shift:.1f} mm …")
            final_geoms = [translate(g, 0, -shift) for g in final_geoms]
            for entry in placed:
                entry["top"] -= shift

    if args.text_overlay:
        print(f"Rendering map layer → {args.output}  (dpi={args.dpi}) …")
        draw_map(placed, final_geoms, args.page_width, args.page_height, args,
                 render_mode="no_text")
        print(f"Rendering text overlay → {args.text_overlay} …")
        draw_map(placed, final_geoms, args.page_width, args.page_height, args,
                 render_mode="text_only")
    else:
        print(f"Rendering → {args.output}  (dpi={args.dpi}) …")
        draw_map(placed, final_geoms, args.page_width, args.page_height, args)


if __name__ == "__main__":
    main()
