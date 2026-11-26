# islands-of-the-salish-sea
100 Largest Islands of the Salish Sea (poster + scripts to regenerate it)

## Reproducing the poster (v1.0.0)

### Step 1 — Layout + labels

```bash
python3 draw_islands.py \
  --page-width 330.2 --page-height 482.6 \
  --margin 12.7 \
  --exclude "Harbor Island,Penn Islands,Copeland Islands,Secretary Islands" \
  --title-size 42 --title-wrap 25 \
  --subtitle-wrap 54 \
  --break-after "west redonda,nelson,maurelle,sidney,newcastle,coal" \
  --row-buffers "12,8,4,4,4,4,4,4,4" \
  --row-paddings "12,8,4,5,8,10,10,10" \
  --label-size 7 \
  --justify-from-row 3 \
  --justify-margin 19.05 \
  --justify-row2-margin 28 \
  --expand-to-margins \
  --snap-top \
  --label-offsets "Whidbey Island:0,-12.7;Texada Island:12.7,-5;Quadra Island:12.7,0;West Redonda Island:6.35,0;Orcas Island:0,6.35;Camano Island:0,-6.35;Galiano Island:0,-1.35;Valdes Island:0,-1.35;Lummi Island:0,-3.175;North Thormanby Island:0,1.175" \
  --label-no-split "West Redonda Island" \
  --hillshade-relievo hillshade_relievo_az200_exag02.png \
  --hillshade-blend multiply --hillshade-alpha 0.5 \
  --hillshade-shadow 0.2 --hillshade-highlight 0.55 \
  --cc-badge cc_by_sa.png \
  --output poster_v1.png --dpi 300
```

### Step 2 — Build the Blender hillshade (required before Step 1)

**2a. Export composite DEM** (uses same layout parameters as draw_islands.py):

```bash
.venv/bin/python3 export_composite_dem.py \
  --page-width 330.2 --page-height 482.6 \
  --margin 12.7 \
  --exclude "Harbor Island,Penn Islands,Copeland Islands,Secretary Islands" \
  --break-after "west redonda,nelson,maurelle,sidney,newcastle,coal" \
  --row-buffers "12,8,4,4,4,4,4,4,4" \
  --row-paddings "12,8,4,5,8,10,10,10" \
  --label-size 7 \
  --justify-from-row 3 \
  --justify-margin 19.05 \
  --justify-row2-margin 28 \
  --expand-to-margins \
  --snap-top \
  --output composite_dem.tif
```

**2b. Add +30 m elevation bias** (gives flat islands a cliff edge for cast shadows):

```python
# python3 add_elevation_bias.py  — or run inline:
import numpy as np
from osgeo import gdal
# ... see project notes
```

**2c. Run Relievo/Blender**:

```bash
.venv/bin/relievo \
  --dem composite_dem_plus30.tif \
  --template relief.blend \
  --output hillshade_relievo_az200_exag02.png \
  --blender /Applications/Blender.app/Contents/MacOS/Blender \
  --light-azimuth 200 --light-altitude 10 \
  --exaggeration 0.2 --max-size 4096 --samples 64 \
  --filter-values "0:"
```

### Layout row breakdown (8 rows)

| Row | Islands |
|-----|---------|
| 1 | Whidbey → West Redonda |
| 2 | Sonora → Nelson |
| 3 | Camano → Maurelle |
| 4 | Gabriola → Anderson |
| 5 | Shaw → Sidney |
| 6 | Decatur → Newcastle |
| 7 | Mink → Coal |
| 8 | Protection → Johns |
