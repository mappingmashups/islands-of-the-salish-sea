# islands-of-the-salish-sea
100 Largest Islands of the Salish Sea (poster + scripts to regenerate it)

## Reproducing the poster

### Layout v1 (locked 2026-11-25, reference: `test_13x19b.svg`)

This is the canonical 8-row layout. Do not change island arrangement parameters
without updating the version label.

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
  --output poster_v1.png --dpi 200
```

Add `--hillshade-relievo hillshade_relievo.png --hillshade-alpha 0.5` for the
Blender hillshade overlay (requires running `export_composite_dem.py` and
`relievo` first — see project notes).
