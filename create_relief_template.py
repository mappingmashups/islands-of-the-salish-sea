"""
create_relief_template.py  — run inside Blender to create relief.blend

  /Applications/Blender.app/Contents/MacOS/Blender --background --python create_relief_template.py

Creates a Huffman-style Cycles displacement template:
  - Plane + Image Texture → Displacement node → Material Output
  - Diffuse BSDF gray surface so lighting shows relief without colour bias
  - Adaptive subdivision for true displacement + ray-traced shadows
  - Orthographic camera straight down
  - Sun light at NW 315° / altitude 35° (classic Huffman low-angle)
"""

import math
import sys
import pathlib

import bpy

OUTPUT = str(pathlib.Path(__file__).parent / "relief.blend")

# ── clean slate ───────────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
for col in list(bpy.data.collections):
    bpy.data.collections.remove(col)

scene = bpy.context.scene
scene.name = "Relief"

# ── plane ─────────────────────────────────────────────────────────────────────
bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
plane = bpy.context.active_object
plane.name = "Plane"

# Subdivision Surface + Cycles adaptive subdivision.
# In Blender 2.83 the adaptive flag lives on the object's Cycles settings,
# not on the modifier itself (it moved to the modifier in 2.90+).
subsurf = plane.modifiers.new(name="Subdivision", type="SUBSURF")
subsurf.subdivision_type = "SIMPLE"   # linear split, no corner rounding
subsurf.levels = 1
subsurf.render_levels = 1
plane.cycles.use_adaptive_subdivision = True   # Blender 2.83 API

# ── material ──────────────────────────────────────────────────────────────────
mat = bpy.data.materials.new("Relief")
mat.use_nodes = True
plane.data.materials.append(mat)

nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

# Image texture (DEM, loaded at render time by blender_script.py)
tex = nodes.new("ShaderNodeTexImage")
tex.name = "DEM_Texture"
tex.location = (-500, 100)
tex.interpolation = "Linear"

# Texture coordinate — use UV so the full image maps 1:1 to the plane
uv = nodes.new("ShaderNodeTexCoord")
uv.location = (-700, 100)
links.new(uv.outputs["UV"], tex.inputs["Vector"])

# Displacement
disp = nodes.new("ShaderNodeDisplacement")
disp.name = "Displacement"
disp.location = (-200, 100)
disp.inputs["Scale"].default_value = 1.0   # overridden by --exaggeration
disp.inputs["Midlevel"].default_value = 0.0

links.new(tex.outputs["Color"], disp.inputs["Height"])

# Gray diffuse surface (neutral so only lighting matters)
diffuse = nodes.new("ShaderNodeBsdfDiffuse")
diffuse.location = (-200, -150)
diffuse.inputs["Color"].default_value = (0.75, 0.75, 0.75, 1.0)

# Material output
out = nodes.new("ShaderNodeOutputMaterial")
out.location = (100, 0)
links.new(diffuse.outputs["BSDF"], out.inputs["Surface"])
links.new(disp.outputs["Displacement"], out.inputs["Displacement"])

# Displacement method (true geometry, not bump)
mat.cycles.displacement_method = "DISPLACEMENT"

# ── camera ────────────────────────────────────────────────────────────────────
bpy.ops.object.camera_add(location=(0, 0, 20))
cam = bpy.context.active_object
cam.name = "Camera"
cam.rotation_euler = (0.0, 0.0, 0.0)   # straight down (-Z)
cam.data.type = "ORTHO"
cam.data.ortho_scale = 2.0             # overridden at render time
scene.camera = cam

# ── sun light ─────────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type="SUN", location=(0, 0, 10))
sun = bpy.context.active_object
sun.name = "Sun"
sun.data.energy = 4.0
sun.data.angle = math.radians(8.0)   # angular diameter → soft, diffuse shadow edges
az  = 315.0
alt = 35.0
sun.rotation_euler = (math.radians(90.0 - alt), 0.0, math.radians(az))

# ── world (ambient) ───────────────────────────────────────────────────────────
world = bpy.data.worlds.new("World")
world.use_nodes = True
scene.world = world
bg = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
bg.inputs["Color"].default_value = (0.12, 0.12, 0.12, 1.0)
bg.inputs["Strength"].default_value = 0.7   # brighter ambient lifts shadow darkness

# ── render settings ───────────────────────────────────────────────────────────
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 256
scene.cycles.use_denoising = False
# Experimental: required for adaptive subdivision
scene.cycles.feature_set = "EXPERIMENTAL"
scene.cycles.dicing_rate = 1.0          # 1 pixel per subdivision edge — fine detail
scene.cycles.offscreen_dicing_scale = 16.0  # coarser outside camera view

# Output resolution — overridden per-DEM by blender_script.py
scene.render.resolution_x = 4096
scene.render.resolution_y = 4096
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_depth = "16"
scene.render.image_settings.color_mode = "BW"

# ── save ──────────────────────────────────────────────────────────────────────
bpy.ops.wm.save_as_mainfile(filepath=OUTPUT)
print(f"Saved → {OUTPUT}")
