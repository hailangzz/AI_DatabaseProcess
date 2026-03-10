import sys, os, random, math, json
import bpy
from mathutils import Euler
from bpy_extras.object_utils import world_to_camera_view
from pathlib import Path
import numpy as np
import argparse

# ---------- parse args ----------
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--")+1:]
else:
    argv = []

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='./output')
parser.add_argument('--num', type=int, default=100)
parser.add_argument('--res', nargs=2, type=int, default=[640,480])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--samples', type=int, default=32)
parser.add_argument('--engine', type=str, default='CYCLES', choices=['CYCLES','BLENDER_EEVEE'])
args = parser.parse_args(argv)

OUTDIR = os.path.abspath(args.outdir)
NUM = int(args.num)
WIDTH, HEIGHT = int(args.res[0]), int(args.res[1])
SEED = int(args.seed)
SAMPLES = int(args.samples)
ENGINE = args.engine

random.seed(SEED)
np.random.seed(SEED)

IMG_DIR = os.path.join(OUTDIR,"images")
LBL_DIR = os.path.join(OUTDIR,"labels")
META_DIR = os.path.join(OUTDIR,"meta")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# ---------- 横向线材距离范围 ----------
HORIZ_DIST_RANGE = (0.05, 0.15)

# ---------- Blender scene setup ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

if ENGINE=='CYCLES':
    scene.render.engine='CYCLES'
    try:
        scene.cycles.device='GPU'
    except:
        pass
elif ENGINE=='BLENDER_EEVEE':
    scene.render.engine='BLENDER_EEVEE'

scene.render.image_settings.file_format='PNG'
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT

if scene.render.engine=='CYCLES':
    scene.cycles.samples = random.randint(16,64)

# ---------- 渲染后随机模糊 ----------
scene.use_nodes = True
tree = scene.node_tree

for n in tree.nodes:
    tree.nodes.remove(n)

rlayers = tree.nodes.new('CompositorNodeRLayers')
blur = tree.nodes.new('CompositorNodeBlur')
comp = tree.nodes.new('CompositorNodeComposite')

blur.filter_type = 'GAUSS'
blur.size_x = random.randint(1,3)
blur.size_y = random.randint(1,3)

tree.links.new(rlayers.outputs['Image'], blur.inputs['Image'])
tree.links.new(blur.outputs['Image'], comp.inputs['Image'])

# ---------- helpers ----------
def ensure_world():
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    if not scene.world.use_nodes:
        scene.world.use_nodes=True
    nt = scene.world.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    bg = nt.nodes.new(type='ShaderNodeBackground')
    out = nt.nodes.new(type='ShaderNodeOutputWorld')
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    return bg

def clear_scene_keep(names_keep):
    for ob in list(scene.objects):
        if ob.name in names_keep: continue
        bpy.data.objects.remove(ob, do_unlink=True)

# ---------- 材质 ----------
def make_pbr_mat(name, base_color=(1,1,1), roughness=None, texture_path=None):

    mat = bpy.data.materials.new(name)
    mat.use_nodes=True
    nt = mat.node_tree

    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new(type='ShaderNodeBsdfPrincipled')

    bsdf.inputs['Base Color'].default_value = (*base_color,1.0)

    if roughness is not None:
        bsdf.inputs['Roughness'].default_value = roughness

    if texture_path:
        tex = nt.nodes.new(type='ShaderNodeTexImage')
        tex.image = bpy.data.images.load(texture_path)
        nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])

    # ---------- 表面随机纹理 ----------
    noise = nt.nodes.new(type='ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = random.uniform(50,200)

    bump = nt.nodes.new(type='ShaderNodeBump')
    bump.inputs['Strength'].default_value = random.uniform(0.02,0.08)

    nt.links.new(noise.outputs['Fac'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    nt.links.new(bsdf.outputs[0], out.inputs[0])

    return mat

# ---------- Resources ----------
floor_folder = "/home/chenkejing/database/Floor/floor_background"
wire_folder = "/home/chenkejing/database/WireDatabase/wire_object_images"

floor_images = list(Path(floor_folder).glob("*.jpg"))
wire_images = list(Path(wire_folder).glob("*.jpg"))

wire_colors = [
    (1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0.5,0),(0.5,0,1),(0,1,1)
]

# ---------- Ground ----------
bpy.ops.mesh.primitive_plane_add(size=5, location=(0,0,0))
ground = bpy.context.active_object
ground.name = "Ground"
ground.data.materials.append(make_pbr_mat("floor_mat", texture_path=str(random.choice(floor_images))))

# ---------- Camera ----------
def make_camera():

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object

    cam.name = "SimCam"
    cam.data.lens = 10.0

    cam.location = (0.0, -0.6, 0.25)

    # ---------- 景深 ----------
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = random.uniform(0.15,0.35)
    cam.data.dof.aperture_fstop = random.uniform(0.5,2.0)

    cam.rotation_euler = Euler((math.radians(85),0,0),'XYZ')

    scene.camera = cam

    return cam

cam = make_camera()

# ---------- Lighting ----------
def randomize_lighting():

    bg = ensure_world()

    # ---------- 随机环境亮度 ----------
    light_mode = random.choice(["dark","normal","normal","bright"])

    if light_mode == "dark":
        strength = random.uniform(0.008,0.13)   # 非常暗

    elif light_mode == "normal":
        strength = random.uniform(0.3,1.3)     # 普通室内光

    else:
        strength = random.uniform(2.0,4.0)     # 强光环境

    bg.inputs["Color"].default_value = (1,1,1,1)
    bg.inputs["Strength"].default_value = strength

    # ---------- 删除旧灯 ----------
    for ob in [o for o in scene.objects if o.type=='LIGHT']:
        bpy.data.objects.remove(ob, do_unlink=True)

    # ---------- 随机灯数量 ----------
    light_num = random.randint(0,3)

    for i in range(light_num):

        bpy.ops.object.light_add(
            type=random.choice(['POINT','AREA']),
            location=(
                random.uniform(-1,1),
                random.uniform(-1,1),
                random.uniform(0.6,1.6)
            )
        )

        l = bpy.context.active_object

        # ---------- 灯光强度 ----------
        if light_mode == "dark":
            l.data.energy = random.uniform(5,30)

        elif light_mode == "normal":
            l.data.energy = random.uniform(30,150)

        else:
            l.data.energy = random.uniform(150,500)

# ---------- Wire generator ----------
def add_wire(idx, direction=None):

    wire_length = random.uniform(0.7,1.5)
    roughness = random.uniform(0.2,0.9)
    num_points = random.randint(5,7)

    if direction is None:
        direction = random.choice(['y','x'])

    if direction == 'y':

        loc_x = random.uniform(-0.2,0.2)
        # loc_y = random.uniform(0.5,-0.4)
        loc_y = random.uniform(-0.4,0.5)
    else:

        loc_x = random.uniform(-0.2,0.2)
        loc_y = random.uniform(*HORIZ_DIST_RANGE)

    dist = abs(loc_y - cam.location.y)

    dist_norm = min(max(dist / 0.8, 0), 1)

    wire_radius = 0.008 + dist_norm * (0.03 - 0.008)

    bpy.ops.curve.primitive_bezier_curve_add(location=(loc_x, loc_y, 0.02))

    curve_obj = bpy.context.active_object
    curve_obj.name=f"wire_{idx}"

    spline = curve_obj.data.splines[0]
    spline.bezier_points.add(num_points-2)

    points = spline.bezier_points

    positions = np.linspace(0, wire_length, num_points)

    bend_factor = random.uniform(0.5,1.5)

    for i,p in enumerate(points):

        t = i/(num_points-1)

        z = random.uniform(0,0.02) * math.sin(t*math.pi*bend_factor)

        if direction == 'y':
            x = random.uniform(-0.15,0.15) * math.sin(t*math.pi*bend_factor)
            y = positions[i]
        else:
            x = positions[i]
            y = random.uniform(-0.15,0.15) * math.sin(t*math.pi*bend_factor)

        p.co = (x,y,z)
        p.handle_left_type = p.handle_right_type = 'AUTO'

    curve_obj.data.bevel_depth = wire_radius
    curve_obj.data.bevel_resolution = 8

    bpy.ops.object.convert(target='MESH')

    wire_obj = bpy.context.active_object

    color = random.choice(wire_colors)

    use_texture = wire_images and random.random()<0.5
    tex_path = str(random.choice(wire_images)) if use_texture else None

    mat = make_pbr_mat(
        f"wire_mat_{idx}",
        base_color=color,
        roughness=roughness,
        texture_path=tex_path
    )

    wire_obj.data.materials.append(mat)

    wire_obj.pass_index = idx+1
    wire_obj["cls"] = 0

    wire_obj.rotation_euler[2] = random.uniform(0,2*math.pi)

    return wire_obj

# ---------- Render loop ----------
def render_frame(i):

    clear_scene_keep(["Ground", cam.name])

    ground.data.materials[0] = make_pbr_mat(
        "floor_mat",
        texture_path=str(random.choice(floor_images))
    )

    instances=[]

    n_wires = random.randint(1,2)

    n_y = n_wires // 2
    n_x = n_wires - n_y

    idx = 0

    for _ in range(n_y):
        instances.append(add_wire(idx,'y'))
        idx+=1

    for _ in range(n_x):
        instances.append(add_wire(idx,'x'))
        idx+=1

    cam.location.x = random.uniform(-0.1,0.1)
    cam.location.y = random.uniform(-0.4,-0.2)
    cam.location.z = random.uniform(0.2,0.3)

    cam.rotation_euler = Euler((math.radians(90-random.uniform(0,10)),0,0),'XYZ')

    randomize_lighting()

    img_path = os.path.join(IMG_DIR,f"3DWire_{i:06d}.png")

    scene.render.filepath = img_path

    bpy.ops.render.render(write_still=True)

# ---------- Run ----------
print("Start generation:",NUM,"images ->",OUTDIR)

for i in range(NUM):

    try:
        render_frame(i)

    except Exception as e:
        print("Error rendering frame",i,":",e)

print("Generation finished.")