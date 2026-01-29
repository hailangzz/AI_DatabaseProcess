import sys, os, random, math, json
import bpy
import bmesh
from mathutils import Vector, Euler
from bpy_extras.object_utils import world_to_camera_view
from pathlib import Path

# ---------- parse args ----------
import argparse

argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []
parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='./output')
parser.add_argument('--num', type=int, default=100)
parser.add_argument('--res', nargs=2, type=int, default=[640, 480])
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--samples', type=int, default=32, help='Cycles samples (set low for speed)')
parser.add_argument('--engine', type=str, default='CYCLES', choices=['CYCLES', 'BLENDER_EEVEE'])
args = parser.parse_args(argv)

OUTDIR = os.path.abspath(args.outdir)
NUM = int(args.num)
WIDTH = int(args.res[0])
HEIGHT = int(args.res[1])
SEED = int(args.seed)
SAMPLES = int(args.samples)
ENGINE = args.engine

random.seed(SEED)

IMG_DIR = os.path.join(OUTDIR, "images")
LBL_DIR = os.path.join(OUTDIR, "labels")
META_DIR = os.path.join(OUTDIR, "meta")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# ---------- basic blender scene reset ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# choose engine
if ENGINE == 'CYCLES':
    scene.render.engine = 'CYCLES'
    try:
        scene.cycles.device = 'GPU'
    except Exception:
        pass
elif ENGINE == 'BLENDER_EEVEE':
    scene.render.engine = 'BLENDER_EEVEE'

scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
# set samples if cycles
if scene.render.engine == 'CYCLES':
    scene.cycles.samples = SAMPLES

# ensure view layer passes
vl = scene.view_layers["View Layer"]
vl.use_pass_object_index = True
vl.use_pass_z = True


# ---------- helpers ----------
def ensure_world():
    """Ensure scene.world exists and has node tree with Background connected."""
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    if not scene.world.use_nodes:
        scene.world.use_nodes = True
    node_tree = scene.world.node_tree
    # clear nodes but be tolerant
    for n in list(node_tree.nodes):
        node_tree.nodes.remove(n)
    bg = node_tree.nodes.new(type='ShaderNodeBackground')
    out = node_tree.nodes.new(type='ShaderNodeOutputWorld')
    node_tree.links.new(bg.outputs['Background'], out.inputs['Surface'])
    return bg


def clear_scene_keep(names_keep):
    """Remove all objects except those named in names_keep"""
    for ob in list(scene.objects):
        if ob.name in names_keep:
            continue
        bpy.data.objects.remove(ob, do_unlink=True)


def make_pbr_mat_from_image(name, image_path):
    """Create a PBR material from an image texture."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    # clear nodes
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new(type='ShaderNodeBsdfPrincipled')

    # Load the image texture
    tex_image = nt.nodes.new(type='ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(image_path)

    nt.links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    return mat


# ---------- Load local resources ----------
floor_folder = "/home/chenkejing/database/Floor/floor_background"
wire_folder = "/home/chenkejing/database/WireDatabase/wire_object_images"  # Path to your wire samples folder

floor_images = list(Path(floor_folder).glob("*.jpg"))
wire_images = list(Path(wire_folder).glob("*.jpg"))

# ---------- create ground + camera ----------
bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
ground_mat = make_pbr_mat_from_image("floor_mat", str(random.choice(floor_images)))
ground.data.materials.append(ground_mat)


def make_camera():
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "SimCam"
    cam.data.lens = 10.0
    cam.location = (0.0, -0.6, 0.25)  # 相机的起始位置
    cam.rotation_euler = Euler((math.radians(90 - 5), 0, 0), 'XYZ')  # 固定旋转
    scene.camera = cam
    return cam


# Initialize camera (fixed position)
cam = make_camera()


# ---------- lighting ----------
def randomize_lighting():
    # ensure world
    bg = ensure_world()
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = random.uniform(0.05, 0.4)  # 世界环境能量强度
    # remove existing lights
    for ob in [o for o in scene.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(ob, do_unlink=True)
    # add 1-3 point lights
    for i in range(random.randint(1, 3)):
        bpy.ops.object.light_add(type='POINT',
                                 location=(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.6, 1.6)))
        l = bpy.context.active_object
        l.data.energy = random.uniform(20, 150)


# ---------- object generators ----------
def add_wire(idx):
    """Create a realistic wire lying on the ground, randomly bent and twisted."""
    wire_length = random.uniform(0.5, 3.0)
    wire_radius = random.uniform(0.004, 0.03)
    num_points = random.randint(5, 7)

    bpy.ops.curve.primitive_bezier_curve_add(location=(random.uniform(-1, 1), random.uniform(-1, 1), 0.02))
    curve_obj = bpy.context.active_object
    curve_obj.name = f"wire_{idx}"
    curve = curve_obj.data
    spline = curve.splines[0]
    spline.bezier_points.add(num_points - 2)
    points = spline.bezier_points

    for i, p in enumerate(points):
        t = i / (num_points - 1)
        y = t * wire_length
        x = random.uniform(-0.2, 0.2) * math.sin(t * math.pi * random.uniform(0.5, 2.0))
        z = random.uniform(0, 0.02) * math.sin(t * math.pi * random.uniform(1.0, 2.0))
        p.co = (x, y, z)
        p.handle_left_type = p.handle_right_type = 'AUTO'

    curve.bevel_depth = wire_radius
    curve.bevel_resolution = 8

    bpy.ops.object.convert(target='MESH')
    wire_obj = bpy.context.active_object

    wire_mat = make_pbr_mat_from_image(f"wire_mat_{idx}", str(random.choice(wire_images)))
    wire_obj.data.materials.append(wire_mat)

    wire_obj.pass_index = idx + 1
    wire_obj["cls"] = 0
    wire_obj.rotation_euler[2] = random.uniform(0, 2 * math.pi)

    return wire_obj


# ---------- main render + label loop ----------
def render_frame(i):
    clear_scene_keep(["Ground", cam.name])

    # 每帧使用固定的地板贴图（没有随机化）
    new_floor_img = str(random.choice(floor_images))
    new_mat = make_pbr_mat_from_image("floor_mat", new_floor_img)
    ground.data.materials[0] = new_mat

    # 创建线材对象
    instances = []
    obj_idx = 0
    n_wires = random.randint(1, 2)
    for k in range(n_wires):
        o = add_wire(obj_idx)
        instances.append(o)
        obj_idx += 1

    # 随机化场景照明
    randomize_lighting()

    # 渲染保存图片
    img_path = os.path.join(IMG_DIR, f"3DWire_{i:06d}.png")
    lbl_path = os.path.join(LBL_DIR, f"3DWire_{i:06d}.txt")
    meta_path = os.path.join(META_DIR, f"3DWire_{i:06d}.json")
    scene.render.filepath = img_path
    bpy.ops.render.render(write_still=True)

    # 处理标签
    labels = []
    meta_instances = []
    for obj in instances:
        if obj.type != 'MESH':
            continue
        bbox = object_bbox_yolo(obj, cam, scene)
        if bbox is None:
            continue
        cls = int(obj.get("cls", 0))
        x_c, y_c, w, h = bbox
        if w < 0.002 or h < 0.002:
            continue
        labels.append(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        meta_instances.append({"name": obj.name, "cls": cls, "bbox": [x_c, y_c, w, h]})

    # 保存标签和元数据
    with open(lbl_path, 'w') as f:
        f.write("\n".join(labels))

    meta = {
        "img": os.path.relpath(img_path, OUTDIR),
        "width": WIDTH,
        "height": HEIGHT,
        "camera_location": list(cam.location),
        "camera_rotation_euler": list(cam.rotation_euler),
        "instances": meta_instances
    }
    # with open(meta_path, 'w') as f:
    #     json.dump(meta, f, indent=2)


# ---------- run generation ----------
print("Start generation:", NUM, "images ->", OUTDIR)
for i in range(NUM):
    try:
        render_frame(i)
    except Exception as e:
        print("Error rendering frame", i, ":", e)
print("Generation finished.")

# YOLO data.yaml
data_yaml = {
    "train": os.path.join(OUTDIR, "images"),
    "val": os.path.join(OUTDIR, "images"),
    "nc": 1,
    "names": ["wire"]
}
with open(os.path.join(OUTDIR, "data.yaml"), 'w') as f:
    json.dump(data_yaml, f, indent=2)
print("Wrote data.yaml")
