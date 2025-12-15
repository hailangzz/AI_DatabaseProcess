import sys, os, random, math, json
import bpy
from mathutils import Euler
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
parser.add_argument('--samples', type=int, default=32)
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

# ---------- Blender scene reset ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# render engine
if ENGINE == 'CYCLES':
    scene.render.engine = 'CYCLES'
    try:
        scene.cycles.device = 'GPU'
    except:
        pass
else:
    scene.render.engine = 'BLENDER_EEVEE'

scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
if scene.render.engine == 'CYCLES':
    scene.cycles.samples = SAMPLES

vl = scene.view_layers["View Layer"]
vl.use_pass_object_index = True
vl.use_pass_z = True

# ---------- helpers ----------
def ensure_world():
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    if not scene.world.use_nodes:
        scene.world.use_nodes = True
    nt = scene.world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg = nt.nodes.new(type='ShaderNodeBackground')
    out = nt.nodes.new(type='ShaderNodeOutputWorld')
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])
    return bg

def clear_scene_keep(names_keep):
    for ob in list(scene.objects):
        if ob.name in names_keep:
            continue
        bpy.data.objects.remove(ob, do_unlink=True)

def make_pbr_mat_from_image(name, image_path, scale_uv=(1, 1)):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new(type='ShaderNodeBsdfPrincipled')
    tex_image = nt.nodes.new(type='ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(image_path)

    tex_coord = nt.nodes.new(type='ShaderNodeTexCoord')
    mapping = nt.nodes.new(type='ShaderNodeMapping')
    mapping.inputs['Scale'].default_value[0] = scale_uv[0]
    mapping.inputs['Scale'].default_value[1] = scale_uv[1]
    nt.links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    nt.links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])

    nt.links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    return mat

# ---------- Load resources ----------
floor_folder = "/home/chenkejing/database/Floor/floor_background"
carpet_folder = "/home/chenkejing/database/carpetDatabase/carpet_object_images"

floor_images = list(Path(floor_folder).glob("*.jpg"))
carpet_images = list(Path(carpet_folder).glob("*.jpg"))

# ---------- Ground plane ----------
bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
ground_mat = make_pbr_mat_from_image("floor_mat", str(random.choice(floor_images)))
ground.data.materials.append(ground_mat)

# ---------- Camera ----------
def make_camera():
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "SimCam"
    cam.data.lens = 10.0
    cam.location = (0.0, -0.6, 0.25)
    cam.rotation_euler = Euler((math.radians(85), 0, 0), 'XYZ')
    scene.camera = cam
    return cam

cam = make_camera()

# ---------- Lighting ----------
def randomize_lighting():
    bg = ensure_world()
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = random.uniform(0.05, 0.4)
    for ob in [o for o in scene.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(ob, do_unlink=True)
    for i in range(random.randint(1, 3)):
        bpy.ops.object.light_add(type='POINT',
                                 location=(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(0.6, 1.6)))
        l = bpy.context.active_object
        l.data.energy = random.uniform(20, 150)

# ---------- Carpet generator with min distance ----------
def add_carpet(idx, cam, min_dist=0.5):
    carpet_w = random.uniform(3, 5.0)
    carpet_h = random.uniform(3, 5.0)
    carpet_thickness = random.uniform(0.03, 0.05)

    for _ in range(20):
        x_loc = random.uniform(-2, 2)
        y_loc = random.uniform(-2, 2)
        z_loc = carpet_thickness / 2
        dist = math.sqrt((x_loc - cam.location.x)**2 + (y_loc - cam.location.y)**2 + (z_loc - cam.location.z)**2)
        if dist >= min_dist:
            break

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_loc, y_loc, z_loc))
    carpet = bpy.context.active_object
    carpet.name = f"carpet_{idx}"
    carpet.scale.x = carpet_w / 2
    carpet.scale.y = carpet_h / 2
    carpet.scale.z = carpet_thickness / 2

    mat = make_pbr_mat_from_image(f"carpet_mat_{idx}", str(random.choice(carpet_images)))
    carpet.data.materials.append(mat)

    carpet.pass_index = idx + 1
    carpet["cls"] = 0
    carpet.rotation_euler[2] = random.uniform(0, 2 * math.pi)

    return carpet

# ---------- YOLO bbox with padding and clipping ----------
def object_bbox_yolo(obj, cam, scene, min_size=0.01, padding=0.02):
    deps = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(deps)
    me = obj_eval.to_mesh()
    verts_world = [obj_eval.matrix_world @ v.co for v in me.vertices]
    obj_eval.to_mesh_clear()
    coords = []
    for v in verts_world:
        co_ndc = world_to_camera_view(scene, cam, v)
        x, y, z = co_ndc.x, co_ndc.y, co_ndc.z
        if z <= 0:
            continue
        y_img = 1.0 - y
        coords.append((x, y_img))
    if not coords:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min = max(0.0, min(xs) - padding)
    x_max = min(1.0, max(xs) + padding)
    y_min = max(0.0, min(ys) - padding)
    y_max = min(1.0, max(ys) + padding)
    w = min(max(x_max - x_min, min_size), 1.0)
    h = min(max(y_max - y_min, min_size), 1.0)
    x_c = (x_min + x_max) / 2
    y_c = (y_min + y_max) / 2
    return (x_c, y_c, w, h)

# ---------- Render frame ----------
def render_frame(i):
    clear_scene_keep(["Ground", cam.name])
    ground.data.materials[0] = make_pbr_mat_from_image("floor_mat", str(random.choice(floor_images)))

    instances = []
    for idx in range(1):
        o = add_carpet(idx, cam, min_dist=0.5)
        instances.append(o)

    cam.location.x = random.uniform(-0.15, 0.15)
    cam.location.y = -random.uniform(0.3, 0.8)
    cam.location.z = random.uniform(0.65, 0.99)
    cam.rotation_euler = Euler((math.radians(90 - random.uniform(0, 10)), 0, 0), 'XYZ')
    randomize_lighting()

    img_path = os.path.join(IMG_DIR, f"{i:06d}.png")
    lbl_path = os.path.join(LBL_DIR, f"{i:06d}.txt")
    meta_path = os.path.join(META_DIR, f"{i:06d}.json")
    scene.render.filepath = img_path
    bpy.ops.render.render(write_still=True)

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
        labels.append(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        meta_instances.append({"name": obj.name, "cls": cls, "bbox": [x_c, y_c, w, h]})

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
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[{i + 1}/{NUM}] saved image={img_path} labels={len(labels)}")

# ---------- run generation ----------
print("Start generation:", NUM, "images ->", OUTDIR)
for i in range(NUM):
    try:
        render_frame(i)
    except Exception as e:
        print("Error rendering frame", i, ":", e)
print("Generation finished.")

# ---------- YOLO data.yaml ----------
data_yaml = {
    "train": os.path.join(OUTDIR, "images"),
    "val": os.path.join(OUTDIR, "images"),
    "nc": 1,
    "names": ["carpet"]
}
with open(os.path.join(OUTDIR, "data.yaml"), 'w') as f:
    json.dump(data_yaml, f, indent=2)
print("Wrote data.yaml")
