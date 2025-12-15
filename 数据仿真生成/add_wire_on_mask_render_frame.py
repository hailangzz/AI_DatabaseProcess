import bpy, bmesh, random, math, json, os
from mathutils import Vector, Euler
from bpy_extras.object_utils import world_to_camera_view
from pathlib import Path

# ---------- 配置 ----------
COCO_JSON = "/home/chenkejing/database/Floor/Train_Data.v2i.coco/train/_annotations.coco.json"
IMG_DIR = "/home/chenkejing/database/Floor/Train_Data.v2i.coco/train"
WIRE_FOLDER = "/home/chenkejing/database/WireDatabase/Cable.v3-v3-wosplits.coco/train/imgs"
OUTDIR = "./output"


IMG_DIR = "/output/images"
LBL_DIR = "/output/labels"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)


NUM_IMAGES = 10
SEED = 42
random.seed(SEED)

# ---------- 加载地面标注 ----------
with open(COCO_JSON, 'r') as f:
    coco = json.load(f)

# 将 image_id -> segmentation 对应
ground_polygons = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    if img_id not in ground_polygons:
        ground_polygons[img_id] = []
    ground_polygons[img_id].append(ann["segmentation"][0])  # 假设 polygon 用第一个 segmentation

# ---------- 材质生成 ----------
def make_pbr_mat_from_image(name, image_path):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(image_path)
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    return mat

# ---------- 场景初始化 ----------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 32

# 创建相机
bpy.ops.object.camera_add(location=(0, -0.6, 0.25))
cam = bpy.context.active_object
cam.name = "SimCam"
cam.data.lens = 10.0
cam.rotation_euler = Euler((math.radians(90-5), 0, 0), 'XYZ')
scene.camera = cam

# ---------- 光照 ----------
def add_lighting():
    bpy.ops.object.light_add(type='POINT', location=(0,0,1))
    l = bpy.context.active_object
    l.data.energy = 120
add_lighting()

# ---------- 创建地面 plane ----------
def add_ground_plane(segmentation, image_path):
    """
    segmentation: [x1, y1, x2, y2, ...] in image space
    """
    # 计算 polygon 包围盒
    xs = segmentation[::2]
    ys = segmentation[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = (max_x - min_x)/416 * 5  # 映射到 Blender 单位 5x5 plane
    depth = (max_y - min_y)/416 * 5

    bpy.ops.mesh.primitive_plane_add(size=1, location=(0,0,0))
    plane = bpy.context.active_object
    plane.scale = (width/2, depth/2, 1)

    mat = make_pbr_mat_from_image("ground_mat", image_path)
    plane.data.materials.append(mat)
    return plane

# ---------- 创建墙面 ----------
def add_walls(ground_plane, wall_height=2.0):
    # 四周生成墙面 plane
    x, y, _ = ground_plane.location
    sx, sy, _ = ground_plane.scale
    walls = []
    coords = [
        ((x - sx, y, wall_height/2), (0, sy, wall_height)),  # 左
        ((x + sx, y, wall_height/2), (0, sy, wall_height)),  # 右
        ((x, y - sy, wall_height/2), (sx, 0, wall_height)),  # 前
        ((x, y + sy, wall_height/2), (sx, 0, wall_height)),  # 后
    ]
    for loc, scale in coords:
        bpy.ops.mesh.primitive_cube_add(location=loc)
        w = bpy.context.active_object
        w.scale = scale
        walls.append(w)
    return walls

# ---------- 添加线材 ----------
def add_wire(idx, ground_plane):
    wire_length = random.uniform(0.5, 3)
    wire_radius = random.uniform(0.004, 0.03)
    num_points = random.randint(5,7)

    bpy.ops.curve.primitive_bezier_curve_add(
        location=(random.uniform(-ground_plane.scale[0], ground_plane.scale[0]),
                  random.uniform(-ground_plane.scale[1], ground_plane.scale[1]), 0.02)
    )
    curve_obj = bpy.context.active_object
    curve = curve_obj.data
    spline = curve.splines[0]
    spline.bezier_points.add(num_points-2)
    points = spline.bezier_points
    for i, p in enumerate(points):
        t = i / (num_points-1)
        y = t * wire_length
        x = random.uniform(-0.2, 0.2) * math.sin(t*math.pi*random.uniform(0.5,2.0))
        z = random.uniform(0,0.02) * math.sin(t*math.pi*random.uniform(1.0,2.0))
        p.co = (x, y, z)
        p.handle_left_type = p.handle_right_type = 'AUTO'
    curve.bevel_depth = wire_radius
    curve.bevel_resolution = 8
    bpy.ops.object.convert(target='MESH')
    wire_obj = bpy.context.active_object
    wire_obj.pass_index = idx+1
    wire_obj["cls"] = 0
    wire_obj.rotation_euler[2] = random.uniform(0, 2*math.pi)
    return wire_obj

# ---------- YOLO bbox ----------
def mesh_world_vertices(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(deps)
    me = obj_eval.to_mesh()
    verts = [obj_eval.matrix_world @ v.co for v in me.vertices]
    obj_eval.to_mesh_clear()
    return verts

def object_bbox_yolo(obj, cam):
    verts = mesh_world_vertices(obj)
    coords = []
    for v in verts:
        co_ndc = world_to_camera_view(scene, cam, v)
        x, y, z = co_ndc.x, co_ndc.y, co_ndc.z
        if z <= 0: continue
        y_img = 1.0 - y
        coords.append((x, y_img))
    if not coords: return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x_min, x_max = max(0,min(xs)), min(1,max(xs))
    y_min, y_max = max(0,min(ys)), min(1,max(ys))
    w, h = x_max - x_min, y_max - y_min
    x_c, y_c = (x_min+x_max)/2, (y_min+y_max)/2
    return x_c, y_c, max(w,0.005), max(h,0.005)

# ---------- 渲染循环 ----------
for i in range(NUM_IMAGES):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 32
    # 随机选一张地面图
    img_idx = random.choice(list(ground_polygons.keys()))
    img_info = coco["images"][img_idx]
    img_file = os.path.join("/path/to/images", img_info["file_name"])
    seg = ground_polygons[img_idx][0]

    ground = add_ground_plane(seg, img_file)
    walls = add_walls(ground)
    wires = []
    n_wires = random.randint(1,4)
    for w_idx in range(n_wires):
        wires.append(add_wire(w_idx, ground))

    # 渲染设置
    scene.render.filepath = os.path.join(IMG_DIR, f"{i:06d}.png")
    bpy.ops.render.render(write_still=True)

    # YOLO 标签
    labels = []
    for w_obj in wires:
        bbox = object_bbox_yolo(w_obj, cam)
        if bbox:
            x_c, y_c, w, h = bbox
            labels.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    with open(os.path.join(LBL_DIR,f"{i:06d}.txt"),'w') as f:
        f.write("\n".join(labels))

    print(f"[{i+1}/{NUM_IMAGES}] rendered {img_file}, wires={len(wires)}")