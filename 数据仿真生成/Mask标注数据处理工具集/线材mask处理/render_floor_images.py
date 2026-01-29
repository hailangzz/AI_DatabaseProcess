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
    node_tree = mat.node_tree

    # 清空现有节点
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    # 添加 PBR 材质和输出节点
    bsdf = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    out = node_tree.nodes.new(type='ShaderNodeOutputMaterial')

    # 添加纹理节点并加载图像
    tex_image = node_tree.nodes.new(type='ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(image_path)

    # 将纹理颜色连接到 PBR 材质的 Base Color 输入
    node_tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    node_tree.links.new(bsdf.outputs[0], out.inputs[0])

    return mat


# ---------- Load local resources ----------
floor_folder = "/home/chenkejing/database/Floor/floor_background"
floor_images = list(Path(floor_folder).glob("*.jpg"))

# ---------- create ground + camera ----------
bpy.ops.mesh.primitive_plane_add(size=5, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"


def make_camera():
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "SimCam"
    cam.data.lens = 10.0
    cam.location = (0.0, 0.0, 3.0)  # 将相机放在地面正上方，3单位高度
    cam.rotation_euler = Euler((math.radians(90), 0, 0), 'XYZ')  # 垂直向下拍摄
    scene.camera = cam
    return cam


cam = make_camera()


# ---------- object generators ----------
def create_rectangle_frame(width=2, height=1):
    """
    在地面上绘制一个矩形框，默认大小为宽2、高1，位于场景的中心。
    """
    # 删除已有的矩形框（如果存在）
    if "RectangleFrame" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["RectangleFrame"], do_unlink=True)

    # 创建矩形的四个顶点
    verts = [
        (-width / 2, -height / 2, 0),  # 左下角
        (width / 2, -height / 2, 0),  # 右下角
        (width / 2, height / 2, 0),  # 右上角
        (-width / 2, height / 2, 0),  # 左上角
        (-width / 2, -height / 2, 0)  # 闭环：左下角
    ]

    # 定义连接这些顶点的边
    edges = [
        (0, 1),  # 左下到右下
        (1, 2),  # 右下到右上
        (2, 3),  # 右上到左上
        (3, 4)  # 左上到左下
    ]

    # 创建网格对象
    mesh = bpy.data.meshes.new(name="RectangleFrame")
    obj = bpy.data.objects.new("RectangleFrame", mesh)
    bpy.context.collection.objects.link(obj)

    # 创建网格并填充数据
    mesh.from_pydata(verts, edges, [])
    mesh.update()

    # 设置矩形框的显示材质为黄色
    mat = bpy.data.materials.new(name="YellowMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1, 1, 0, 1)  # 黄色 (R, G, B, A)
    obj.data.materials.append(mat)

    # 确保矩形框位于地面中心
    obj.location = (0, 0, 0)

    return obj


def add_light():
    """添加一个光源，以确保场景有足够的照明。"""
    bpy.ops.object.light_add(type='POINT', location=(0, 0, 3))
    light = bpy.context.active_object
    light.data.energy = 1000  # 设置光源亮度
    return light


# ---------- main render + label loop ----------
def render_images():
    for i in range(NUM):
        clear_scene_keep([ground.name, cam.name])

        # 每帧使用固定的地板贴图
        new_floor_img = str(random.choice(floor_images))  # 随机选择一个地板图片
        new_mat = make_pbr_mat_from_image("floor_mat", new_floor_img)

        # 确保 ground 对象有材质槽
        if len(ground.data.materials) == 0:
            ground.data.materials.append(new_mat)
        else:
            ground.data.materials[0] = new_mat

        # 在地面上绘制矩形框
        create_rectangle_frame(width=4, height=2)  # 绘制固定尺寸的矩形框

        # 设置相机位置
        cam.location.x = 0.0
        cam.location.y = 0.0
        cam.location.z = 3.0  # 相机在地面正上方
        cam.rotation_euler = Euler((math.radians(90), 0, 0), 'XYZ')  # 垂直向下拍摄

        # 添加光源
        add_light()

        # 渲染保存图片
        img_path = os.path.join(IMG_DIR, f"FloorImage_{i:06d}.png")
        scene.render.filepath = img_path
        bpy.ops.render.render(write_still=True)

        print(f"Saved image: {img_path}")


# ---------- run generation ----------
print("Start generation:", NUM, "images ->", OUTDIR)
render_images()
print("Generation finished.")
