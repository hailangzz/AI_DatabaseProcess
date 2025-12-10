# blender_generate.py
# 用法（示例）:
# blender -b -P blender_generate.py -- --outdir ./output --num 100 --res 640 480

import bpy, bmesh
import random, sys, os, json, math
from mathutils import Vector, Euler

# -------- 参数（可从命令行覆盖） --------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='./output')
parser.add_argument('--num', type=int, default=100)
parser.add_argument('--res', nargs=2, type=int, default=[640,480])
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
random.seed(args.seed)

OUTDIR = os.path.abspath(args.outdir)
NUM = args.num
WIDTH, HEIGHT = args.res

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.join(OUTDIR, "rgb"), exist_ok=True)
os.makedirs(os.path.join(OUTDIR, "mask"), exist_ok=True)
os.makedirs(os.path.join(OUTDIR, "depth"), exist_ok=True)
os.makedirs(os.path.join(OUTDIR, "meta"), exist_ok=True)

# -------- 清空场景 --------
bpy.ops.wm.read_factory_settings(use_empty=True)

# -------- 渲染设置 --------
scene = bpy.context.scene
scene.render.engine = 'CYCLES'     # 或 'BLENDER_EEVEE'（若想更快）
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.cycles.samples = 32          # 可根据显卡调整
scene.view_layers["View Layer"].use_pass_object_index = True
scene.view_layers["View Layer"].use_pass_z = True

# -------- 辅助函数：新建材质 --------
def make_pbr_mat(name, base_color=(0.8,0.8,0.8,1.0), roughness=0.5, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    for n in nodes:
        nodes.remove(n)
    out = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    mat.node_tree.links.new(bsdf.outputs[0], out.inputs[0])
    return mat

# 地毯/地板材质样例（可扩展为贴图）
floor_mat = make_pbr_mat("floor", (0.6,0.55,0.5,1), roughness=0.7)
carpet_mat = make_pbr_mat("carpet", (0.2,0.15,0.1,1), roughness=0.9)

# 水渍材质（半透明+高光）
def make_water_mat():
    mat = bpy.data.materials.new("water")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.05,0.1,0.12,1)
    bsdf.inputs['Transmission'].default_value = 0.6
    bsdf.inputs['Roughness'].default_value = 0.05
    out = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(bsdf.outputs[0], out.inputs[0])
    return mat
water_mat = make_water_mat()

# -------- 创建地面平面 --------
bpy.ops.mesh.primitive_plane_add(size=5, location=(0,0,0))
ground = bpy.context.active_object
ground.name = "Ground"
ground.data.materials.append(floor_mat)
# 给 ground 设置 passive collision （如需物理模拟）
# ground.game.physics_type = 'STATIC'

# -------- 相机（贴地） --------
def make_camera():
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.lens = 8.0  # 低高度时使用较短焦距模拟广角
    # 把相机放在靠近地面的 Y 方向后面，稍微抬高
    cam.location = (0.0, -0.6, 0.06)  # 离地面 6cm 作为起点（可随机）
    cam.rotation_euler = Euler((math.radians(90-5), 0, 0), 'XYZ')  # 朝向平面
    scene.camera = cam
    return cam
cam = make_camera()

# -------- 灯光 --------
def randomize_lighting():
    # 删除旧光源
    for ob in [o for o in scene.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(ob, do_unlink=True)
    # 环境光（HDR 更真实，可替换为 HDRI）
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    # attach simple background brightness
    bg = world_nodes.get('Background', None)
    if bg:
        bg.inputs[1].default_value = random.uniform(0.2, 1.2)  # strength
    # 添加点光源随机位置
    for i in range(random.randint(1,3)):
        bpy.ops.object.light_add(type='POINT', location=(random.uniform(-1,1), random.uniform(-1,1), random.uniform(0.5,1.5)))
        l = bpy.context.active_object
        l.data.energy = random.uniform(300, 1500)
randomize_lighting()

# -------- 生成线材（曲线转 mesh） --------
def add_wire(idx):
    # 使用 Bézier 曲线
    bpy.ops.curve.primitive_bezier_curve_add(location=(random.uniform(-0.8,0.8), random.uniform(-0.8,0.8), 0.01))
    curve = bpy.context.active_object
    curve.name = f"wire_{idx}"
    curve.data.dimensions = '3D'
    # 随机化控制点
    for spline in curve.data.splines:
        for bp in spline.bezier_points:
            bp.co.z = 0.01
            bp.co.x += random.uniform(-0.3,0.3)
            bp.co.y += random.uniform(-0.3,0.3)
            bp.handle_left_type = 'AUTO'
            bp.handle_right_type = 'AUTO'
    # 转 mesh 并挤出圆截面（做成细长管）
    bpy.ops.object.convert(target='MESH')
    obj = bpy.context.active_object
    # 使用 solidify/modifier 或 bevel 来给线材粗细
    mod = obj.modifiers.new("bevel", type='BEVEL')
    mod.width = random.uniform(0.002, 0.01)
    # 赋材质
    mat = make_pbr_mat("wire_mat", (0.05,0.05,0.05,1), roughness=0.4)
    obj.data.materials.append(mat)
    # 设置 object index for mask
    obj.pass_index = idx+1
    return obj

# -------- 生成水渍（扁平 mesh，带透明材质） --------
def add_water_patch(idx):
    bpy.ops.mesh.primitive_circle_add(radius=random.uniform(0.02, 0.15), location=(random.uniform(-0.8,0.8), random.uniform(-0.8,0.8), 0.001))
    circle = bpy.context.active_object
    circle.name = f"water_{idx}"
    # 填充圈
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(circle.data)
    bmesh.ops.triangle_fill(bm, use_beauty=True, edges=bm.edges[:])
    bmesh.update_edit_mesh(circle.data)
    bpy.ops.object.mode_set(mode='OBJECT')
    circle.data.materials.append(water_mat)
    circle.pass_index = idx+1
    return circle

# -------- 生成地毯小块（plane） --------
def add_carpet_patch(idx):
    bpy.ops.mesh.primitive_plane_add(size=random.uniform(0.1, 0.6), location=(random.uniform(-0.8,0.8), random.uniform(-0.8,0.8), 0.002))
    patch = bpy.context.active_object
    patch.name = f"carpet_{idx}"
    # 随机旋转
    patch.rotation_euler = Euler((0,0,random.uniform(0,math.pi)), 'XYZ')
    patch.data.materials.append(carpet_mat)
    patch.pass_index = idx+1
    return patch

# -------- 渲染与输出 --------
def render_frame(i):
    # 清除除地面和相机以外的对象
    for ob in [o for o in scene.objects if o.name not in ("Ground", scene.camera.name)]:
        bpy.data.objects.remove(ob, do_unlink=True)

    instances = []
    # 随机生成若干线材 / 水渍 / 地毯
    obj_id = 0
    for k in range(random.randint(1,4)):  # wires
        obj = add_wire(obj_id)
        instances.append({"name":obj.name, "type":"wire", "pass_index": obj.pass_index})
        obj_id += 1
    for k in range(random.randint(0,3)):  # water
        obj = add_water_patch(obj_id)
        instances.append({"name":obj.name, "type":"water", "pass_index": obj.pass_index})
        obj_id += 1
    if random.random() < 0.6:
        obj = add_carpet_patch(obj_id)
        instances.append({"name":obj.name, "type":"carpet", "pass_index": obj.pass_index})
        obj_id += 1

    # 随机相机小抖动
    scene.camera.location.x = random.uniform(-0.15, 0.15)
    scene.camera.location.y = -random.uniform(0.3, 0.8)
    scene.camera.location.z = random.uniform(0.03, 0.12)  # 3cm ~ 12cm
    scene.camera.rotation_euler = Euler((math.radians(90-random.uniform(0,10)), 0, 0), 'XYZ')

    # 随机化光照
    randomize_lighting()

    # 设置输出路径
    rgb_path = os.path.join(OUTDIR, "rgb", f"{i:06d}.png")
    mask_path = os.path.join(OUTDIR, "mask", f"{i:06d}_idx.png")
    depth_path = os.path.join(OUTDIR, "depth", f"{i:06d}_depth.png")
    meta_path = os.path.join(OUTDIR, "meta", f"{i:06d}.json")

    # 设置 file output nodes
    scene.render.filepath = rgb_path
    bpy.ops.render.render(write_still=True)

    # 导出 object index pass（先确保开启）
    # 使用 Blender 的合成节点把 IndexOB 输出保存为 PNG
    tree = scene.node_tree
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new('CompositorNodeRLayers')
    file_node = tree.nodes.new('CompositorNodeOutputFile')
    file_node.base_path = os.path.join(OUTDIR, "mask")
    file_node.file_slots[0].path = f"{i:06d}_idx_"
    # 使用 ID Mask 节点保存 index（这里简单保存 Object Index pass）
    idmask = tree.nodes.new('CompositorNodeIDMask')
    # NOTE: For multiple objects with different pass_index you'd create separate file nodes per index.
    # For simplicity, here we just save the full index pass as a grayscale image:
    file_node.format.file_format = 'PNG'
    # connect Z or Index? Blender >=2.80 supports 'IndexOB' in RLayers -> use a Normalize node then file
    norm = tree.nodes.new('CompositorNodeNormalize')
    tree.links.new(rl.outputs['IndexOB'], norm.inputs[0])
    tree.links.new(norm.outputs[0], file_node.inputs[0])
    # render again to flush
    bpy.ops.render.render(write_still=True)

    # 保存 depth（Z pass）
    # Reconfigure nodes to save Z
    tree.nodes.clear()
    rl = tree.nodes.new('CompositorNodeRLayers')
    file_node = tree.nodes.new('CompositorNodeOutputFile')
    file_node.base_path = os.path.join(OUTDIR, "depth")
    file_node.file_slots[0].path = f"{i:06d}_depth_"
    # Z needs normalization to 0..1 for PNG
    norm = tree.nodes.new('CompositorNodeNormalize')
    tree.links.new(rl.outputs['Depth'], norm.inputs[0])
    tree.links.new(norm.outputs[0], file_node.inputs[0])
    file_node.format.file_format = 'PNG'
    bpy.ops.render.render(write_still=True)

    # 保存 metadata
    cam = scene.camera
    meta = {
        "img": os.path.relpath(rgb_path, OUTDIR),
        "width": WIDTH,
        "height": HEIGHT,
        "camera_location": list(cam.location),
        "camera_rotation_euler": list(cam.rotation_euler),
        "instances": instances
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print("Saved frame", i)

# -------- 主循环 --------
for i in range(NUM):
    render_frame(i)

print("All done")
