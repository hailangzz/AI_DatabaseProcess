import bpy
import os
import json
import numpy as np
from PIL import Image
from math import radians

import cv2  # 确保已安装 opencv-python

# ---------------------------
# 创建只包含地面 mask 区域的平面并贴图
# ---------------------------
def create_ground_plane_from_mask(image_path, mask_path, plane_size=10.0):
    # 加载原图
    im = Image.open(image_path).convert("RGBA")
    w, h = im.size
    img_array = np.array(im)

    # 加载 mask polygon
    with open(mask_path, 'r') as f:
        mask_data = json.load(f)
    mask_poly = mask_data[0]['polygons'][0]  # Nx2
    mask_array = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_array, [np.array(mask_poly).reshape(-1, 2).astype(np.int32)], 1)

    # 创建 RGBA 图像：非地面区域透明
    img_array[..., 3] = mask_array * 255  # alpha 通道

    tmp_path = "/tmp/ground_mask_texture.png"
    Image.fromarray(img_array).save(tmp_path)

    # 创建 plane
    bpy.ops.mesh.primitive_plane_add(size=plane_size)
    plane = bpy.context.active_object

    # 缩放匹配原图比例
    plane.scale.x = plane_size * w / max(w, h) / 2
    plane.scale.y = plane_size * h / max(w, h) / 2

    # 材质
    mat = bpy.data.materials.new("GroundMaskMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(tmp_path)
    tex.image.colorspace_settings.name = 'sRGB'
    tex.interpolation = 'Smart'
    mat.node_tree.links.new(bsdf.inputs['Base Color'], tex.outputs['Color'])
    mat.node_tree.links.new(bsdf.inputs['Alpha'], tex.outputs['Alpha'])
    mat.blend_method = 'BLEND'
    mat.shadow_method = 'NONE'
    plane.data.materials.append(mat)

    # UV 映射
    bpy.ops.object.select_all(action='DESELECT')
    plane.select_set(True)
    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')

    return plane, mask_poly, w, h

# ---------------------------
# 渲染函数
# ---------------------------
def render_ground_only(images_dir, mask_dir, output_dir, render_resolution=512):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scene = bpy.context.scene
    scene.render.resolution_x = render_resolution
    scene.render.resolution_y = render_resolution
    scene.render.film_transparent = True  # 背景透明

    # 环境光：避免全黑
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes['Background']
    bg_node.inputs[1].default_value = 1.0  # 强度

    # 添加光源
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    light = bpy.context.active_object
    light.rotation_euler = (radians(45), 0, radians(45))
    light.data.energy = 5.0

    files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for file in files:
        img_path = os.path.join(images_dir, file)
        mask_path = os.path.join(mask_dir, os.path.splitext(file)[0] + ".json")
        if not os.path.exists(mask_path):
            print("No mask for", file)
            continue

        # 删除之前 mesh 和相机
        for obj in bpy.data.objects:
            if obj.type in ['MESH', 'CAMERA']:
                bpy.data.objects.remove(obj, do_unlink=True)

        # 创建只含地面的 plane
        plane, mask_poly, img_w, img_h = create_ground_plane_from_mask(img_path, mask_path)

        # -----------------------------
        # 添加相机（扫地机器人视角）
        # -----------------------------
        mask_points = np.array(mask_poly).reshape(-1, 2)
        min_y = mask_points[:,1].min()   # 最下边缘
        center_x = (mask_points[:,0].min() + mask_points[:,0].max()) / 2

        # 将 mask 像素坐标映射到 plane 坐标系
        plane_w = plane.scale.x * 2
        plane_h = plane.scale.y * 2

        scale_x = plane_w / img_w
        scale_y = plane_h / img_h

        cam_x = (center_x - img_w/2) * scale_x
        cam_y = (min_y - img_h/2) * scale_y
        cam_z = 1.2  # 高度，可根据实际调节

        bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
        cam = bpy.context.active_object
        # 俯视角度
        cam.rotation_euler = (radians(75), 0, 0)
        scene.camera = cam

        # -----------------------------
        # 渲染
        # -----------------------------
        out_path = os.path.join(output_dir, os.path.splitext(file)[0] + "_render.png")
        scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        print("Rendered", out_path)

# ---------------------------
# 执行
# ---------------------------
if __name__ == "__main__":
    images_dir = "/home/chenkejing/database/Floor/floor.v1i.coco/train"
    mask_dir = "/home/chenkejing/database/Floor/floor.v1i.coco/mask_contours"
    output_dir = "./renders"
    render_ground_only(images_dir, mask_dir, output_dir, render_resolution=512)
