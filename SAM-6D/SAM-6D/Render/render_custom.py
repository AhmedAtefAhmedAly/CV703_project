#!/usr/bin/env python3
import blenderproc as bproc
import os
import argparse
import cv2
import numpy as np
import trimesh
import random

# ----------------------------------
# ARGUMENTS
# ----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--cad_path',   help="The path of CAD model")
parser.add_argument('--output_dir', help="The path to save CAD templates")
parser.add_argument('--normalize',  default=True, help="Whether to normalize CAD model or not")
parser.add_argument('--colorize',   action='store_true', help="Whether to randomize PBR materials")
args = parser.parse_args()

# ----------------------------------
# INIT & CAMERA POSES LOADING
# ----------------------------------
render_dir     = os.path.dirname(os.path.abspath(__file__))
cnos_cam_fpath = os.path.join(
    render_dir,
    '../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy'
)
bproc.init()

def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')
    pts  = trimesh.sample.sample_surface(mesh, 1024)[0].astype(np.float32)
    r    = max(np.linalg.norm(pts.min(0)), np.linalg.norm(pts.max(0)))
    return 1/(2*r)

cam_poses = np.load(cnos_cam_fpath)
scale     = get_norm_info(args.cad_path) if args.normalize else 1.0

# ----------------------------------
# MULTI-VIEW RENDERING WITH COUNTER
# ----------------------------------
counter = 0
for cam_base in cam_poses:
    for _ in range(3):  # generate 3 slightly jittered views per pose
        bproc.clean_up()

        # Load & scale object
        obj = bproc.loader.load_obj(args.cad_path)[0]
        obj.set_scale([scale]*3)
        obj.set_cp('category_id', 1)

        # Apply random PBR material if requested
        if args.colorize:
            mat    = bproc.material.create(f'mat_{counter}')
            choice = random.choice(['black','white','metallic','reflective'])
            if choice == 'black':        base, rough, metal = [0,0,0,1], 0.8, 0.0
            elif choice == 'white':      base, rough, metal = [1,1,1,1], 0.8, 0.0
            elif choice == 'metallic':   base, rough, metal = [0.8,0.8,0.8,1], 0.2, 1.0
            else:                         base, rough, metal = [0.9,0.9,0.9,1], 0.1, 0.9
            mat.set_principled_shader_value('Base Color', base)
            mat.set_principled_shader_value('Roughness',   rough)
            mat.set_principled_shader_value('Metallic',    metal)
            obj.set_material(0, mat)

        # Camera pose jitter & transform
        cam_pose = cam_base.copy()
        jitter   = np.random.normal(scale=0.005, size=3)
        cam_pose[:3,-1] += jitter
        cam_pose[:3,1:3] = -cam_pose[:3,1:3]
        cam_pose[:3,-1] *= 0.002
        bproc.camera.add_camera_pose(cam_pose)

        # Choose background (grey, black, or checker)
        bg_choice = random.choice(['grey','black','checker'])
        if bg_choice == 'grey':
            bproc.renderer.set_world_background([0.2]*3, strength=0)
        elif bg_choice == 'black':
            bproc.renderer.set_world_background([0.0]*3, strength=0)
        else:
            bproc.renderer.set_world_background([0.2]*3, strength=0)

        # Single point light setup
        light_scale  = np.random.uniform(0.5, 5)
        light_energy = int(np.random.uniform(1000, 5000))
        light1 = bproc.types.Light()
        light1.set_type('POINT')
        light1.set_location([
            light_scale * cam_pose[0,-1],
            light_scale * cam_pose[1,-1],
            light_scale * cam_pose[2,-1]
        ])
        light1.set_energy(light_energy)

        # Render NOCS & color
        bproc.renderer.set_max_amount_of_samples(50)
        data = bproc.renderer.render()
        data.update(bproc.renderer.render_nocs())

        # Prepare output directory
        save_dir = os.path.join(args.output_dir, 'templates')
        os.makedirs(save_dir, exist_ok=True)

        # Save RGB with background overlay if needed
        rgb = data['colors'][0][..., :3][..., ::-1]
        if bg_choice == 'checker':
            mask = (data['nocs'][0][..., -1] * 255).astype(np.uint8)
            h,w   = mask.shape
            tile  = 32
            light_sq = np.array([180,180,180], dtype=np.uint8)
            dark_sq  = np.array([80,80,80],   dtype=np.uint8)
            ys = (np.arange(h)//tile)[:,None]
            xs = (np.arange(w)//tile)[None,:]
            pattern = ((ys+xs)%2==0).astype(np.uint8)
            checker = pattern[:,:,None]*light_sq + (1-pattern[:,:,None])*dark_sq
            bg_pixels = (mask==0)
            rgb[bg_pixels] = checker[bg_pixels]
        rgb_path = os.path.join(save_dir, f'rgb_{counter}.png')
        cv2.imwrite(rgb_path, rgb)

        # Save mask & NOCS using same counter
        mask = (data['nocs'][0][..., -1] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'mask_{counter}.png'), mask)
        xyz = 2*(data['nocs'][0][..., :3] - 0.5)
        np.save(os.path.join(save_dir, f'xyz_{counter}.npy'), xyz.astype(np.float16))

        print(f'Saved sample {counter} (bg={bg_choice})')
        counter += 1

if __name__=='__main__':
    pass
