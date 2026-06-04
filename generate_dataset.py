from pathlib import Path
from typing import List, Dict, Any
import json

import numpy as np
import scipy.io as sio
from scipy.linalg import rq
import cv2
import h5py
from utils import generate_frame_dict, crop_image, gray_to_rgba, extract_camera_params_from_P

# TODO now the sparse script is without the tracking
# TODO initialize from wand points or 3D hull?
# TODO now that camera axes are correct, the image flipping needs to be examined.

def generate_dataset(_data_dir: str, _sparse_dir: str, target_frame: int) -> None:
    """
    Generate a Nerfstudio-compatible dataset from EasyWand calibration and sparse frame data.
    """
    # 1. Path initialization and directory setup
    data_dir = Path(_data_dir)
    img_dir = data_dir / "images"
    mat_path = data_dir / "calibration_easyWandData.mat"
    json_path = data_dir / "transforms.json"
    
    # Path handling for server/local compatibility
    sparse_dir = Path(_sparse_dir.replace('X:', '/mnt/x').replace('\\', '/'))
    img_dir.mkdir(parents=True, exist_ok=True)

    sparse_files = sorted(list(sparse_dir.glob("Camera*_sparse.mat")))
    if not sparse_files:
        print("Error: No sparse files found.")
        return

    # 2. Load Matlab EasyWand calibration data
    mat = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    # print(f"mat import type:{type(mat)}")
    ew_data = mat['easyWandData']
    # print(f"ew data import type:{type(ew_data)}")
    # print(f"coefs precision: {ew_data.coefs.dtype}")
    n_cams = ew_data.nCams
    w_full = int(ew_data.imageWidth[0])
    h_full = int(ew_data.imageHeight[1] if isinstance(ew_data.imageHeight, np.ndarray) else ew_data.imageHeight) 

    frames = []
    for i in range(n_cams):
        cam_idx = i + 1
        img_name = f"P{target_frame}CAM{cam_idx}.png" 
        sparse_file = sparse_files[i]

        # 3. Reconstruct image from sparse pixel data
        with h5py.File(sparse_file, 'r') as sp:
            refs = sp['/frames/indIm'][0]
            indIm = sp[refs[target_frame]][:]
            if indIm.shape[0] == 3:
                indIm = indIm.T  
            
            frame_size = (h_full, w_full)
            im = np.zeros(frame_size, dtype=np.uint8)

            if indIm.size > 0:
                rows = indIm[:, 0].astype(int) - 1
                cols = indIm[:, 1].astype(int) - 1
                vals = indIm[:, 2].astype(float)
                
                # Boundary check and pixel assignment
                valid = (rows >= 0) & (rows < frame_size[0]) & (cols >= 0) & (cols < frame_size[1])
                im[rows[valid], cols[valid]] = vals[valid].astype(np.uint8)

        '''
        Calibration EasyWandData fields
        1. Intrinsics:
            1.1 using focalLengths, ppts=principalPoints(default center)
            1.2 using coefs to decompose K, R, X0, normalize K
            1.3 compare with Roni's code generated camera_KRX0.mat
        2. Extrinsics
            2.1 using DLTrotationMatrices.T --> get [R|T] --> gravity flip --> reasonable config, not intersecting beam
            2.2 old setup: using coefs to construct P, use inv(K) from 1.1 to get [R|T]
            2.3 new setup, hull recon: using rotationMatrices to get R - camera direction, DLTtranslationVector to get X0 - camera center
            2.4 Roni's version: using coefs and QR decomposition

        testing Roni's version of KRX0 mat file.
            
        '''
        # 1.1 Process Intrinsics
        fl_x_1 = np.float64(ew_data.focalLengths[i])
        fl_y_1 = np.float64(ew_data.focalLengths[i])
        cx_orig_1 = np.float64(ew_data.principalPoints[2*i]) - 1
        cy_orig_1 = np.float64(ew_data.principalPoints[2*i+1]) - 1
        K_1 = np.array([
            [fl_x_1, 0.0,  cx_orig_1],
            [0.0,  fl_y_1, cy_orig_1],
            [0.0,  0.0,  1.0]
        ])
        # print("[1] K from focolLengths and ppts:\n", K_1)
        
        # 2.1 using DLTrotationMatrices
        # R_dlt = ew_data.DLTrotationMatrices[:, :, i].T    # 4x4
        # R_w2c = R_dlt[0:3, 0:3]                        # roation matrix 3x3
        # R_c2w = R_w2c.T
        # X0 = R_dlt[0:3, 3]                         # camera center position X0 3x1

        # 2.2. using rotationMatrices, translationVector
        R_w2c = np.array(ew_data.rotationMatrices[:, :, i])      # 3x3
        R_c2w = R_w2c.T
        # print("[1] R_c2w from ew_data.rotationMatrices:\n", R_c2w)
        t = np.array(ew_data.translationVector[:, i])     # 3x1
        X0 = np.array(ew_data.DLTtranslationVector[:, i])     # 3x1
        # print("[1] X0 from ew_data.DLTtranslationVector:\n", X0)

        # 2.3 import .mat and create transform matrix
        # mat = sio.loadmat(str(data_dir / "camera_KRX0.mat"), struct_as_record=False, squeeze_me=True)
        # KRX0_data = mat['camera']
        # # print(KRX0_data[:,:,i])
        # K = KRX0_data[:,0:3,i]
        # K[0,2]  = K[0,2] - 1
        # K[1,2]  = K[1,2] - 1

        # R = KRX0_data[:,3:6,i].T
        # X0 = KRX0_data[:,6,i]
        
        # print("[DEBUG] K from Roni .mat data:\n", K)
        # print("[DEBUG] R from Roni .mat data:\n", R)
        # print("[DEBUG] X0 from Roni .mat data:\n", X0)

        # 1.2 decompose KRX0 from coefs using Roni's RQ decomposition
        P = np.append(ew_data.coefs[:, i], 1.0).reshape(3, 4)
        # print(f"Perspective mat from 11 coefs P=[H|h]:\n{P}")
        H = P[:, :3]  # 3x3
        # print(f"H 3x3:\n{H}")
        h = P[:, 3]  # 3x1
        # print(f"h 3x1:\n{h}")

        X0_2 = -np.linalg.inv(H) @ h
        K_2, R_w2c = rq(H)        # R_w2c: world to cam
        # print(f"decomposed R w2c:\n{R_w2c}")
        # the decomposition is not unique!! potentially with sign flip

        # align with roationMatrices direction
        ew_rot = ew_data.rotationMatrices[:, :, i]
        # print(f"ew_rotation matrix:\n{ew_rot}")
        change_ax_dir = np.sign(np.sum(ew_rot * R_w2c, axis=1))
        # print(f"change_ax_dir:\n{change_ax_dir}")
        Rot_to_ew = np.diag(change_ax_dir)
        # print(f"Rot_to_ew:\n{Rot_to_ew}")

        K_2 = K_2  @ Rot_to_ew
        K_2 = K_2 / K_2[2, 2]

        # # K_2[1,1] = - K_2[1,1]       #        K(2,2) = -K(2,2)
        # # K_2[1,2] = h_full - K_2[1,2]                        #K(2,3) = 801 -K(2,3)
        # # K_2 = K_2 / K_2[2, 2]
        # # print("[2] K after flipping:\n", K_2)

        R_w2c = Rot_to_ew @ R_w2c
        # print(f"[diag] change_ax_dir = {change_ax_dir}")
        # print(f"[diag] det(R_w2c) = {np.linalg.det(R_w2c):.4f}")
        print(f"[diag] det(R)={np.linalg.det(R_w2c):.3f}, Kdiag={np.diag(K_2)}")

        # print(f"after correction:\n{R_w2c}")
        R_2 = R_w2c.T

        ############################ NEW ##############################
        # F = np.array([[-1.0, 0.0, w_full - 1.0],
        #               [ 0.0, 1.0, 0.0],
        #               [ 0.0, 0.0, 1.0]])
        # K_2, R_w2c = rq(F @ H)
        # K_2 = K_2 / K_2[2, 2]
        # D = np.diag(np.sign(np.diag(K_2)))     # 强制对角为正,唯一化 rq 符号
        # K_2 = K_2 @ D
        # R_w2c = D @ R_w2c
        # R_2 = R_w2c.T
        # print(f"[diag] det(R)={np.linalg.det(R_w2c):.3f}, Kdiag={np.diag(K_2)}")
        ############################ NEW ##############################

        # print("[2] K from RQ decomposition:\n", K_2)
        # print("[2] R_c2w from RQ decomposition:\n", R_2)
        # print("[2] X0 from inv(H) @ h:\n", X0_2)

        # exit()

        # 3. use P=[H|h] and use rotationMatrix
        # P = np.append(ew_data.coefs[:, i], 1.0).reshape(3, 4)
        # H = P[:, :3]  # 3x3
        # h = P[:, 3]  # 3x1

        # R_w2c = np.array(ew_data.rotationMatrices[:, :, i])      # 3x3
        # R_c2w = R_w2c.T
        # t = np.array(ew_data.translationVector[:, i])     # 3x1
        # X0 = np.array(ew_data.DLTtranslationVector[:, i])     # 3x1

        # K = H @ R_w2c.T
        # print(K / K[2,2])


        # 5. Dynamic Cropping
        # DO_CROP = False
        # if DO_CROP:
        #     im, cx, cy, w, h = crop_image(im, cx_orig, cy_orig, crop_size=160)
        # else:
        #     cx, cy, w, h = cx_orig, cy_orig, w_full, h_full

        # Save processed image
        im = cv2.flip(im, 1)   # NEW 水平翻转，吸收反射
        cv2.imwrite(str(img_dir / img_name), im)
        # gray_to_rgba(str(img_dir / img_name), str(img_dir / img_name))

        # Append frame metadata
        frame = generate_frame_dict(img_name, w_full, h_full, K_2, R_2, X0_2)
        
        frames.append(frame)

    # 7. Export metadata to JSON
    transforms = {
        "ply_file_path": "init_points.ply",
        "camera_model": "OPENCV",
        "frames": frames,
    }

    with open(json_path, 'w') as f:
        json.dump(transforms, f, indent=4)


    print(f"Dataset generated successfully at: {data_dir}")


if __name__ == "__main__":
    # Example usage: Extract frame 1500 directly to our processed folder
    # sudo mount -t drvfs X: /mnt/x
    data_dir = r"./data/removed_002_009"
    sparse_dir = r"X:\antenna\removed\002_26112024\Sparse\Expr_002_mov_009"
    # sparse_dir = r"X:\antenna\control\009_25052026\Sparse\Expr_009_mov_002"
    generate_dataset(data_dir, sparse_dir, target_frame=2000)