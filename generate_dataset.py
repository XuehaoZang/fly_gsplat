from pathlib import Path
import json
import numpy as np
import scipy.io as sio
import cv2
import h5py
from utils.camera import CameraConfig
from utils.dataset import generate_frame_dict
from utils.image import binarize_mask, crop_image, gray_to_rgba
from utils.calib import mask_centroid

def generate_dataset(_data_dir: str, _sparse_dir: str, target_frame: int, if_crop: bool = False, crop_size: int = 160) -> None:
    """
    Generate a Nerfstudio-compatible dataset from EasyWand calibration and sparse frame data.
    """
    # Path initialization and directory setup
    data_dir = Path(_data_dir)
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    mat_path = data_dir / "calibration_easyWandData.mat"
    json_path = data_dir / "transforms.json"
    
    # Path handling for server/local compatibility
    sparse_dir = Path(_sparse_dir.replace('X:', '/mnt/x').replace('\\', '/'))
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    sparse_files = sorted(list(sparse_dir.glob("Camera*_sparse.mat")))
    if not sparse_files:
        print("Error: No sparse files found.")
        return

    # Load Matlab EasyWand calibration data
    mat = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    ew_data = mat['easyWandData']
    n_cams = ew_data.nCams
    w_full = int(ew_data.imageWidth[0] if isinstance(ew_data.imageWidth, np.ndarray) else 1280) 
    h_full = int(ew_data.imageHeight[0] if isinstance(ew_data.imageHeight, np.ndarray) else 800) 

    frames = []
    for i in range(n_cams):
        cam_idx = i + 1
        img_name = f"P{target_frame}CAM{cam_idx}.png" 
        sparse_file = sparse_files[i]

        # Reconstruct image from sparse pixel data
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

        # calibration: RQ decomposition of EasyWand DLT coefs
        cam = CameraConfig.easywand_dlt(ew_data, i)

        # Dynamic Cropping
        if if_crop:
            u, v = mask_centroid(im)
            if np.isnan(u) or np.isnan(v):
                u, v = cam.cx, cam.cy
                print(f"[Warning] Cam {cam_idx}: empty mask, using principal point as crop centre")
            im, x_min, y_min = crop_image(im, u, v, crop_size=crop_size)
            cam.apply_crop(x_min, y_min, crop_size, crop_size)

        # Save processed image
        cv2.imwrite(str(img_dir / img_name), im)
        # gray_to_rgba(str(img_dir / img_name), str(img_dir / img_name))

        # Generate and save binary mask (white = train, black = ignore)
        mask = binarize_mask(im, threshold=1)
        mask_name = img_name
        cv2.imwrite(str(mask_dir / mask_name), mask)

        # Append frame metadata
        frame = generate_frame_dict(img_name, mask_name, cam)
        frames.append(frame)

    # Export metadata to JSON
    transforms = {
        "ply_file_path": "init_points.ply",
        "camera_model": "OPENCV",
        "frames": frames,
    }

    with open(json_path, 'w') as f:
        json.dump(transforms, f, indent=4)
    print(f"Dataset generated successfully at: {data_dir}")


if __name__ == "__main__":
    # sudo mount -t drvfs X: /mnt/x
    data_dir = r"./data/ctrl_009_002"
    # sparse_dir = r"X:\antenna\removed\002_26112024\Sparse\Expr_002_mov_009"
    sparse_dir = r"X:\antenna\control\009_25052026\Sparse\Expr_009_mov_002"
    generate_dataset(data_dir, sparse_dir, target_frame=10, if_crop=True, crop_size=160)