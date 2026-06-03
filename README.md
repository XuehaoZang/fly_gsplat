## File tree
fly_gsplat/
└── data/
    ├── images/                 # Full-res grayscale PNGs (1280x800, fly=grayscale, background=black)
    │   ├── P1500CAM1.png
    │   ├── P1500CAM2.png
    │   ├── P1500CAM3.png
    │   └── P1500CAM4.png
    │
    ├── masks/                  # Binary masks (1280x800, fly=white, background=black)
    │   ├── mask_P1500CAM1.png
    │   ├── mask_P1500CAM2.png
    │   ├── mask_P1500CAM3.png
    │   └── mask_P1500CAM4.png
    │
    └── transforms.json         # Camera parameters and file mappings


## core functions
generate_dataset(data_dir: str, sparse_dir: str, target_frame: int):
    creates directories, processes data, and builds transforms.json.
`easyWand.mat` -> `transforms.json`

# Run training
```bash
sudo mount -t drvfs X: /mnt/x

ns-train splatfacto --data ./data --pipeline.model.background-color black --pipeline.datamanager.masks-on-gpu True

ns-train splatfacto --data ./data \
--pipeline.model.background-color random
```
Viser at: http://localhost:7007/

test1: X:\antenna\control\008_08042026\Sparse\Expr_008_mov_015