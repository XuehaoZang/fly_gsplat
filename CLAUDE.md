# fly_gsplat
用 Nerfstudio (splatfacto) 从多相机实验室录像重建果蝇 3D 点云。数据来自 4 相机 + EasyWand MATLAB 标定。

## 文件读取规则
- 除非用户明确指定文件名或路径，否则不要主动读取任何文件
- 不要探索或索引 data/、outputs/、results/、__pycache__/ 等目录
- 不要读取 .gitignore 中列出的任何文件或目录
- 回答问题时优先用已知项目结构推断，不要主动 grep 或 glob

## 项目结构
- utils/camera.py — CameraConfig 核心数据结构（OpenCV↔OpenGL 转换）
- utils/calib.py — proj, backproj, triangulate
- utils/dataset.py — generate_frame_dict（生成 transforms.json frame）
- utils/image.py — binarize_mask, dilate_mask, crop_image
- utils/viz.py — Viser 可视化工具
- generate_dataset.py — 生成 images/ + transforms.json
- generate_hull.py — 生成 visual hull init_points.ply
- validate_calib.py — 重投影误差验证
- validate_dataset.py — Viser 相机几何验证
- debug_splat_ply.py — 对比 hull 和训练后 splat 的坐标系和形态

## 技术栈
- Python 3.10 + PyTorch 2.1.2 + CUDA 11.8
- gsplat 1.4.0 + nerfstudio 1.1.5
- WSL2 + conda (fly_gsplat) + VS Code + MATLAB (EasyWand)

## 工作方式
- 你是顾问，我做所有实现，不要直接生成完整文件
- 代码改动只给 before/after 片段，不给整文件
- 中文回复，一次一个小步骤，等我反馈再继续
- 不要主动运行命令，除非我明确说"你来运行"

## 相机坐标约定
EasyWand DLT → OpenCV (R_w2c, X0) → OpenGL (c2w, Y up, Z backward) → transforms.json

## 实验记录
所有实验结果记录在对话里，不写入代码注释