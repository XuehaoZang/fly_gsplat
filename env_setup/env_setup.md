This tutorial provides a streamlined guide to setting up gsplat/Nerfstudio environments on WSL2 (Ubuntu 24.04).

---

## System & Anaconda Setup
Initial preparation for WSL and the base Python manager.

```bash
wsl.exe --install Ubuntu
sudo apt update && sudo apt upgrade
sudo apt install g++ ninja-build -y
```

```bash
wget https://repo.continuum.io/archive/Anaconda3-2025.06-1-Linux-x86_64.sh
bash Anaconda3-2025.06-1-Linux-x86_64.sh

# Initialize conda for your bash shell
eval "$(/home/$USER/anaconda3/bin/conda shell.bash hook)"
conda init
```

---

## Primary Env: `fly_gsplat` (Python 3.10)
**Best for:** 3D Gaussian Splatting (3DGS) and `gsplat`. 
**Specs:** Python 3.10 | CUDA 11.8 | PyTorch 2.0.1.

```bash
conda create -n fly_gsplat python=3.10 -y
conda activate fly_gsplat
```

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

```bash
pip install ninja numpy jaxtyping rich
pip install gsplat==1.4.0 --index-url https://docs.gsplat.studio/whl/pt20cu118
```

```bash
# Downgrade fpsample for Python compatibility
pip install fpsample==0.2.0

python -m pip install --upgrade setuptools==69.5.1

# Install Nerfstudio
pip install nerfstudio

# Verify Installation
python -c "import nerfstudio; print('nerfstudio import successful! \npath:', nerfstudio.__file__)"
```

### New
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
---

## Environment Verification
Run this script in either environment to generate a status report.

```bash
echo "========================================="
echo "      Nerfstudio & 3DGS Env Report       "
echo "========================================="

echo -e "\n[1] Hardware & OS"
lsb_release -d
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo -e "\n[2] Compilers & CUDA (System vs Conda)"
echo "GCC Path:  $(which gcc) | $(gcc --version | head -n 1)"
echo "G++ Path:  $(which g++) | $(g++ --version | head -n 1)"
echo "NVCC Path: $(which nvcc)"
nvcc --version | grep release

echo -e "\n[3] Core Python Dependencies"
python -c "
import sys
import torch

print(f'Python Version:       {sys.version.split()[0]}')
print(f'PyTorch Version:      {torch.__version__}')
print(f'PyTorch CUDA Build:   {torch.version.cuda}')
print(f'GPU Available:        {torch.cuda.is_available()}')
print(f'CUDA Device Count:    {torch.cuda.device_count()}')

# Check tinycudann
try:
    import tinycudann
    print('tiny-cuda-nn:          Loaded Successfully (v1.6 Source Build)')
except ImportError:
    print('tiny-cuda-nn:          [ERROR] Missing')

# Check Nerfstudio
try:
    import nerfstudio
    print(f'nerfstudio:           Loaded from {nerfstudio.__file__}')
except ImportError:
    print('nerfstudio:           [ERROR] Missing')

# Check Ninja
try:
    import ninja
    print(f'ninja (Builder):      {ninja.__version__}')
except ImportError:
    print('ninja (Builder):      [ERROR] Missing')
"
echo "========================================="
```

---

## Test Run
Verify 3DGS functionality using the `lego` dataset.

```bash
conda activate fly_gsplat  # Or activate nerfstudio
python -c "import gsplat; from gsplat import csrc; print('GSplat Engine Ready.')"

cd ~/fly_project/fly_gsplat/test_env

# static scenes
# Download dataset
ns-download-data nerfstudio --capture-name=campanile

# Train the first model!
ns-train splatfacto --data data/campanile

# moving scenes
# Download dnerf dataset
ns-download-data dnerf

ns-train splatfacto blender-data --data data/dnerf/lego
# splatfacto cannot train moving scenes
```

```bash
"campanile": grab_file_id("https://drive.google.com/file/d/13aOfGJRRH05pOOk9ikYGTwqFc2L1xskU/view?usp=sharing"),
"library": grab_file_id("https://drive.google.com/file/d/1Hjbh_-BuaWETQExn2x2gGD74UwrFugHx/view?usp=sharing"),
"poster": grab_file_id("https://drive.google.com/file/d/1FceQ5DX7bbTbHeL26t0x6ku56cwsRs6t/view?usp=sharing"),
"redwoods2": grab_file_id("https://drive.google.com/file/d/1rg-4NoXT8p6vkmbWxMOY6PSG4j3rfcJ8/view?usp=sharing"),
"vegetation": grab_file_id("https://drive.google.com/file/d/1wBhLQ2odycrtU39y2akVurXEAts9SsVI3/view?usp=sharing"),
"storefront": grab_file_id("https://drive.google.com/file/d/16b792AguPZWDA_YC4igKCwXJqW0Tb21o/view?usp=sharing")
```




----------
----------

## Alternative Env: `nerfstudio` (Python 3.8)
**Best for:** Legacy NeRF models requiring `tiny-cuda-nn`. 
**Specs:** Python 3.8 | CUDA 11.7 | PyTorch 2.0.1.

```bash
# Enter WSL and create the environment
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip

# Clean existing packages if necessary
pip uninstall torch torchvision functorch tinycudann -y

# Install PyTorch & CUDA Toolkit
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit -y
```

### Build `tiny-cuda-nn` from Source
```bash
sudo apt install g++-10 gcc-10 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

# Fix missing libcuda.so linker error
ln -sf $CONDA_PREFIX/lib/stubs/libcuda.so $CONDA_PREFIX/lib/
```

```bash
pip install ninja

# Clone and install from source
cd ~
rm -rf tiny-cuda-nn
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git
cd tiny-cuda-nn
git checkout 466aa1c
git submodule update --init --recursive
cd bindings/torch
python setup.py install
```

```bash
# Downgrade fpsample for Python compatibility
pip install fpsample==0.2.0

python -m pip install --upgrade setuptools==69.5.1

# Install Nerfstudio
pip install nerfstudio

# Verify Installation
python -c "import nerfstudio; print('nerfstudio import successful! \npath:', nerfstudio.__file__)"
```
---
