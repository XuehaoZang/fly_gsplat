# Roni's code env setup: WSL, Nerfstudio & Gaussian Splatting

## 1. WSL & OS Prep
Install WSL and core Ubuntu packages. Restart your PC after the WSL installation if required.

```bash
wsl.exe --install Ubuntu
sudo apt update && sudo apt upgrade
sudo apt install g++ ninja-build -y
```

## 2. Anaconda Setup
Download and install Anaconda. Skip auto-initialization during setup and do it manually.

```bash
wget https://repo.continuum.io/archive/Anaconda3-2025.06-1-Linux-x86_64.sh
bash Anaconda3-2025.06-1-Linux-x86_64.sh

# Initialize conda for your shell
eval "$(/home/$USER/anaconda3/bin/conda shell.bash hook)"
conda init
```

## 3. Nerfstudio Environment (Python 3.8 / CUDA 11.7)
This sets up `nerfstudio`, including downgrading GCC to v10 and fixing the `libcuda.so` linking error for `tiny-cuda-nn`.

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip

# Install PyTorch & CUDA Toolkit
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit -y

# Downgrade GCC/G++ to v10 (required for CUDA 11.7 + tiny-cuda-nn)
sudo apt install g++-10 gcc-10 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100

# Fix missing libcuda.so linker error
ln -sf $CONDA_PREFIX/lib/stubs/libcuda.so $CONDA_PREFIX/lib/

# Install tiny-cuda-nn & Nerfstudio
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio
```

## 4. Roni's Gaussian Splatting Environment
*Note: Do not fetch submodules from GitHub. Use Roni's modified local submodules.*

```bash
conda create -n gaussian-splatting python=3.9 -y
conda activate gaussian-splatting

# Install PyTorch & CUDA Toolkit (Assuming CUDA 11.8 based on the notes)
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Configure Persistent CUDA_HOME Variables
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export CUDA_HOME=/usr/local/cuda' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Reactivate environment to apply paths
conda deactivate
conda activate gaussian-splatting

# Install Standard Dependencies
conda install -c conda-forge plyfile tqdm -y
pip install opencv-python joblib ninja numpy scipy matplotlib scikit-image plotly open3d 
sudo apt install libglm-dev -y

# Install Local Submodules (Bypassing build isolation to detect Conda's PyTorch)
pip install --no-build-isolation ./diff-gaussian-rasterization
pip install --no-build-isolation ./simple-knn
```

## 5. VS Code Integration
Fix local directory permissions so VS Code can edit/save files within the WSL environment. (Ensure you have the WSL, Python, and Jupyter extensions installed in VS Code).

```bash
# Replace 'abby' with your unix username if different
sudo chown -R $USER /home/$USER/FlyProject/gaussian_reconstruction
```

**Python Script Requirement:** For scripts inside the `gaussian_reconstruction` folder, ensure the base path is appended so python can find the modules:
```python
import sys
sys.path.append('/home/abby/FlyProject/gaussian_reconstruction/')
```