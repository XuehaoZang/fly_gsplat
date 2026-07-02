[06/21]
Test PyTorch==2.1.2+cu118

1. 120918 baseline
minimum setting as default, no mask
```bash
ns-train splatfacto \
  --data ./data/ctrl_009_002/transforms_no_mask.json \
  --vis viewer+tensorboard
```

2. 152535 
bg color = black, no mask
```bash
ns-train splatfacto \
  --data ./data/ctrl_009_002/transforms_no_mask.json \
  --vis viewer+tensorboard \
  --pipeline.model.background-color black
```
curve looks good
.ply is round fluffy point cloud with no shape

3. 155910 
bg color = black, with mask, eval image
```bash
ns-train splatfacto \
  --data ./data/ctrl_009_002 \
  --vis viewer+tensorboard \
  --pipeline.model.background-color black \
  nerfstudio-data \
  --eval-mode all
```

sphere test
测试集 1: 灰度、黑底白球、无 Mask (最基础的 Baseline)
2026-07-02_151818
ns-train splatfacto \
  --vis tensorboard \
  --max-num-iterations 5000 \
  nerfstudio-data \
  --data ./data/test_01_gray_nomask \
  --center-method none \
  --auto-scale-poses False \
  --orientation-method none \
  --eval-mode all

测试集 2: RGBA 透明背景、白色球、无 Mask

测试集 3: RGB 白底灰球、无 Mask (测试高亮背景的反向梯度影响)
