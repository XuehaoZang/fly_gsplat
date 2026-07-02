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
2026-07-02_154827
ns-train splatfacto \
  --data ./data/test_01_gray_nomask \
  --vis tensorboard \
  --max-num-iterations 5000 \
  --pipeline.model.background-color black \
  nerfstudio-data \
  --eval-mode all

  ns-export gaussian-splat \
  --load-config outputs/test_01_gray_nomask/splatfacto/2026-07-02_154827/config.yml \
  --output-dir outputs/test_01_gray_nomask/splatfacto/2026-07-02_154827

测试集 2: RGBA 透明背景、白色球、无 Mask
2026-07-02_161704
ns-train splatfacto \
  --data ./data/test_02_rgba_nomask \
  --vis tensorboard \
  --max-num-iterations 5000 \
  nerfstudio-data \
  --eval-mode all

测试集 3: RGB 白底灰球、无 Mask (测试高亮背景的反向梯度影响)
2026-07-02_162920
ns-train splatfacto \
  --data ./data/test_03_whitebg_grayfg_nomask \
  --vis tensorboard \
  --max-num-iterations 5000 \
  --pipeline.model.background-color white \
  nerfstudio-data \
  --eval-mode all

| | test_01 黑底 | test_02 RGBA透明 | test_03 白底灰球 |
|---|---|---|---|
| gaussian_count | 1496 | 1077 | 854（最少） |
| radius mean | 0.000319 | 0.000359 | **0.000450（最接近真值）** |
| anisotropy | 1.153 | 1.052 | **1.107** |
| loss 终值 | 最低但平滑 | 抖动最大 | **最低且最平滑** |

test_03 三项指标都最好——高斯数最少（没有多余 floater 去填充白色背景）、径向分布最接近真实球壳（mean 是期望值的 45%，三组里最高）、loss 最干净。

test run ctrl 009 002
1. white bg, no mask, 5k step
2026-07-02_165328
有初步形状，仍有散点
 ns-train splatfacto   --data ./data/ctrl_009_002   --vis tensorboard   --max-num-iterations 5000   --pipeline.model.background-color white   nerfstudio-data   --eval-mode all

12 white bg, no mask, 30k step
 ns-train splatfacto   --data ./data/ctrl_009_002   --vis tensorboard   --max-num-iterations 30000   --pipeline.model.background-color white   nerfstudio-data   --eval-mode all