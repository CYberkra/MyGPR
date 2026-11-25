# 设置页实测 B-scan 样本

样本取自 Moritz Koch、Jorge Berkhoff、Norbert Blindow、David Farías-Barahona、Pedro Skvarca、Johannes Fürst 和 Matthias Braun（2026）。原记录为 *Ground Penetrating Radar Data, Ice thickness fields and bedrock topography maps of Perito Moreno, Viedma and Upsala Glacier*，版本 4。作者以 CC BY 4.0 发布原始数据，附带的片段按相同许可分发。[Zenodo 数据集](https://doi.org/10.5281/zenodo.21099957) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

源成员为 `06_GPR_unprocessed_sgy.zip` 中的 `PeritoMoreno_1.SGY`，SHA-256 为 `5cac1e3e9fc2067312118131cda57e0862df31d4b8e5e22388016fca7da78f0a`。按成员中的 IBM 浮点解码为 IEEE float32，矩阵为 6,400 个采样点 × 43,228 道。取源矩阵的零基半开区域 `[128:1152, 28500:29268]`。保留边界滤波的邻域后，纵向降采样 4 倍，横向降采样 2 倍。振幅保留符号与原数据单位。坐标以降采样后片段的道号和采样点表示。

用项目依赖的 SciPy 按以下方式再生成样本。其中 `source` 是上述已解码矩阵。

```python
import numpy as np
from scipy.signal import resample_poly

patch = source[80:1200, 28476:29292]
filtered = resample_poly(resample_poly(patch, 1, 4, axis=0), 1, 2, axis=1)
matrix = np.ascontiguousarray(filtered[12:-12, 12:-12], dtype="<f4")
np.savez_compressed("settings_field_bscan.npz", matrix=matrix)
```

输出为 256 × 384 的 float32 振幅矩阵。按行优先顺序计算的矩阵载荷 SHA-256 为 `784ba84d6a729885278eb879c2f00330df58a96191670335eaa7ffd4da4b77e2`。
