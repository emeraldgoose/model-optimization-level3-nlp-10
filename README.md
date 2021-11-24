# model-optimization-level3-nlp-10

`efficientnetv2.yaml`은 configs/model/에 넣어주시고 `mbconvv2.py`는 src/modules에 넣고 `model.py`는 src에 넣어주시면 됩니다.

그리고 modules에 `__init__.py`에 아래 항목들을 추가해주세요
```
from src.modules.mbconvv2 import MBConvv2, MBConvv2Generator

__all__ = [
    "MBConvv2Generator",
]
```

## Model Architecture

idx |   n |     params |          module |            arguments |   in_channel |   out_channel
----------------------------------------------------------------------------------------------
  0 |   1 |        696 |            Conv |           [24, 3, 2] |            3           24
  1 |   2 |     11,712 |        MBConvv2 |        [1, 24, 1, 0] |           24           24
  2 |   4 |    303,552 |        MBConvv2 |        [4, 48, 2, 0] |           24           48
  3 |   4 |    589,184 |        MBConvv2 |        [4, 64, 2, 0] |           48           64
  4 |   6 |    917,680 |        MBConvv2 |       [4, 128, 2, 1] |           64          128
  5 |   9 |  3,463,840 |        MBConvv2 |       [6, 160, 1, 1] |          128          160
  6 |  15 | 14,561,832 |        MBConvv2 |       [6, 256, 2, 1] |          160          256
  7 |   1 |    330,240 |            Conv |         [1280, 1, 1] |          256         1280
  8 |   1 |          0 |   GlobalAvgPool |                   [] |         1280         1280
  9 |   1 |          0 |         Flatten |                   [] |         1280         1280
 10 |   1 |     12,810 |          Linear |                 [10] |         1280           10
