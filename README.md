# model-optimization-level3-nlp-10

`efficientnetv2.yaml`은 configs/model/에 넣어주시고 `mbconvv2.py`는 src/modules에 넣고 `model.py`는 src에 넣어주시면 됩니다.

그리고 modules에 `__init__.py`에 아래 항목들을 추가해주세요
```
from src.modules.mbconvv2 import MBConvv2, MBConvv2Generator

__all__ = [
    "MBConvv2Generator",
]
```
