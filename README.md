# model-optimization-level3-nlp-10

## Augmentation 최적화 결과

/code/src/augmentation/policies.py 에 추가하시고,

/code/configs/data/taco.yaml 의 AUG_TRAIN을 "simple_augment_taco"로 바꿔주시면 됩니다.

```
def simple_augment_taco(
    dataset: str = "TACO", img_size: float = 224
) -> transforms.Compose:
    """Simple data augmentation rule for training TACO."""
    return transforms.Compose(
        [	
	SquarePad(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(degrees=90),
            transforms.RandomPerspective(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                DATASET_NORMALIZE_INFO[dataset]["STD"],
            ),
        ]
    )
```
- Public F1 Score 0.7013
- Public Inference Time 55.7934
- Private F1 Score 0.6962
- Private Inference Time 55.7934

## EfficientNetv2
`efficientnetv2.yaml`은 configs/model/에 넣어주시고 `mbconvv2.py`는 src/modules에 넣고 `model.py`는 src에 넣어주시면 됩니다.

그리고 modules에 `__init__.py`에 아래 항목들을 추가해주세요
```
from src.modules.mbconvv2 import MBConvv2, MBConvv2Generator

__all__ = [
    "MBConvv2Generator",
]
```
