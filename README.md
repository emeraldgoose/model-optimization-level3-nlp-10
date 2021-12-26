# model-optimization-level3-nlp-10

## Baseline
주어진 src 폴더는 베이스라인에 포함되어 있었습니다.

`aug_optuna.py`는 베이스라인으로 제공된 `tune.py`에서 jiho님께서 다시 작성해서 모델을 찾는데 사용했습니다.

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
### 결과
- Public F1 Score 0.7013
- Public Inference Time 55.7934
- Private F1 Score 0.6962
- Private Inference Time 55.7934

## EfficientNetv2
`mbconvv2.py`는 EfficientNetv2의 fused-MBConv 모듈입니다.  

`efficientnetv2.yaml`은 configs/model/에 넣어주시고 `mbconvv2.py`는 src/modules에 넣고 `model.py`는 src에 넣어주시면 됩니다.

그리고 modules에 `__init__.py`에 아래 항목들을 추가해주세요
```
from src.modules.mbconvv2 import MBConvv2, MBConvv2Generator

__all__ = [
    "MBConvv2Generator",
]
```

## pruning.py
모델 최적화 기법 중 Pruning을 시도한 파일입니다.

torch에서 제공하는 `torch.nn.utils.prune` 도구를 사용했습니다.

Pruning을 하게되면 F1기준 0.1정도의 성능 하락이 있지만 다시 학습하는 과정을 반복하면 성능이 오르는 경우가 있습니다.

하지만 그만큼 너무 오래 걸리는 단점이 있습니다.
