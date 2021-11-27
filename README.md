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

* 세부적인 parameter들은 추후에 모델 결정 후 최적화할 예정입니다
