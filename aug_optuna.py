import argparse
import glob
import os, sys
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import yaml
from tqdm import tqdm
from PIL.Image import Image

import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder, VisionDataset

if "./" not in sys.path:
    sys.path.append("./")
    
from src.utils.data import weights_for_balanced_classes
from src.utils.torch_utils import split_dataset_index, check_runtime, model_info
from src.utils.common import get_label_counts, read_yaml
from src.augmentation.policies import simple_augment_test, randaugment_taco, DATASET_NORMALIZE_INFO
from src.augmentation.transforms import FILLCOLOR, SquarePad, Grid, transforms_info
from src.augmentation.methods import Augmentation, SequentialAugmentation

from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer

def load_data_config(path="configs/data/taco.yaml"):
    data_config = read_yaml(cfg=path)
    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])
    return data_config

def load_model_config(path="configs/model/efficientnetv2.yaml"):
    model_config = read_yaml(cfg=path)
    return model_config

def save_yml(data_config, model_config):
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", 'latest'))
    
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
        
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

def load_yml():
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", 'latest'))
    data_config = read_yaml(os.path.join(log_dir, "data.yml"))
    model_config = read_yaml(os.path.join(log_dir, "model.yml"))
    return data_config, model_config

def generate_transform(resize: int = 224, aug_fcns: Tuple = ()) -> transforms.transforms.Compose:
    """Generate train augmentation policy."""
    
    transform_fcns = [
        SquarePad(),
        transforms.Resize((resize, resize))
    ]
    transform_fcns += list(aug_fcns)
    
    transform_fcns.append(transforms.ToTensor())
    transform_fcns.append(transforms.Normalize(mean_v, std_v))
    
    return transforms.Compose(transform_fcns)

def create_dataset(data_path: str,
                   img_size: int = 224,
                   aug_fcns: Tuple = (),
                  ) -> Tuple[VisionDataset, VisionDataset, VisionDataset]:
    tf_train = generate_transform(resize=img_size, aug_fcns=aug_fcns)
    tf_test = generate_transform(resize=img_size)
    
    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "val")
    test_path = os.path.join(data_path, "test")

    train_dataset = ImageFolder(root=train_path, transform=tf_train)
    val_dataset = ImageFolder(root=val_path, transform=tf_test)
    test_dataset = ImageFolder(root=test_path, transform=tf_test)
    
    return train_dataset, val_dataset, test_dataset

def create_loader(
    train_dataset: VisionDataset,
    val_dataset: VisionDataset,
    test_dataset: VisionDataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get dataloader for training and testing."""

    train_loader = DataLoader(
        dataset=train_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
        drop_last=True
    )
    valid_loader = DataLoader(
        dataset=val_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=batch_size,
        num_workers=6
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=batch_size,
        num_workers=6
    )
    return train_loader, valid_loader, test_loader

def tensor_to_img(tensor_img: torch.Tensor) -> np.ndarray:
    return ((tensor_img.squeeze().permute(1, 2, 0).numpy() * std_v + mean_v) * 255).astype(np.uint8)

def visualize_datasets(
    _train_dataset: VisionDataset,
    _val_dataset: VisionDataset,
    _test_dataset: VisionDataset,
    title_prefix: str = ""
) -> None:
    
    fig, ax = plt.subplots(2, 7, figsize=(20, 10))

    for i in range(7):
        idx = np.random.randint(0, len(_train_dataset))
        
        ax[0][i].imshow(tensor_to_img(_train_dataset[idx][0]))
        ax[1][i].imshow(tensor_to_img(_val_dataset[idx][0]))

        ax[0][i].axis('off')
        ax[1][i].axis('off')

    fig.suptitle(f"{title_prefix} Visualization of Augmentation.\n(Each row represents train, validation dataset accordingly)")
    fig.show()
    
    
class CustomRandAugmentation(Augmentation):

    def __init__(
        self,
        chosen_transforms: List[str],
        level: int = 7,
        n_level: int = 15,
    ) -> None:
        """Initialize."""
        super().__init__(n_level)
        self.level = level if isinstance(level, int) and 0 <= level <= n_level else None
        self.chosen_transforms = chosen_transforms

    def __call__(self, img: Image) -> Image:
        """Run augmentations."""
        for transf in self.chosen_transforms:
            level = self.level if self.level else random.randint(0, self.n_level)
            img = self._apply_augment(img, transf, level)
        return img
    

def choose_transforms(n_select: int = 2,
               level: int = 7,
               n_level: int = 15
              ):
    
    operators = [
        "Identity",
        "Contrast",
        "AutoContrast",
        "Rotate",
        "TranslateX",
        "TranslateY",
        "Sharpness",
        "ShearX",
        "ShearY",
        "Color",
        "Brightness",
        "Equalize",
        "Solarize",
        "Posterize",
    ]
    
    chosen_transforms = list(random.sample(operators, k=n_select))
    aug = CustomRandAugmentation(chosen_transforms, level, n_level)
    return aug, chosen_transforms

def objective(trial: optuna.Trial) -> float:
    img_size = 64
    data_path = '../new_data'
    fp16 = True
    epochs = 12
    batch_size = 256

    aug_fcns = []
    
    use_gaussian_blur, use_Contrast, use_AutoContrast, use_Rotate, use_TranslateX, use_TranslateY = None, None, None, None, None, None
    use_Sharpness, use_ShearX, use_ShearY, use_Color, use_Brightness, use_Equalize, use_Solarize, use_Posterize = None, None, None, None, None, None, None, None
    use_cutout, use_color_jitter, use_random_affine, use_random_perspective = None, None, None, None
    use_random_flip, use_random_hflip, use_random_vflip, use_random_resized_crop = None, None, None, None
    
    use_gaussian_blur = trial.suggest_categorical("aug_gaussian_blur", [True, False])
    # use_Contrast = trial.suggest_categorical("aug_Contrast", [True, False])
    # use_AutoContrast = trial.suggest_categorical("aug_AutoContrast", [True, False])
    # use_Rotate = trial.suggest_categorical("aug_Rotate", [True, False])
    # use_TranslateX = trial.suggest_categorical("aug_TranslateX", [True, False])
    # use_TranslateY = trial.suggest_categorical("aug_TranslateY", [True, False])
    use_Sharpness = trial.suggest_categorical("aug_Sharpness", [True, False])
    # use_ShearX = trial.suggest_categorical("aug_ShearX", [True, False])
    # use_ShearY = trial.suggest_categorical("aug_ShearY", [True, False])
    # use_Color = trial.suggest_categorical("aug_Color", [True, False])
    # use_Brightness = trial.suggest_categorical("aug_Brightness", [True, False])
    # use_Equalize = trial.suggest_categorical("aug_Equalize", [True, False])
    # use_Solarize = trial.suggest_categorical("aug_Solarize", [True, False])
    # use_Posterize = trial.suggest_categorical("aug_Posterize", [True, False])
    use_cutout = trial.suggest_categorical("aug_cutout", [True, False])
    use_color_jitter = trial.suggest_categorical("aug_color_jitter", [True, False])
    use_random_affine = trial.suggest_categorical("aug_random_affine", [True, False])
    use_random_perspective = trial.suggest_categorical("aug_random_perspective", [True, False])
    use_random_flip = trial.suggest_categorical("aug_random_flip", [True, False])
    # use_random_hflip = trial.suggest_categorical("aug_random_hflip", [True, False])
    # use_random_vflip = trial.suggest_categorical("aug_random_vflip", [True, False])
    use_random_resized_crop = trial.suggest_categorical("aug_random_resized_crop", [True, False])
    
    if use_gaussian_blur:
        aug_fcns.append(transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)))
    
    if use_Contrast:
        aug_fcns.append(CustomRandAugmentation(['Contrast']))
        
    if use_AutoContrast:
        aug_fcns.append(CustomRandAugmentation(['AutoContrast']))
        
    if use_Rotate:
        aug_fcns.append(CustomRandAugmentation(['Rotate']))
        
    if use_TranslateX:
        aug_fcns.append(CustomRandAugmentation(['TranslateX']))
        
    if use_TranslateY:
        aug_fcns.append(CustomRandAugmentation(['TranslateY']))
        
    if use_Sharpness:
        aug_fcns.append(CustomRandAugmentation(['Sharpness']))
        
    if use_ShearX:
        aug_fcns.append(CustomRandAugmentation(['ShearX']))
        
    if use_ShearY:
        aug_fcns.append(CustomRandAugmentation(['ShearY']))
        
    if use_Color:
        aug_fcns.append(CustomRandAugmentation(['Color']))
        
    if use_Brightness:
        aug_fcns.append(CustomRandAugmentation(['Brightness']))
        
    if use_Equalize:
        aug_fcns.append(CustomRandAugmentation(['Equalize']))
        
    if use_Solarize:
        aug_fcns.append(CustomRandAugmentation(['Solarize']))
        
    if use_Posterize:
        aug_fcns.append(CustomRandAugmentation(['Posterize']))
        
    if use_cutout:
        aug_fcns.append(SequentialAugmentation([("Cutout", 0.8, 9)]))
        
    if use_color_jitter:
        aug_fcns.append(transforms.ColorJitter(brightness=(0.5, 1.5), 
                                                             contrast=(0.5, 1.5), 
                                                             saturation=(0.5, 1.5)))
    if use_random_affine:
        aug_fcns.append(transforms.RandomAffine(degrees=50))
    
    if use_random_perspective:
        aug_fcns.append(transforms.RandomPerspective())
    
    if use_random_flip:
        aug_fcns.append(transforms.RandomHorizontalFlip())
        aug_fcns.append(transforms.RandomVerticalFlip())
        
    if use_random_hflip:
        aug_fcns.append(transforms.RandomHorizontalFlip())
        
    if use_random_vflip:
        aug_fcns.append(transforms.RandomVerticalFlip())
    
    if use_random_resized_crop:
        aug_fcns.append(transforms.RandomResizedCrop(size=img_size,
                                                    ratio=(0.75, 1.0, 1.3333333333333333)))
    
    train_dataset, val_dataset, test_dataset = create_dataset(data_path=data_path,
                                                              img_size=img_size,
                                                              aug_fcns = tuple(aug_fcns))
    
    # visualize_datasets(train_dataset, val_dataset, test_dataset, title_prefix=f"Trial {trial.number:03d} //")
    # plt.draw()
    # plt.show()
    print(aug_fcns)
    
    data_cfg, model_cfg = load_yml()

    model_cfg['backbone'][-1][-1] = [10]
    print()

    model = Model(model_cfg, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
        amsgrad=False
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=8e-4,
        steps_per_epoch=len(train_dataset)//batch_size,
        epochs=epochs,
        pct_start=0.2,
        div_factor=5.0, # initial_lr = max_lr/div_factor
        final_div_factor=1.0, # min_lr = initial_lr/final_div_factor = max_lr/(div_factor*final_div_factor)
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_path),
        device=device
    )
                        
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )
    
    train_loader, val_loader, test_loader = create_loader(train_dataset,
                                                          val_dataset,
                                                          test_dataset,
                                                          batch_size=batch_size)

    exp_dir = "./exp/autoaug"
    os.makedirs(exp_dir, exist_ok=True)
    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        verbose=1,
        model_path=os.path.join(exp_dir, "best.pt")
    )

    best_acc, best_f1 = trainer.train(
        train_dataloader=train_loader,
        n_epoch=epochs,
        val_dataloader=val_loader
    )
                        
    print("TEST DATASET")
    test_loss, test_f1, test_accuracy = trainer.test(model, val_loader if val_loader else test_loader)
    
    return test_f1

def get_best_trial_with_condition(optuna_study: optuna.study.Study) -> Dict[str, Any]:
    df = optuna_study.trials_dataframe().rename(
        columns={
            "values_0": "f1_score",
        }
    )
    ## minimum condition : accuracy >= threshold
    threshold = 0.2
    minimum_cond = df.f1_score >= threshold

    if minimum_cond.any():
        df_min_cond = df.loc[minimum_cond]
        ## get the best trial idx with lowest parameter numbers
        best_idx = df_min_cond.f1_score.idxmax()

        best_trial_ = optuna_study.trials[best_idx]
        print("Best trial which satisfies the condition")
        print(df.loc[best_idx])
    else:
        print("No trials satisfies minimum condition")
        best_trial_ = None

    return best_trial_

def tune(gpu_id, storage: str = None):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    sampler = optuna.samplers.MOTPESampler()
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None
    # study = optuna.create_study(
    #     directions=["maximize", "minimize", "minimize"],
    #     study_name="automl101",
    #     sampler=sampler,
    #     storage=rdb_storage,
    #     load_if_exists=True,
    # )
    
    wandb_kwargs = {"project": "autoaug1"}
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)
    
    study = optuna.create_study(
        direction="maximize", 
        study_name="autoaug", 
        sampler=sampler, 
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial), n_trials=200, callbacks=[wandbc])

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts

#     for tr in best_trials:
#         print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
#         for key, value in tr.params.items():
#             print(f"    {key}:{value}")

#     best_trial = get_best_trial_with_condition(study)
    print(best_trial)
    
    df = optuna_study.trials_dataframe()
    df.to_csv("optuna_aug.csv")
    
if __name__ == "__main__":
    data_normalize_info = DATASET_NORMALIZE_INFO["TACO"]
    mean_v = data_normalize_info["MEAN"]
    std_v = data_normalize_info["STD"]
    
    tune(0)