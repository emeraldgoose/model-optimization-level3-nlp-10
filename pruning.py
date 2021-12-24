import sys
import os

import torch
from torch.nn.utils import prune
from tqdm import tqdm

from src.dataloader import create_dataloader
from src.trainer import TorchTrainer
from src.loss import CustomCriterion
from src.utils.common import get_label_counts, read_yaml
from src.model import Model

def main(model, data_path, prev_f1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp2/", 'pruned'))
    model_path = os.path.join(log_dir, 'best.pt')

    # evaluate model with test set
    model.load_state_dict(torch.load(model_path))

    model = model.to(device)

    data_config = read_yaml(cfg=data_path)

    parameters_to_prune = (
        (model[0].conv, 'weight'),
        (model[1][0].conv[0][0], 'weight'),
        (model[1][0].conv[1][0], 'weight'),
        (model[1][0].conv[2], 'weight'),
        (model[2].conv, 'weight'),
        (model[3][0].conv[0][0], 'weight'),
        (model[3][0].conv[1], 'weight'),
        (model[4].conv, 'weight'),
        (model[5][0].conv[0], 'weight'),
        (model[5][0].conv[3], 'weight'),
        (model[5][0].conv[5].fc1, 'weight'),
        (model[5][0].conv[5].fc2, 'weight'),
        (model[5][0].conv[7], 'weight'),
        (model[6][0].conv[0][0], 'weight'),
        (model[6][0].conv[1], 'weight'),
        (model[7].conv, 'weight'),
        (model[9].conv, 'weight'),
    )

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )

    train_dl, val_dl, test_dl = create_dataloader(data_config)

    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )

    fp16 = True
    
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=None,
        scheduler=None,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )

    test_loss, test_f1, test_acc = trainer.test(
        model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    )
    
    if test_f1 < prev_f1: sys.exit(0)

    module = model[5][0].conv[5].fc1

    print(list(module.named_parameters()))
    print(list(module.named_buffers()))

    for (module, tar) in parameters_to_prune:
        prune.remove(module=module, name=tar)

    
    print(list(module.named_parameters()))
    print(list(module.named_buffers()))

    torch.save(model.state_dict(), '/opt/ml/code/exp2/pruned/pruned.pt')



if __name__ == "__main__":
    prev_f1 = 0.64

    # prepare datalaoder
    weight = os.path.join('/opt/ml/code/exp/latest/', 'best.pt')
    model_config = os.path.join('/opt/ml/code/exp/latest/', 'model.yml')
    data_path = os.path.join('/opt/ml/code/exp/latest/', 'data.yml')

    # prepare model
    model_instance = Model(model_config, verbose=True)
    model_instance.model.load_state_dict(
        torch.load(weight, map_location=torch.device("cpu"))
    )
    model = model_instance.model

    # module = model[5][0].conv[5].fc1

    # print(list(module.named_parameters()))
    # print(list(module.named_buffers()))
    
    main(model, data_path, prev_f1)
    