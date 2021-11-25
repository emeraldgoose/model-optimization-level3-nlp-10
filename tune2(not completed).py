"""Tune Model.
- Author: Junghoon Kim, Jongsun Shin
- Contact: placidus36@gmail.com, shinn1897@makinarocks.ai
"""

# 1. valid가 2번 돌아가는 거...? # trainer 파일도 수정 (if val_dataloader is not None)
# 2. worker 개수 8로 수정해주고 (completed) # dataloader 파일 수정 (n_workers = 8)
# 3. 데이터 저장할 수 있도록 수정해야함 (completed)

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import create_dataloader
from src.model import Model
from src.utils.torch_utils import model_info, check_runtime
from src.trainer import TorchTrainer, count_model_params
from typing import Any, Dict, List, Tuple
from optuna.pruners import HyperbandPruner
from subprocess import _args_from_interpreter_flags
import argparse

import os
import yaml
import pickle

EPOCH = 100
OBJ_CALLED = 0
DATA_PATH = "/opt/ml/data"  # type your data path here that contains test, train and val directories
RESULT_MODEL_PATH = "./result_model.pt" # result model will be saved in this path
HYPER_PARAM_PATH = "/opt/ml/code/configs/hyperparams"
RESULT_ROOT = "/opt/ml/code/results"

class Module_maker():

    def __init__(self, trial):

        self.trial = trial
        config_path = os.path.join(HYPER_PARAM_PATH, "module_config.yaml")
        with open(config_path, "r") as f:
            self.config_dic = yaml.load(f, Loader=yaml.FullLoader)
        self.model = []
        self.n_stride = 0
        self._change_dic()

        self.input_channel = self.config_dic["input_channel"]
        self.depth_multiple = self.config_dic["depth_multiple"]
        self.width_multiple = self.config_dic["width_multiple"]

        self.activation = self.config_dic['activation']
        self.num_modules = self.config_dic["num_modules"]
        self.max_num_stride = self.config_dic["max_num_stride"]
        self.upper_stride = self.config_dic['upper_stride']
        self.module_info = {}

    def get_trial(self):
        return self.trial

    def _change_dic(self):
        """
        dictionary의 형태 변환
        """
        new_dic = {}

        for key, item in self.config_dic.items():
            if type(item) == dict: 
                for sub_key, sub_item in item.items():
                    if sub_key not in new_dic.keys():
                        new_dic[sub_key] = {}
                    new_dic[sub_key][key] = sub_item
                    
            else:
                new_dic[key] = item

        self.config_dic = new_dic

    def _make_a_module(self, module_id_int):
        """
        module_id : 해당 모듈이 몇 번째 모듈인지 (1, 2, 3...)
        layer_name : 어떤 레이어를 적용할 것인지 (Conv, DWConv, Inverted...)
        """
        # stride 조절이 필요

        module_id = "m" + str(module_id_int)
        module_infos = self.config_dic[module_id]

        # 공통 설정 정의
        args =[]
        layer = self.trial.suggest_categorical(
            module_id, module_infos["candidates"]
        )
        reps = self.trial.suggest_int(
            module_id + "/repeats", *module_infos["repeats"])

        out_channels = self.trial.suggest_int(
            module_id + "/out_channels", *module_infos["out_channels"])
        
        temp_stride = module_infos["stride"]

        if temp_stride[1] == "upper_stride":
            temp_stride[1] = self.upper_stride

        stride = self.trial.suggest_int(
            module_id + "/stride", *temp_stride)

        activation = self.trial.suggest_categorical(
            module_id + "/activation", self.activation
        )
        if activation == "Pass":
            activation = None

        kernel_size = self.trial.suggest_int(
            module_id + "/kernel_size", *module_infos["kernel_size"])
        
        # 상세 설정 정의
        if layer == "Conv":
            args = [out_channels, kernel_size, stride, None, 1, activation]
        elif layer == "DWConv":
            args = [out_channels, kernel_size, stride, None, activation]
        elif layer == "InvertedResidualv2":
            c = self.trial.suggest_int(
                module_id + "/c", *module_infos["c"]
            )
            t = self.trial.suggest_int(
                module_id + "/t", *module_infos["t"]
            )
            args = [c, t, stride]
        elif layer == "InvertedResidualv3":
            v3_kernel = self.trial.suggest_int(
                module_id + "/v3_kernel", *module_infos["v3_kernel"]
            )
            temp_v3_t = module_infos["v3_t"]

            v3_t = round(self.trial.suggest_float(
                module_id + "/v3_t", low = temp_v3_t[0], high = temp_v3_t[1], step = temp_v3_t[2]
            ),1)
            v3_c = self.trial.suggest_int(
                module_id + "/v3_c", *module_infos["v3_c"]
            )
            se = self.trial.suggest_categorical(
                module_id + "/se", module_infos["se"]
            )
            hs = self.trial.suggest_categorical(
                module_id + "/hs", module_infos["hs"]
            )
            args = [v3_kernel, v3_t, v3_c, se, hs, stride]

        # stride 조정 및 최종 처리
        if not layer == "Pass":
            if stride == 2:
                self.n_stride += 1
                if self.n_stride >= self.max_num_stride:
                    self.upper_stride = 1
        
            self.model.append([reps, layer, args])
        self.module_info[module_id] = {"type":layer, "repeat":reps, "stride":stride}
    
    def make_modules(self):
        for i in range(1, self.num_modules +1):
            self._make_a_module(i)

        last_dim = self.trial.suggest_int("last_dim", *self.config_dic["last_dim"])
        classifier = self.config_dic["classifier"]
        if classifier[0][-1][0] == "last_dim":
            classifier[0][-1][0] = last_dim
        
        for i in range(len(classifier[-1][-1])):
            if classifier[-1][-1][i] == "None":
                classifier[-1][-1][i] = None

        self.model += classifier

        return self.model, self.module_info

def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:

    hyperp_yaml_path = os.path.join(HYPER_PARAM_PATH, "hyperparam.yaml")

    with open (hyperp_yaml_path, "r") as f:
        hyper_dic = yaml.load(f, Loader = yaml.FullLoader)

    """Search hyperparam from user-specified search space."""
    epochs = trial.suggest_int("epochs", *hyper_dic["epochs"])
    img_size = trial.suggest_categorical("img_size", hyper_dic["img_size"])
    n_select = trial.suggest_int("n_select", *hyper_dic["n_select"])
    batch_size = trial.suggest_int("batch_size", *hyper_dic["batch_size"])
    
    return {
        "EPOCHS": epochs,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "BATCH_SIZE": batch_size,
    }

def objective(trial: optuna.trial.Trial, device, study_name) -> Tuple[float, int, float]:
    """Optuna objective.
    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    global OBJ_CALLED
    OBJ_CALLED += 1

    module_maker = Module_maker(trial)
    model_config: Dict[str, Any] = {}

    model_config["backbone"], module_info = module_maker.make_modules()
    trial = module_maker.get_trial()
    hyperparams = search_hyperparam(trial)

    model_config["input_channel"] = module_maker.input_channel

    img_size = hyperparams["IMG_SIZE"]
    model_config["INPUT_SIZE"] = [img_size, img_size]
    model_config["depth_multiple"] = trial.suggest_categorical(
        "depth_multiple", module_maker.depth_multiple
    )
    model_config["width_multiple"] = trial.suggest_categorical(
        "width_multiple", module_maker.width_multiple
    )

    model = Model(model_config, verbose=True)
    model.to(device)
    model.model.to(device)

    # check ./data_configs/data.yaml for config information
    data_config: Dict[str, Any] = {}
    data_config["DATA_PATH"] = DATA_PATH
    data_config["DATASET"] = "TACO"
    data_config["AUG_TRAIN"] = "randaugment_train"
    data_config["AUG_TEST"] = "simple_augment_test"
    data_config["AUG_TRAIN_PARAMS"] = {
        "n_select": hyperparams["n_select"],
    }
    data_config["AUG_TEST_PARAMS"] = None
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["VAL_RATIO"] = 0.8
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]

    print("data==============================")
    print(model_config)
    print("data==============================")

    mean_time = check_runtime(
        model.model,
        [model_config["input_channel"]] + model_config["INPUT_SIZE"],
        device,
    )
    model_info(model, verbose=True)
    train_loader, val_loader, test_loader = create_dataloader(data_config)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        steps_per_epoch=len(train_loader),
        epochs=hyperparams["EPOCHS"],
        pct_start=0.05,
    )

    trainer = TorchTrainer(
        model,
        criterion,
        optimizer,
        scheduler,
        device=device,
        verbose=1,
        model_path=RESULT_MODEL_PATH,
    )
    
    best_acc, best_f1 = trainer.train(train_loader, hyperparams["EPOCHS"]) 
    loss, f1_score, acc_percent = trainer.test(model, test_dataloader=val_loader)
    params_nums = count_model_params(model)

    model_info(model, verbose=True)

    summary = {"data":data_config, "model":model_config}
    
    results_path = os.path.join(RESULT_ROOT, study_name)
    try:
        os.stat(results_path)
    except:
        os.mkdir(results_path)

    with open(os.path.join(results_path,str(OBJ_CALLED)) + ".pkl", "wb") as f:
        pickle.dump(summary, f)

    return f1_score, params_nums, mean_time

def get_best_trial_with_condition(optuna_study: optuna.study.Study) -> Dict[str, Any]:
    """Get best trial that satisfies the minimum condition(e.g. accuracy > 0.8).
    Args:
        study : Optuna study object to get trial.
    Returns:
        best_trial : Best trial that satisfies condition.
    """
    df = optuna_study.trials_dataframe().rename(
        columns={
            "values_0": "acc_percent",
            "values_1": "params_nums",
            "values_2": "mean_time",
        }
    )
    ## minimum condition : accuracy >= threshold
    threshold = 0.7
    minimum_cond = df.acc_percent >= threshold

    if minimum_cond.any():
        df_min_cond = df.loc[minimum_cond]
        ## get the best trial idx with lowest parameter numbers
        best_idx = df_min_cond.loc[
            df_min_cond.params_nums == df_min_cond.params_nums.min()
        ].acc_percent.idxmax()

        best_trial_ = optuna_study.trials[best_idx]
        print("Best trial which satisfies the condition")
        print(df.loc[best_idx])
    else:
        print("No trials satisfies minimum condition")
        best_trial_ = None

    return best_trial_


def tune(gpu_id, study_name, n_trial, storage: str = None):
    print("tune called!!!")
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    sampler = optuna.samplers.MOTPESampler()
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None

    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        study_name= study_name,
        sampler=sampler,
        storage=rdb_storage,
        load_if_exists=True,
    )

    study.optimize(lambda trial: objective(trial, device, study_name), n_trials=n_trial)

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
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    best_trial = get_best_trial_with_condition(study)
    print(best_trial)

    df = study.trials_dataframe().rename(
        columns={
            "values_0": "acc_percent",
            "values_1": "params_nums",
            "values_2": "mean_time",
        }
    )

    df.to_csv(os.path.join(RESULT_ROOT, study_name, study_name) + ".csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--storage", default="", type=str, help="Optuna database storage path.")
    parser.add_argument("--study_name", default = "study_temp", type = str, help = "study name")
    parser.add_argument("--n_trial", default = "1", type = int, help = "how much you try")
    args = parser.parse_args()
    tune(args.gpu, study_name = args.study_name, n_trial = args.n_trial, storage=args.storage if args.storage != "" else None)
