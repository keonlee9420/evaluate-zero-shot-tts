# -*- coding: utf-8 -*-
import datetime
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict

import fsspec
import torch


def to_cuda(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.contiguous()
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
    return x


def get_cuda():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return use_cuda, device


def get_git_branch():
    try:
        out = subprocess.check_output(["git", "branch"]).decode("utf8")
        current = next(line for line in out.split("\n") if line.startswith("*"))
        current.replace("* ", "")
    except subprocess.CalledProcessError:
        current = "inside_docker"
    except FileNotFoundError:
        current = "unknown"
    return current


def get_commit_hash():
    """https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script"""
    # try:
    #     subprocess.check_output(['git', 'diff-index', '--quiet',
    #                              'HEAD'])  # Verify client is clean
    # except:
    #     raise RuntimeError(
    #         " !! Commit before training to get the commit hash.")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    # Not copying .git folder into docker container
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "0000000"
    return commit


def get_experiment_folder_path(root_path, model_name):
    """Get an experiment folder path with the current date and time"""
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    commit_hash = get_commit_hash()
    output_folder = os.path.join(root_path, model_name + "-" + date_str + "-" + commit_hash)
    return output_folder


def remove_experiment_folder(experiment_path):
    """Check folder if there is a checkpoint, otherwise remove the folder"""
    fs = fsspec.get_mapper(experiment_path).fs
    checkpoint_files = fs.glob(experiment_path + "/*.pth.tar")
    if not checkpoint_files:
        if fs.exists(experiment_path):
            fs.rm(experiment_path, recursive=True)
            print(" ! Run is removed from {}".format(experiment_path))
    else:
        print(" ! Run is kept in {}".format(experiment_path))


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_camel(text):
    text = text.capitalize()
    text = re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)
    text = text.replace("Tts", "TTS")
    return text


def find_module(module_path: str, module_name: str) -> object:
    module_name = module_name.lower()
    module = importlib.import_module(module_path + "." + module_name)
    class_name = to_camel(module_name)
    return getattr(module, class_name)


def get_user_data_dir(appname):
    if sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")
    return ans.joinpath(appname)


def set_init_dict(model_dict, checkpoint_state, c):
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k, v in checkpoint_state.items():
        if k not in model_dict:
            print(" | > Layer missing in the model definition: {}".format(k))
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in checkpoint_state.items() if k in model_dict}
    # 2. filter out different size layers
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if v.numel() == model_dict[k].numel()}
    # 3. skip reinit layers
    if c.has("reinit_layers") and c.reinit_layers is not None:
        for reinit_layer_name in c.reinit_layers:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if reinit_layer_name not in k}
    # 4. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    print(" | > {} / {} layers are restored.".format(len(pretrained_dict), len(model_dict)))
    return model_dict


def format_aux_input(def_args: Dict, kwargs: Dict) -> Dict:
    """Format kwargs to hande auxilary inputs to models.

    Args:
        def_args (Dict): A dictionary of argument names and their default values if not defined in `kwargs`.
        kwargs (Dict): A `dict` or `kwargs` that includes auxilary inputs to the model.

    Returns:
        Dict: arguments with formatted auxilary inputs.
    """
    for name in def_args:
        if name not in kwargs:
            kwargs[def_args[name]] = None
    return kwargs


class KeepAverage:
    def __init__(self):
        self.avg_values = {}
        self.iters = {}

    def __getitem__(self, key):
        return self.avg_values[key]

    def items(self):
        return self.avg_values.items()

    def add_value(self, name, init_val=0, init_iter=0):
        self.avg_values[name] = init_val
        self.iters[name] = init_iter

    def update_value(self, name, value, weighted_avg=False):
        if name not in self.avg_values:
            # add value if not exist before
            self.add_value(name, init_val=value)
        else:
            # else update existing value
            if weighted_avg:
                self.avg_values[name] = 0.99 * self.avg_values[name] + 0.01 * value
                self.iters[name] += 1
            else:
                self.avg_values[name] = self.avg_values[name] * self.iters[name] + value
                self.iters[name] += 1
                self.avg_values[name] /= self.iters[name]

    def add_values(self, name_dict):
        for key, value in name_dict.items():
            self.add_value(key, init_val=value)

    def update_values(self, value_dict):
        for key, value in value_dict.items():
            self.update_value(key, value)
