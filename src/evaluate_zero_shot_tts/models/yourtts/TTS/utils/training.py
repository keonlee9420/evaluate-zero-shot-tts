import numpy as np
import torch


def check_update(model, grad_clip, ignore_stopnet=False, amp_opt_params=None):
    r"""Check model gradient against unexpected jumps and failures"""
    skip_flag = False
    if ignore_stopnet:
        if not amp_opt_params:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [param for name, param in model.named_parameters() if "stopnet" not in name], grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(amp_opt_params, grad_clip)
    else:
        if not amp_opt_params:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(amp_opt_params, grad_clip)

    # compatibility with different torch versions
    if isinstance(grad_norm, float):
        if np.isinf(grad_norm):
            print(" | > Gradient is INF !!")
            skip_flag = True
    else:
        if torch.isinf(grad_norm):
            print(" | > Gradient is INF !!")
            skip_flag = True
    return grad_norm, skip_flag


# pylint: disable=protected-access
class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=0.1, last_epoch=-1):
        self.warmup_steps = float(warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        return [
            base_lr * self.warmup_steps ** 0.5 * min(step * self.warmup_steps ** -1.5, step ** -0.5)
            for base_lr in self.base_lrs
        ]


def gradual_training_scheduler(global_step, config):
    """Setup the gradual training schedule wrt number
    of active GPUs"""
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        num_gpus = 1
    new_values = None
    # we set the scheduling wrt num_gpus
    for values in config.gradual_training:
        if global_step * num_gpus >= values[0]:
            new_values = values
    return new_values[1], new_values[2]


def lr_decay(init_lr, global_step, warmup_steps):
    r"""from https://github.com/r9y9/tacotron_pytorch/blob/master/train.py
    It is only being used by the Speaker Encoder trainer."""
    warmup_steps = float(warmup_steps)
    step = global_step + 1.0
    lr = init_lr * warmup_steps ** 0.5 * np.minimum(step * warmup_steps ** -1.5, step ** -0.5)
    return lr


# pylint: disable=dangerous-default-value
def set_weight_decay(model, weight_decay, skip_list={"decoder.attention.v", "rnn", "lstm", "gru", "embedding"}):
    """
    Skip biases, BatchNorm parameters, rnns.
    and attention projection layer v
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or any((skip_name in name for skip_name in skip_list)):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]
