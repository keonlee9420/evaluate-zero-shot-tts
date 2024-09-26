import argparse
import glob
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


task_dict = {
    "cont": {
        "task_name": "wav_c",
        "task_dir": {
            "librispeech-test-clean": "evalsets/librispeech-test-clean/exp_base_pl3_r3",
        },
    },
    "cross": {
        "task_name": "wav_p",
        "task_dir": {
            "librispeech-test-clean": "evalsets/librispeech-test-clean/exp_aligned_pl3_r3",
        },
    },
}


def read_text_from_path(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_txt(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)


def fix_random_seed(SEED=49):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_unique_key():
    now = datetime.now()
    unique_key = now.strftime("%Y%m%d%H%M%S%f")
    return unique_key


class InferDataset(Dataset):
    def __init__(self, task_key, dataset_key="en"):
        super().__init__()
        assert task_key in ["cont", "cross"]
        task_name, task_dir = (
            task_dict[task_key]["task_name"],
            task_dict[task_key]["task_dir"][dataset_key],
        )
        assert task_name in ["wav_c", "wav_p"]

        fps = glob.glob(os.path.join(task_dir, f"*_{task_name}_*.wav"))
        fids = set([os.path.basename(x).split("_wav_")[0] for x in fps])
        assert len(fps) % len(fids) == 0
        num_trial = len(fps) // len(fids)

        # Preload dataset
        self.prompt_wav_path = []
        self.prompt_text = []
        self.target_text = []
        for fid in tqdm(fids, desc="preload dataset ..."):
            for i in range(num_trial):
                self.prompt_wav_path.append(
                    os.path.join(task_dir, f"{fid}_{task_name}_{i}.wav")
                )
                self.prompt_text.append(
                    read_text_from_path(
                        os.path.join(task_dir, f"{fid}_{task_name}_{i}.txt")
                    ).lower()
                    if task_name == "wav_p"
                    else ""
                )  # prompt_text is included in text in continuation task
                self.target_text.append(
                    read_text_from_path(
                        os.path.join(task_dir, f"{fid}_wav_g.txt")
                    ).lower()
                )
        assert len(self.prompt_text) == len(self.target_text)

    def __len__(self):
        return len(self.prompt_wav_path)

    def __getitem__(self, idx):
        # prompt_wav, sr = torchaudio.load(self.prompt_wav_path[idx])
        meta_data = torchaudio.info(self.prompt_wav_path[idx])
        wav_len = meta_data.num_frames
        sr = meta_data.sample_rate

        sample = {
            "text": self.target_text[idx],
            "prompt_text": self.prompt_text[idx],
            # "prompt_wav": prompt_wav,
            "prompt_wav_len": wav_len,
            "prompt_wav_sr": sr,
            "prompt_wav_path": self.prompt_wav_path[idx],
        }
        return sample

    def to_device(self, batch, device):
        # batch["prompt_wav"] = batch["prompt_wav"][:, :, : batch["prompt_wav_len"].max()].to(device)
        batch["prompt_wav_len"] = batch["prompt_wav_len"].to(device)
        return batch


class InferModel:

    def __init__(self, model_name):
        self.model_name = model_name
        if model_name == "yourtts":
            from evaluate_zero_shot_tts.models.yourtts_inference import YourTTSModel
            self.model = YourTTSModel()
        elif model_name == "valle_lifeiteng":
            from evaluate_zero_shot_tts.models.valle_lifeiteng_inference import ValleLifeitengModel
            self.model = ValleLifeitengModel()
        else:
            raise NotImplementedError()

    def __call__(self, text, prompt_wav_path, prompt_text=None):
        wav, wav_p, sr = self.model.inference(text, prompt_wav_path, prompt_text)
        assert isinstance(wav, torch.Tensor) and len(wav.shape) == 2
        if wav_p is not None:
            assert isinstance(wav_p, torch.Tensor) and len(wav_p.shape) == 2
        return wav, wav_p, sr


def main(args):
    # prepare running
    run_id = generate_unique_key()
    fix_random_seed(args.seed)

    # set output directory
    assert args.dataset_key in task_dict[args.task_key]["task_dir"]
    output_dir = f"{args.output_dir}/{args.dataset_key}/{args.model_name}/{os.path.basename(task_dict[args.task_key]['task_dir'][args.dataset_key])}_{run_id}/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"output_dir: {output_dir}\n")

    # data loader
    infer_dataset = InferDataset(args.task_key, args.dataset_key)

    infer_dataloader = DataLoader(
        infer_dataset,
        batch_size=1, # currently, batch inference is not supported.
        shuffle=False,
        num_workers=args.num_workers,
    )

    # TODO: model loader
    model = InferModel(args.model_name)

    # prompted generation
    for batch in tqdm(infer_dataloader, desc="sampling ..."):
        batch = infer_dataset.to_device(batch, "cuda:0")

        sampled_wav, prompt_wav_recon, sr = model(batch['text'][0], batch['prompt_wav_path'][0], None if args.task_key == "cont" else batch['prompt_text'][0])

        # save wav
        sampled_wav_path = os.path.basename(batch["prompt_wav_path"][0])
        torchaudio.save(
            os.path.join(output_dir, sampled_wav_path),
            sampled_wav.detach().cpu(),
            sample_rate=sr,
        )

        # save txt
        sample_txt_path = sampled_wav_path.replace(".wav", ".txt")
        save_txt(
            os.path.join(output_dir, sample_txt_path),
            batch["text"][0],
        )

        # save (reconstructed) prmopt wav & text
        if prompt_wav_recon is not None:
            ref_wav_path = os.path.basename(batch["prompt_wav_path"][0]).replace(
                ".wav", "_ref.wav"
            )
            torchaudio.save(
                os.path.join(output_dir, ref_wav_path),
                prompt_wav_recon.detach().cpu(),
                sample_rate=sr,
            )
            ref_txt_path = ref_wav_path.replace(".wav", ".txt")
            save_txt(
                os.path.join(output_dir, ref_txt_path),
                batch["prompt_text"][0],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference arguments")
    parser.add_argument(
        "--task_key",
        type=str,
        default="cont",
        choices=["cont", "cross"],
    )
    parser.add_argument(
        "--dataset_key",
        type=str,
        default="librispeech-test-clean",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="yourtts",
        choices=["yourtts", "valle_lifeiteng"],
    )
    parser.add_argument("--output_dir", type=str, default="samples")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
