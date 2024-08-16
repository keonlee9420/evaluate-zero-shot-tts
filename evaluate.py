import argparse
import glob
import os
import random

import numpy as np
import torch
import torchaudio
import tqdm

from src.utils.metric_stats import get_metric
from src.utils.model_factory import get_inference


DATA_LANG_DICT = {
    "librispeech-test-clean": "english",
}


whisper_language_id_dict = {
    "english": "en",
    "korean": "ko",
    "chinese": "zh",
    "japanese": "ja",
    "german": "de",
    "dutch": "nl",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "polish": "pl",
}


hubert_language_id_dict = {
    "english": "en",
}


normalizer_language_id_dict = {
    "english": "en",
    "korean": "ko",
    "chinese": "zh",
    "japanese": "jp",
    "german": "de",
    "dutch": "",
    "french": "fr",
    "spanish": "es",
    "italian": "it",
    "portuguese": "",
    "polish": "",
}


evaluator_dict = {
    "hubert": {
        "model_name": "hubert",
        "device": "cuda:0",
        "org_sample_rate": None,  # will be automatically detected
        "tar_sample_rate": 16000,  # hubert is trained with 16k
        "model_type": "facebook/hubert-large-ls960-ft",
        "lang": "en",
        "name": "wer",
        "amp": False,
        "sdpa_kernel": False,
    },
    "whisper": {
        "model_name": "whisper",
        "device": "cuda:0",
        "org_sample_rate": None,  # will be automatically detected
        "tar_sample_rate": 16000,  # whisper is trained with 16k
        "model_type": "large-v2",
        "lang": None,
        "name": "wer",
        "options": {
            "language": "english",
            "fp16": False,
        },
        "amp": False,
        "sdpa_kernel": False,
    },
    "wavlmuni": {
        "model_name": "wavlmuni",
        "device": "cuda:0",
        "org_sample_rate": None,  # will be automatically detected
        "tar_sample_rate": 16000,  # wavlm is trained with 16k
        "model_type": "src/utils/speaker_verification/ckpt/wavlm_large_finetune.pth",
        "lang": None,
        "name": "sim",
        "amp": False,
        "sdpa_kernel": False,
    },
}


class DotDict(dict):
    """Dictionary subclass that allows dot notation access."""

    def __getattr__(self, key):
        try:
            # Attempt to get the value from dict; if not found, raise AttributeError
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)
            self[key] = value


def read_text_from_path(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def fix_random_seed(SEED=49):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # prepare running
    root_dir = args.root_dir
    evaluator = args.evaluator
    metric = args.metric
    inference_task = args.inference_task
    seed = args.seed
    fix_random_seed(seed)

    # set directory path
    dataset_name, model_name, exp_id = root_dir.split("/")[-3:]
    num_trials = int(exp_id.split("_r")[1].split("_")[0])
    evalset_dir = os.path.join("evalsets", dataset_name, "_".join(exp_id.split("_")[:-1]))
    out_dir = os.path.join(args.out_dir, dataset_name, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # evaluator configs
    cfg = DotDict(evaluator_dict[args.evaluator])

    if args.metric in ["wer", "cer"]:
        assert args.evaluator in ["hubert", "whisper"]
    if "sim" in args.metric:
        assert args.evaluator in ["wavlmuni"]

    # detect dataset and language
    language = DATA_LANG_DICT[dataset_name]
    normalizer_lang = normalizer_language_id_dict[language]
    if args.evaluator == "whisper":
        cfg.options.language = whisper_language_id_dict[language]
        cfg.name = args.metric
        cfg.lang = normalizer_lang
    if args.evaluator == "hubert":
        assert language == "english"
        cfg.name = args.metric
        cfg.lang = normalizer_lang

    # path
    exp_basename = "_".join(
        root_dir.split("/")[-2:]
        + [args.evaluator, inference_task, args.metric]
        + (["trimmed"] if args.truncate else [])
    )
    print("exp_basename:", exp_basename)
    if inference_task == "wav_g":
        assert "exp_base" in exp_id
        wav_g_task = "wav_c" # a fake task used solely for file retrieval

    # (lazy) set up models
    evaluator = None

    # set up metric
    metric = get_metric(name=cfg.name, lang=cfg.lang)
    utt_ids, hyps, refs = [], [], []

    def load_audio(wav_path):
        wavs_hat, sr = torchaudio.load(wav_path)
        wavs_hat = wavs_hat.cuda()
        wav_lens_hat = torch.LongTensor([wavs_hat.shape[-1]]).cuda()
        return wavs_hat, wav_lens_hat, sr

    def get_res(
        evaluator,
        wavs_hat,
        wav_lens_hat,
        wav_path,
        wav_p_path,
        sr,
        wo_dup_prmpt=False,
    ):
        utt_id = [os.path.basename(wav_path).replace(".wav", "")]
        wavs_p, wav_lens_p, sr_p = load_audio(wav_p_path)
        if wo_dup_prmpt:
            wavs_hat = wavs_hat[:, int(wavs_p.shape[-1] * (sr / sr_p)) :]
            wav_lens_hat = torch.LongTensor([wavs_hat.shape[-1]]).cuda()
        if not evaluator:
            evaluator = get_inference(cfg, sample_rate=sr, sample_rate_p=sr_p)

        results = evaluator(
            wavs=wavs_hat,
            wav_lens=wav_lens_hat,
            anchors=wavs_p,
            anchor_lens=wav_lens_p,
            truncate=args.truncate,
        )
        texts_to_gen = [read_text_from_path(wav_path.replace(".wav", ".txt"))]

        utt_ids.extend(utt_id)
        hyps.extend(results)
        refs.extend(texts_to_gen)

        return evaluator

    target_wav_list = sorted(
        glob.glob(os.path.join(root_dir, "*.wav"))
    )
    for i, wav_path in tqdm.tqdm(
        enumerate(target_wav_list),
        desc="iteration : ",
        total=len(target_wav_list),
    ):
        if "_ref" in wav_path:
            continue
        elif (inference_task != "wav_g" and inference_task not in wav_path) or\
            (inference_task == "wav_g" and wav_g_task not in wav_path):
            continue

        if inference_task != "wav_g":
            wavs_hat, wav_lens_hat, sr = load_audio(wav_path)
            wav_p_path = wav_path.replace(".wav", "_ref.wav") if args.metric == 'sim_r' \
                else os.path.join(evalset_dir, os.path.basename(wav_path))
            evaluator = get_res(
                evaluator,
                wavs_hat,
                wav_lens_hat,
                wav_path,
                wav_p_path,
                sr,
                wo_dup_prmpt=args.inference_task == "wav_c"
                and "sim" in args.metric,
            )
        else:
            wav_path = os.path.join(evalset_dir, os.path.basename(wav_path)).split(f"_{wav_g_task}")[0]+"_wav_g.wav"
            wavs_hat, wav_lens_hat, sr = load_audio(wav_path)
            for i in range(num_trials):
                wav_p_path = wav_path.replace("_g.wav", f"_pg_{i}.wav")
                evaluator = get_res(
                    evaluator, wavs_hat, wav_lens_hat, wav_path, wav_p_path, sr
                )

    metric.append(utt_ids, hyps, refs)
    metric.summarize()

    with open(os.path.join(out_dir, f"{exp_basename}.txt"), "wt") as f:
        metric.write_stats(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--root_dir",
        type=str,
        default="samples/librispeech-test-clean/yourtts/exp_base_pl3_r3_20240817044136056885",
    )
    parser.add_argument(
        "-o", "--out_dir", type=str, default="results"
    )
    parser.add_argument(
        "-e",
        "--evaluator",
        type=str,
        required=True,
        choices=["hubert", "whisper", "wavlmuni"],
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        required=True,
        choices=["wer", "cer", "sim_o", "sim_r"],
    )
    parser.add_argument(
        "-t",
        "--inference_task",
        type=str,
        required=True,
        choices=["wav_p", "wav_c", "wav_g"],
    )
    parser.add_argument(
        "-truncate", "--truncate", default=False, action="store_true"
    )
    parser.add_argument("--seed", type=int, default=49)
    args = parser.parse_args()

    main(args)
