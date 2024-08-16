import argparse
import logging
import os
from pathlib import Path

import torch
import torchaudio

from valle.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
)
from valle.data.collation import get_text_token_collater
from valle.models import add_model_arguments, get_model
import attridict

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)


SAMPLE_RATE = 24000

args = attridict({
    "text_tokens": "src/models/valle_lifeiteng/egs/libritts/data/tokenized/unique_text_tokens.k2symbols",
    "checkpoint": "src/models/valle_lifeiteng/ckpt/epoch-100.pt",
    "top_k": -100,
    "temperature": 1.0,
    # models
    "model_name": "valle",
    "decoder_dim": 1024,
    "nhead": 16,
    "num_decoder_layers": 12,
    "norm_first": True,
    "add_prenet": False,
})


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)


class ValleLifeitengModel:

    def __init__(self):
        self.text_tokenizer = TextTokenizer()
        self.text_collater = get_text_token_collater(args.text_tokens)
        self.audio_tokenizer = AudioTokenizer()

        self.model = get_model(args)
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            missing_keys, unexpected_keys = self.model.load_state_dict(
                checkpoint["model"], strict=True
            )
            assert not missing_keys

        self.model.to(device)
        self.model.eval()

    def encodec_recon(self, audio_path):
        encoded_frames = tokenize_audio(self.audio_tokenizer, audio_path)
        samples = self.audio_tokenizer.decode(encoded_frames)
        return samples[0].cpu()

    def inference(self, text, audio_file, text_prompt=None):
        audio_prompt_all = []

        encoded_frames = tokenize_audio(self.audio_tokenizer, audio_file)
        audio_prompt_all.append(encoded_frames[0][0])
        audio_prompt_all = torch.concat(audio_prompt_all, dim=-1).transpose(2, 1)
        audio_prompt_all = audio_prompt_all.to(device)

        text_tokens, text_tokens_lens = self.text_collater(
            [
                tokenize_text(
                    self.text_tokenizer, text=f"{text}".strip() if text_prompt is None \
                        else f"{text_prompt} {text}".strip()
                )
            ]
        )

        # synthesis
        encoded_frames = self.model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            audio_prompt_all,
            top_k=args.top_k,
            temperature=args.temperature,
            continual=text_prompt is None,
        )

        assert len(audio_prompt_all) == 1
        samples = self.audio_tokenizer.decode(
            [(encoded_frames.transpose(2, 1), None)]
        )

        # recon prompt
        audio_recon = self.encodec_recon(audio_file)

        return samples[0].cpu(), audio_recon.cpu(), SAMPLE_RATE
