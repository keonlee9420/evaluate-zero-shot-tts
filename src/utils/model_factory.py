import functools
from typing import Callable, List, Optional

import sox
import torch
import whisper
from pkg_resources import get_distribution
from torch import Tensor
from torchaudio.transforms import Resample
from transformers import HubertForCTC, Wav2Vec2Processor

from .speaker_verification import init_model as init_unispeech_model


def sequence_mask(lengths: Tensor, max_length: Optional[int] = None) -> Tensor:
    if max_length is None:
        max_length = int(lengths.max().item())
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    return (x.unsqueeze(0) < lengths.unsqueeze(1)).float()


def trim_silence(wav, sample_rate):
    np_arr = wav.cpu().numpy().squeeze()
    # create a transformer
    tfm = sox.Transformer()
    tfm.silence(location=-1, silence_threshold=0.1)
    # transform an in-memory array and return an array
    y_out = tfm.build_array(input_array=np_arr, sample_rate_in=sample_rate)
    duration = len(y_out) / sample_rate
    if duration < 0.5:
        return wav

    return torch.tensor(y_out).unsqueeze(0)


# Ampere TF32 acceleration
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch_version = get_distribution("torch").version


def sdpa_wrapper(func=None, *, enabled=True):
    """
    Wrapper for enabling SDPA kernels in PyTorch
    Kernel is enabled when torch version >= 2.0.0 and enabled is True
    """

    if func is None:
        return functools.partial(sdpa_wrapper, enabled=enabled)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch_version >= "2.0.0" and enabled:
            with torch.backends.cuda.sdp_kernel():
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


def get_inference(cfg, sample_rate: int, sample_rate_p: int) -> Callable:
    """
    Generate inference pipeline

    An inference pipeline consists of setting up models, preprocessing inputs
    and postprocessing outputs. The pipeline is generated based on the parameters
    specified by the cfg object.

    :param cfg: containing the parameters for the inference pipeline
    """

    device = cfg.device
    resampler, freq_ratio = None, 1.0
    resampler_p, freq_ratio_p = None, 1.0
    print("sample_rate:", sample_rate)
    print("sample_rate_p:", sample_rate_p)

    if cfg.model_name == "hubert":
        if sample_rate != cfg.tar_sample_rate:
            resampler = Resample(sample_rate, cfg.tar_sample_rate).to(device)
            freq_ratio = cfg.tar_sample_rate / sample_rate

        processor = Wav2Vec2Processor.from_pretrained(cfg.model_type)
        model = HubertForCTC.from_pretrained(cfg.model_type).to(device).eval()

        @torch.inference_mode()
        @torch.cuda.amp.autocast(enabled=cfg.amp)
        @sdpa_wrapper(enabled=cfg.sdpa_kernel)
        def inference(
            wavs: torch.Tensor,
            wav_lens: torch.Tensor,
            truncate=False,
            *args,
            **kwargs,
        ) -> List[str]:
            wavs = wavs.to(device)
            wavs = resampler(wavs) if resampler is not None else wavs
            if truncate:
                wavs = trim_silence(wavs, cfg.tar_sample_rate).to(device)
                # wav_lens = torch.LongTensor([wavs.shape[-1]]).to(device)
            # wav_lens = (wav_lens * freq_ratio).long()
            # wav_masks = sequence_mask(wav_lens, wavs.shape[1])

            input_values = processor(
                wavs[0], return_tensors="pt", sampling_rate=cfg.tar_sample_rate
            ).input_values.to(
                device
            )  # Batch size 1
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])

            results = [transcription]

            return results

    elif cfg.model_name == "whisper":
        # model prep
        if sample_rate != cfg.tar_sample_rate:
            resampler = Resample(sample_rate, cfg.tar_sample_rate).to(device)
            freq_ratio = cfg.tar_sample_rate / sample_rate

        model = whisper.load_model(cfg.model_type, device=device).eval()
        options = whisper.DecodingOptions(**cfg.options)

        @torch.inference_mode()
        @torch.cuda.amp.autocast(enabled=cfg.amp)
        @sdpa_wrapper(enabled=cfg.sdpa_kernel)
        def inference(
            wavs: torch.Tensor,
            wav_lens: torch.Tensor,
            truncate=False,
            *args,
            **kwargs,
        ) -> List[str]:
            wavs = wavs.to(device)
            wavs = resampler(wavs) if resampler is not None else wavs
            if truncate:
                wavs = trim_silence(wavs, cfg.tar_sample_rate).to(device)
                wav_lens = torch.LongTensor([wavs.shape[-1]]).to(device)
            wav_lens = (wav_lens * freq_ratio).long()
            wav_masks = sequence_mask(wav_lens, wavs.shape[1])

            wavs = whisper.pad_or_trim(wavs * wav_masks)
            specs = whisper.log_mel_spectrogram(wavs)

            results = whisper.decode(model, specs, options=options)
            results = [res.text for res in results]

            return results

    elif cfg.model_name == "wavlmuni":
        # model prep
        if sample_rate != cfg.tar_sample_rate:
            resampler = Resample(sample_rate, cfg.tar_sample_rate).to(device)
            freq_ratio = cfg.tar_sample_rate / sample_rate
        if sample_rate_p != cfg.tar_sample_rate:
            resampler_p = Resample(sample_rate_p, cfg.tar_sample_rate).to(
                device
            )
            freq_ratio_p = cfg.tar_sample_rate / sample_rate_p

        # wavlm
        model = init_unispeech_model(
            model_name="wavlm_large", checkpoint=cfg.model_type
        )
        model = model.to(device).eval()

        # criterion
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)

        @torch.inference_mode()
        @torch.cuda.amp.autocast(enabled=cfg.amp)
        @sdpa_wrapper(enabled=cfg.sdpa_kernel)
        def inference(
            wavs: torch.Tensor,
            wav_lens: torch.Tensor,
            anchors: torch.Tensor,
            anchor_lens: torch.Tensor,
            truncate=False,
            *args,
            **kwargs,
        ) -> List[str]:
            wavs, wav_lens, anchors, anchor_lens = (
                wavs.to(device),
                wav_lens.to(device),
                anchors.to(device),
                anchor_lens.to(device),
            )
            wavs = resampler(wavs) if resampler is not None else wavs
            if truncate:
                wavs = trim_silence(wavs, cfg.tar_sample_rate).to(device)
                wav_lens = torch.LongTensor([wavs.shape[-1]]).to(device)
            wav_lens = (wav_lens * freq_ratio).long()
            # wav_masks = sequence_mask(wav_lens, wavs.shape[1]).bool()
            anchors = (
                resampler_p(anchors) if resampler_p is not None else anchors
            )
            anchor_lens = (anchor_lens * freq_ratio_p).long()
            # anchor_masks = sequence_mask(anchor_lens, anchors.shape[1]).bool()

            # batching wav and anchor together may be faster, leave it for now
            wav_embs = model(wavs)
            anchor_embs = model(anchors)
            results = cosine_sim(wav_embs, anchor_embs).tolist()

            return results

    else:
        raise ValueError("Unsupported inference model")

    return inference
