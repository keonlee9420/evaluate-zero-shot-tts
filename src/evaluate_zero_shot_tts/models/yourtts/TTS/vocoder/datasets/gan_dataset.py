import glob
import os
import random
from multiprocessing import Manager

import numpy as np
import torch
from torch.utils.data import Dataset


class GANDataset(Dataset):
    """
    GAN Dataset searchs for all the wav files under root path
    and converts them to acoustic features on the fly and returns
    random segments of (audio, feature) couples.
    """

    def __init__(
        self,
        ap,
        items,
        seq_len,
        hop_len,
        pad_short,
        conv_pad=2,
        return_pairs=False,
        is_training=True,
        return_segments=True,
        use_noise_augment=False,
        use_cache=False,
        verbose=False,
    ):
        super().__init__()
        self.ap = ap
        self.item_list = items
        self.compute_feat = not isinstance(items[0], (tuple, list))
        self.seq_len = seq_len
        self.hop_len = hop_len
        self.pad_short = pad_short
        self.conv_pad = conv_pad
        self.return_pairs = return_pairs
        self.is_training = is_training
        self.return_segments = return_segments
        self.use_cache = use_cache
        self.use_noise_augment = use_noise_augment
        self.verbose = verbose

        assert seq_len % hop_len == 0, " [!] seq_len has to be a multiple of hop_len."
        self.feat_frame_len = seq_len // hop_len + (2 * conv_pad)

        # map G and D instances
        self.G_to_D_mappings = list(range(len(self.item_list)))
        self.shuffle_mapping()

        # cache acoustic features
        if use_cache:
            self.create_feature_cache()

    def create_feature_cache(self):
        self.manager = Manager()
        self.cache = self.manager.list()
        self.cache += [None for _ in range(len(self.item_list))]

    @staticmethod
    def find_wav_files(path):
        return glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        """Return different items for Generator and Discriminator and
        cache acoustic features"""

        # set the seed differently for each worker
        if torch.utils.data.get_worker_info():
            random.seed(torch.utils.data.get_worker_info().seed)

        if self.return_segments:
            item1 = self.load_item(idx)
            if self.return_pairs:
                idx2 = self.G_to_D_mappings[idx]
                item2 = self.load_item(idx2)
                return item1, item2
            return item1
        item1 = self.load_item(idx)
        return item1

    def _pad_short_samples(self, audio, mel=None):
        """Pad samples shorter than the output sequence length"""
        if len(audio) < self.seq_len:
            audio = np.pad(audio, (0, self.seq_len - len(audio)), mode="constant", constant_values=0.0)

        if mel is not None and mel.shape[1] < self.feat_frame_len:
            pad_value = self.ap.melspectrogram(np.zeros([self.ap.win_length]))[:, 0]
            mel = np.pad(
                mel,
                ([0, 0], [0, self.feat_frame_len - mel.shape[1]]),
                mode="constant",
                constant_values=pad_value.mean(),
            )
        return audio, mel

    def shuffle_mapping(self):
        random.shuffle(self.G_to_D_mappings)

    def load_item(self, idx):
        """load (audio, feat) couple"""
        if self.compute_feat:
            # compute features from wav
            wavpath = self.item_list[idx]
            # print(wavpath)

            if self.use_cache and self.cache[idx] is not None:
                audio, mel = self.cache[idx]
            else:
                audio = self.ap.load_wav(wavpath)
                audio, _ = self._pad_short_samples(audio)
                mel = self.ap.melspectrogram(audio)
        else:

            # load precomputed features
            wavpath, feat_path = self.item_list[idx]

            if self.use_cache and self.cache[idx] is not None:
                audio, mel = self.cache[idx]
            else:
                audio = self.ap.load_wav(wavpath)
                mel = np.load(feat_path)
                audio, mel = self._pad_short_samples(audio, mel)

        # correct the audio length wrt padding applied in stft
        audio = np.pad(audio, (0, self.hop_len), mode="edge")
        audio = audio[: mel.shape[-1] * self.hop_len]
        assert (
            mel.shape[-1] * self.hop_len == audio.shape[-1]
        ), f" [!] {mel.shape[-1] * self.hop_len} vs {audio.shape[-1]}"

        audio = torch.from_numpy(audio).float().unsqueeze(0)
        mel = torch.from_numpy(mel).float().squeeze(0)

        if self.return_segments:
            max_mel_start = mel.shape[1] - self.feat_frame_len
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.feat_frame_len
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hop_len
            audio = audio[:, audio_start : audio_start + self.seq_len]

        if self.use_noise_augment and self.is_training and self.return_segments:
            audio = audio + (1 / 32768) * torch.randn_like(audio)
        return (mel, audio)
