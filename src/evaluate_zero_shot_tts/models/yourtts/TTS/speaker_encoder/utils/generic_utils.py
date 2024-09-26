import datetime
import glob
import os
import random
import re
from multiprocessing import Manager

import numpy as np
from scipy import signal

from yourtts.TTS.speaker_encoder.models.lstm import LSTMSpeakerEncoder
from yourtts.TTS.speaker_encoder.models.resnet import ResNetSpeakerEncoder
from yourtts.TTS.utils.io import save_fsspec


class Storage(object):
    def __init__(self, maxsize, storage_batchs, num_speakers_in_batch, num_threads=8):
        # use multiprocessing for threading safe
        self.storage = Manager().list()
        self.maxsize = maxsize
        self.num_speakers_in_batch = num_speakers_in_batch
        self.num_threads = num_threads
        self.ignore_last_batch = False

        if storage_batchs >= 3:
            self.ignore_last_batch = True

        # used for fast random sample
        self.safe_storage_size = self.maxsize - self.num_threads
        if self.ignore_last_batch:
            self.safe_storage_size -= self.num_speakers_in_batch

    def __len__(self):
        return len(self.storage)

    def full(self):
        return len(self.storage) >= self.maxsize

    def append(self, item):
        # if storage is full, remove an item
        if self.full():
            self.storage.pop(0)

        self.storage.append(item)

    def get_random_sample(self):
        # safe storage size considering all threads remove one item from storage in same time
        storage_size = len(self.storage) - self.num_threads

        if self.ignore_last_batch:
            storage_size -= self.num_speakers_in_batch

        return self.storage[random.randint(0, storage_size)]

    def get_random_sample_fast(self):
        """Call this method only when storage is full"""
        return self.storage[random.randint(0, self.safe_storage_size)]


class AugmentWAV(object):
    def __init__(self, ap, augmentation_config):

        self.ap = ap
        self.use_additive_noise = False

        if "additive" in augmentation_config.keys():
            self.additive_noise_config = augmentation_config["additive"]
            additive_path = self.additive_noise_config["sounds_path"]
            if additive_path:
                self.use_additive_noise = True
                # get noise types
                self.additive_noise_types = []
                for key in self.additive_noise_config.keys():
                    if isinstance(self.additive_noise_config[key], dict):
                        self.additive_noise_types.append(key)

                additive_files = glob.glob(os.path.join(additive_path, "**/*.wav"), recursive=True)

                self.noise_list = {}

                for wav_file in additive_files:
                    noise_dir = wav_file.replace(additive_path, "").split(os.sep)[0]
                    # ignore not listed directories
                    if noise_dir not in self.additive_noise_types:
                        continue
                    if not noise_dir in self.noise_list:
                        self.noise_list[noise_dir] = []
                    self.noise_list[noise_dir].append(wav_file)

                print(
                    f" | > Using Additive Noise Augmentation: with {len(additive_files)} audios instances from {self.additive_noise_types}"
                )

        self.use_rir = False

        if "rir" in augmentation_config.keys():
            self.rir_config = augmentation_config["rir"]
            if self.rir_config["rir_path"]:
                self.rir_files = glob.glob(os.path.join(self.rir_config["rir_path"], "**/*.wav"), recursive=True)
                self.use_rir = True

            print(f" | > Using RIR Noise Augmentation: with {len(self.rir_files)} audios instances")

        self.create_augmentation_global_list()

    def create_augmentation_global_list(self):
        if self.use_additive_noise:
            self.global_noise_list = self.additive_noise_types
        else:
            self.global_noise_list = []
        if self.use_rir:
            self.global_noise_list.append("RIR_AUG")

    def additive_noise(self, noise_type, audio):

        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)

        noise_list = random.sample(
            self.noise_list[noise_type],
            random.randint(
                self.additive_noise_config[noise_type]["min_num_noises"],
                self.additive_noise_config[noise_type]["max_num_noises"],
            ),
        )

        audio_len = audio.shape[0]
        noises_wav = None
        for noise in noise_list:
            noiseaudio = self.ap.load_wav(noise, sr=self.ap.sample_rate)[:audio_len]

            if noiseaudio.shape[0] < audio_len:
                continue

            noise_snr = random.uniform(
                self.additive_noise_config[noise_type]["min_snr_in_db"],
                self.additive_noise_config[noise_type]["max_num_noises"],
            )
            noise_db = 10 * np.log10(np.mean(noiseaudio ** 2) + 1e-4)
            noise_wav = np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio

            if noises_wav is None:
                noises_wav = noise_wav
            else:
                noises_wav += noise_wav

        # if all possible files is less than audio, choose other files
        if noises_wav is None:
            return self.additive_noise(noise_type, audio)

        return audio + noises_wav

    def reverberate(self, audio):
        audio_len = audio.shape[0]

        rir_file = random.choice(self.rir_files)
        rir = self.ap.load_wav(rir_file, sr=self.ap.sample_rate)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        return signal.convolve(audio, rir, mode=self.rir_config["conv_mode"])[:audio_len]

    def apply_one(self, audio):
        noise_type = random.choice(self.global_noise_list)
        if noise_type == "RIR_AUG":
            return self.reverberate(audio)

        return self.additive_noise(noise_type, audio)


def to_camel(text):
    text = text.capitalize()
    return re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)


def setup_model(c):
    if c.model_params["model_name"].lower() == "lstm":
        model = LSTMSpeakerEncoder(
            c.model_params["input_dim"],
            c.model_params["proj_dim"],
            c.model_params["lstm_dim"],
            c.model_params["num_lstm_layers"],
        )
    elif c.model_params["model_name"].lower() == "resnet":
        model = ResNetSpeakerEncoder(input_dim=c.model_params["input_dim"], proj_dim=c.model_params["proj_dim"],
            log_input=c.model_params.get("log_input", False),
            use_torch_spec=c.model_params.get("use_torch_spec", False),
            audio_config=c.audio
        )
    return model


def save_checkpoint(model, optimizer, criterion, model_loss, out_path, current_step, epoch):
    checkpoint_path = "checkpoint_{}.pth.tar".format(current_step)
    checkpoint_path = os.path.join(out_path, checkpoint_path)
    print(" | | > Checkpoint saving : {}".format(checkpoint_path))

    new_state_dict = model.state_dict()
    state = {
        "model": new_state_dict,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "criterion": criterion.state_dict(),
        "step": current_step,
        "epoch": epoch,
        "loss": model_loss,
        "date": datetime.date.today().strftime("%B %d, %Y"),
    }
    save_fsspec(state, checkpoint_path)


def save_best_model(model, optimizer, criterion, model_loss, best_loss, out_path, current_step):
    if model_loss < best_loss:
        new_state_dict = model.state_dict()
        state = {
            "model": new_state_dict,
            "optimizer": optimizer.state_dict(),
            "criterion": criterion.state_dict(),
            "step": current_step,
            "loss": model_loss,
            "date": datetime.date.today().strftime("%B %d, %Y"),
        }
        best_loss = model_loss
        bestmodel_path = "best_model.pth.tar"
        bestmodel_path = os.path.join(out_path, bestmodel_path)
        print("\n > BEST MODEL ({0:.5f}) : {1:}".format(model_loss, bestmodel_path))
        save_fsspec(state, bestmodel_path)
    return best_loss
