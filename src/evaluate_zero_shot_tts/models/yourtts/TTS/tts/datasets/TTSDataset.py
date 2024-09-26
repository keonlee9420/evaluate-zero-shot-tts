import collections
import os
import random
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from yourtts.TTS.tts.utils.data import prepare_data, prepare_stop_target, prepare_tensor
from yourtts.TTS.tts.utils.text import pad_with_eos_bos, phoneme_to_sequence, text_to_sequence
from yourtts.TTS.utils.audio import AudioProcessor


class TTSDataset(Dataset):
    def __init__(
        self,
        outputs_per_step: int,
        text_cleaner: list,
        compute_linear_spec: bool,
        ap: AudioProcessor,
        meta_data: List[List],
        characters: Dict = None,
        custom_symbols: List = None,
        add_blank: bool = False,
        return_wav: bool = False,
        batch_group_size: int = 0,
        min_seq_len: int = 0,
        max_seq_len: int = float("inf"),
        use_phonemes: bool = False,
        phoneme_cache_path: str = None,
        phoneme_language: str = "en-us",
        enable_eos_bos: bool = False,
        speaker_id_mapping: Dict = None,
        d_vector_mapping: Dict = None,
        language_id_mapping: Dict = None,
        use_noise_augment: bool = False,
        verbose: bool = False,
    ):
        """Generic 📂 data loader for `tts` models. It is configurable for different outputs and needs.

        If you need something different, you can either override or create a new class as the dataset is
        initialized by the model.

        Args:
            outputs_per_step (int): Number of time frames predicted per step.

            text_cleaner (list): List of text cleaners to clean the input text before converting to sequence IDs.

            compute_linear_spec (bool): compute linear spectrogram if True.

            ap (TTS.tts.utils.AudioProcessor): Audio processor object.

            meta_data (list): List of dataset instances.

            characters (dict): `dict` of custom text characters used for converting texts to sequences.

            custom_symbols (list): List of custom symbols used for converting texts to sequences. Models using its own
                set of symbols need to pass it here. Defaults to `None`.

            add_blank (bool): Add a special `blank` character after every other character. It helps some
                models achieve better results. Defaults to false.

            return_wav (bool): Return the waveform of the sample. Defaults to False.

            batch_group_size (int): Range of batch randomization after sorting
                sequences by length. It shuffles each batch with bucketing to gather similar lenght sequences in a
                batch. Set 0 to disable. Defaults to 0.

            min_seq_len (int): Minimum input sequence length to be processed
                by the loader. Filter out input sequences that are shorter than this. Some models have a
                minimum input length due to its architecture. Defaults to 0.

            max_seq_len (int): Maximum input sequence length. Filter out input sequences that are longer than this.
                It helps for controlling the VRAM usage against long input sequences. Especially models with
                RNN layers are sensitive to input length. Defaults to `Inf`.

            use_phonemes (bool): If true, input text converted to phonemes. Defaults to false.

            phoneme_cache_path (str): Path to cache phoneme features. It writes computed phonemes to files to use in
                the coming iterations. Defaults to None.

            phoneme_language (str): One the languages from supported by the phonemizer interface. Defaults to `en-us`.

            enable_eos_bos (bool): Enable the `end of sentence` and the `beginning of sentences characters`. Defaults
                to False.

            speaker_id_mapping (dict): Mapping of speaker names to IDs used to compute embedding vectors by the
                embedding layer. Defaults to None.

            d_vector_mapping (dict): Mapping of wav files to computed d-vectors. Defaults to None.

            use_noise_augment (bool): Enable adding random noise to wav for augmentation. Defaults to False.

            verbose (bool): Print diagnostic information. Defaults to false.
        """
        super().__init__()
        self.batch_group_size = batch_group_size
        self.items = meta_data
        self.outputs_per_step = outputs_per_step
        self.sample_rate = ap.sample_rate
        self.cleaners = text_cleaner
        self.compute_linear_spec = compute_linear_spec
        self.return_wav = return_wav
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.ap = ap
        self.characters = characters
        self.custom_symbols = custom_symbols
        self.add_blank = add_blank
        self.use_phonemes = use_phonemes
        self.phoneme_cache_path = phoneme_cache_path
        self.phoneme_language = phoneme_language
        self.enable_eos_bos = enable_eos_bos
        self.speaker_id_mapping = speaker_id_mapping
        self.d_vector_mapping = d_vector_mapping
        self.language_id_mapping = language_id_mapping
        self.use_noise_augment = use_noise_augment

        self.verbose = verbose
        self.input_seq_computed = False
        self.rescue_item_idx = 1
        if use_phonemes and not os.path.isdir(phoneme_cache_path):
            os.makedirs(phoneme_cache_path, exist_ok=True)
        if self.verbose:
            print("\n > DataLoader initialization")
            print(" | > Use phonemes: {}".format(self.use_phonemes))
            if use_phonemes:
                print("   | > phoneme language: {}".format(phoneme_language))
            print(" | > Number of instances : {}".format(len(self.items)))

    def load_wav(self, filename):
        audio = self.ap.load_wav(filename)
        return audio

    @staticmethod
    def load_np(filename):
        data = np.load(filename).astype("float32")
        return data

    @staticmethod
    def _generate_and_cache_phoneme_sequence(
        text, cache_path, cleaners, language, custom_symbols, characters, add_blank
    ):
        """generate a phoneme sequence from text.
        since the usage is for subsequent caching, we never add bos and
        eos chars here. Instead we add those dynamically later; based on the
        config option."""
        phonemes = phoneme_to_sequence(
            text,
            [cleaners],
            language=language,
            enable_eos_bos=False,
            custom_symbols=custom_symbols,
            tp=characters,
            add_blank=add_blank,
        )
        phonemes = np.asarray(phonemes, dtype=np.int32)
        np.save(cache_path, phonemes)
        return phonemes

    @staticmethod
    def _load_or_generate_phoneme_sequence(
        wav_file, text, phoneme_cache_path, enable_eos_bos, cleaners, language, custom_symbols, characters, add_blank
    ):
        file_name = os.path.splitext(os.path.basename(wav_file))[0]

        # different names for normal phonemes and with blank chars.
        file_name_ext = "_blanked_phoneme.npy" if add_blank else "_phoneme.npy"
        cache_path = os.path.join(phoneme_cache_path, file_name + file_name_ext)
        try:
            phonemes = np.load(cache_path)
        except FileNotFoundError:
            phonemes = TTSDataset._generate_and_cache_phoneme_sequence(
                text, cache_path, cleaners, language, custom_symbols, characters, add_blank
            )
        except (ValueError, IOError):
            print(" [!] failed loading phonemes for {}. " "Recomputing.".format(wav_file))
            phonemes = TTSDataset._generate_and_cache_phoneme_sequence(
                text, cache_path, cleaners, language, custom_symbols, characters, add_blank
            )
        if enable_eos_bos:
            phonemes = pad_with_eos_bos(phonemes, tp=characters)
            phonemes = np.asarray(phonemes, dtype=np.int32)
        return phonemes

    def load_data(self, idx):
        item = self.items[idx]

        if len(item) == 5:
            text, wav_file, speaker_name, language_name, attn_file = item
        else:
            text, wav_file, speaker_name, language_name = item
            attn = None
        raw_text = text

        wav = np.asarray(self.load_wav(wav_file), dtype=np.float32)

        # apply noise for augmentation
        if self.use_noise_augment:
            wav = wav + (1.0 / 32768.0) * np.random.rand(*wav.shape)

        if not self.input_seq_computed:
            if self.use_phonemes:
                text = self._load_or_generate_phoneme_sequence(
                    wav_file,
                    text,
                    self.phoneme_cache_path,
                    self.enable_eos_bos,
                    self.cleaners,
                    language_name if language_name else self.phoneme_language,
                    self.custom_symbols,
                    self.characters,
                    self.add_blank,
                )
            else:
                text = np.asarray(
                    text_to_sequence(
                        text,
                        [self.cleaners],
                        custom_symbols=self.custom_symbols,
                        tp=self.characters,
                        add_blank=self.add_blank,
                    ),
                    dtype=np.int32,
                )

        assert text.size > 0, self.items[idx][1]
        assert wav.size > 0, self.items[idx][1]

        if "attn_file" in locals():
            attn = np.load(attn_file)

        if len(text) > self.max_seq_len:
            # return a different sample if the phonemized
            # text is longer than the threshold
            # TODO: find a better fix
            return self.load_data(self.rescue_item_idx)

        sample = {
            "raw_text": raw_text,
            "text": text,
            "wav": wav,
            "attn": attn,
            "item_idx": self.items[idx][1],
            "speaker_name": speaker_name,
            "language_name": language_name,
            "wav_file_name": os.path.basename(wav_file),
        }
        return sample

    @staticmethod
    def _phoneme_worker(args):
        item = args[0]
        func_args = args[1]
        text, wav_file, *_ = item
        phonemes = TTSDataset._load_or_generate_phoneme_sequence(wav_file, text, *func_args)
        return phonemes

    def compute_input_seq(self, num_workers=0):
        """compute input sequences separately. Call it before
        passing dataset to data loader."""
        if not self.use_phonemes:
            if self.verbose:
                print(" | > Computing input sequences ...")
            for idx, item in enumerate(tqdm.tqdm(self.items)):
                text, *_ = item
                sequence = np.asarray(
                    text_to_sequence(
                        text,
                        [self.cleaners],
                        custom_symbols=self.custom_symbols,
                        tp=self.characters,
                        add_blank=self.add_blank,
                    ),
                    dtype=np.int32,
                )
                self.items[idx][0] = sequence

        else:
            func_args = [
                self.phoneme_cache_path,
                self.enable_eos_bos,
                self.cleaners,
                self.phoneme_language,
                self.custom_symbols,
                self.characters,
                self.add_blank,
            ]
            if self.verbose:
                print(" | > Computing phonemes ...")
            if num_workers == 0:
                for idx, item in enumerate(tqdm.tqdm(self.items)):
                    phonemes = self._phoneme_worker([item, func_args])
                    self.items[idx][0] = phonemes
            else:
                with Pool(num_workers) as p:
                    phonemes = list(
                        tqdm.tqdm(
                            p.imap(TTSDataset._phoneme_worker, [[item, func_args] for item in self.items]),
                            total=len(self.items),
                        )
                    )
                    for idx, p in enumerate(phonemes):
                        self.items[idx][0] = p

    def sort_items(self):
        r"""Sort instances based on text length in ascending order"""
        lengths = np.array([len(ins[0]) for ins in self.items])

        idxs = np.argsort(lengths)
        new_items = []
        ignored = []
        for i, idx in enumerate(idxs):
            length = lengths[idx]
            if length < self.min_seq_len or length > self.max_seq_len:
                ignored.append(idx)
            else:
                new_items.append(self.items[idx])
        # shuffle batch groups
        if self.batch_group_size > 0:
            for i in range(len(new_items) // self.batch_group_size):
                offset = i * self.batch_group_size
                end_offset = offset + self.batch_group_size
                temp_items = new_items[offset:end_offset]
                random.shuffle(temp_items)
                new_items[offset:end_offset] = temp_items
        self.items = new_items

        if self.verbose:
            print(" | > Max length sequence: {}".format(np.max(lengths)))
            print(" | > Min length sequence: {}".format(np.min(lengths)))
            print(" | > Avg length sequence: {}".format(np.mean(lengths)))
            print(
                " | > Num. instances discarded by max-min (max={}, min={}) seq limits: {}".format(
                    self.max_seq_len, self.min_seq_len, len(ignored)
                )
            )
            print(" | > Batch group size: {}.".format(self.batch_group_size))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.load_data(idx)

    def collate_fn(self, batch):
        r"""
        Perform preprocessing and create a final data batch:
        1. Sort batch instances by text-length
        2. Convert Audio signal to Spectrograms.
        3. PAD sequences wrt r.
        4. Load to Torch.
        """

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.abc.Mapping):

            text_lenghts = np.array([len(d["text"]) for d in batch])

            # sort items with text input length for RNN efficiency
            text_lenghts, ids_sorted_decreasing = torch.sort(torch.LongTensor(text_lenghts), dim=0, descending=True)

            wav = [batch[idx]["wav"] for idx in ids_sorted_decreasing]
            item_idxs = [batch[idx]["item_idx"] for idx in ids_sorted_decreasing]
            text = [batch[idx]["text"] for idx in ids_sorted_decreasing]
            raw_text = [batch[idx]["raw_text"] for idx in ids_sorted_decreasing]

            speaker_names = [batch[idx]["speaker_name"] for idx in ids_sorted_decreasing]

            # get language ids from language names
            if self.language_id_mapping is not None:
                language_names = [batch[idx]["language_name"] for idx in ids_sorted_decreasing]
                language_ids = [self.language_id_mapping[ln] for ln in language_names]
            else:
                language_ids = None
            # get pre-computed d-vectors
            if self.d_vector_mapping is not None:
                wav_files_names = [batch[idx]["wav_file_name"] for idx in ids_sorted_decreasing]
                d_vectors = [self.d_vector_mapping[w]["embedding"] for w in wav_files_names]
            else:
                d_vectors = None
            # get numerical speaker ids from speaker names
            if self.speaker_id_mapping:
                speaker_ids = [self.speaker_id_mapping[sn] for sn in speaker_names]
            else:
                speaker_ids = None
            # compute features
            mel = [self.ap.melspectrogram(w).astype("float32") for w in wav]

            mel_lengths = [m.shape[1] for m in mel]

            # lengths adjusted by the reduction factor
            mel_lengths_adjusted = [
                m.shape[1] + (self.outputs_per_step - (m.shape[1] % self.outputs_per_step))
                if m.shape[1] % self.outputs_per_step
                else m.shape[1]
                for m in mel
            ]

            # compute 'stop token' targets
            stop_targets = [np.array([0.0] * (mel_len - 1) + [1.0]) for mel_len in mel_lengths]

            # PAD stop targets
            stop_targets = prepare_stop_target(stop_targets, self.outputs_per_step)

            # PAD sequences with longest instance in the batch
            text = prepare_data(text).astype(np.int32)

            # PAD features with longest instance
            mel = prepare_tensor(mel, self.outputs_per_step)

            # B x D x T --> B x T x D
            mel = mel.transpose(0, 2, 1)

            # convert things to pytorch
            text_lenghts = torch.LongTensor(text_lenghts)
            text = torch.LongTensor(text)
            mel = torch.FloatTensor(mel).contiguous()
            mel_lengths = torch.LongTensor(mel_lengths)
            stop_targets = torch.FloatTensor(stop_targets)

            if d_vectors is not None:
                d_vectors = torch.FloatTensor(d_vectors)

            if speaker_ids is not None:
                speaker_ids = torch.LongTensor(speaker_ids)

            if language_ids is not None:
                language_ids = torch.LongTensor(language_ids)

            # compute linear spectrogram
            if self.compute_linear_spec:
                linear = [self.ap.spectrogram(w).astype("float32") for w in wav]
                linear = prepare_tensor(linear, self.outputs_per_step)
                linear = linear.transpose(0, 2, 1)
                assert mel.shape[1] == linear.shape[1]
                linear = torch.FloatTensor(linear).contiguous()
            else:
                linear = None

            # format waveforms
            wav_padded = None
            if self.return_wav:
                wav_lengths = [w.shape[0] for w in wav]
                max_wav_len = max(mel_lengths_adjusted) * self.ap.hop_length
                wav_lengths = torch.LongTensor(wav_lengths)
                wav_padded = torch.zeros(len(batch), 1, max_wav_len)
                for i, w in enumerate(wav):
                    mel_length = mel_lengths_adjusted[i]
                    w = np.pad(w, (0, self.ap.hop_length * self.outputs_per_step), mode="edge")
                    w = w[: mel_length * self.ap.hop_length]
                    wav_padded[i, :, : w.shape[0]] = torch.from_numpy(w)
                wav_padded.transpose_(1, 2)

            # collate attention alignments
            if batch[0]["attn"] is not None:
                attns = [batch[idx]["attn"].T for idx in ids_sorted_decreasing]
                for idx, attn in enumerate(attns):
                    pad2 = mel.shape[1] - attn.shape[1]
                    pad1 = text.shape[1] - attn.shape[0]
                    attn = np.pad(attn, [[0, pad1], [0, pad2]])
                    attns[idx] = attn
                attns = prepare_tensor(attns, self.outputs_per_step)
                attns = torch.FloatTensor(attns).unsqueeze(1)
            else:
                attns = None
            # TODO: return dictionary
            return (
                text,
                text_lenghts,
                speaker_names,
                linear,
                mel,
                mel_lengths,
                stop_targets,
                item_idxs,
                d_vectors,
                speaker_ids,
                attns,
                wav_padded,
                raw_text,
                language_ids,
            )

        raise TypeError(
            (
                "batch must contain tensors, numbers, dicts or lists;\
                         found {}".format(
                    type(batch[0])
                )
            )
        )
