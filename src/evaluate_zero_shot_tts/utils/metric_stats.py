import re
import sys
from typing import Callable, List, TextIO

import numpy as np

from .preprocess.normalize_text import TextNormalizer

from .edit_distance import (
    print_alignments,
    print_wer_summary,
    wer_details_for_batch,
    wer_summary,
)


def base_cleaner(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()


def process_text(metric: str, lang: str = "ko") -> Callable[[str], List[str]]:
    normalizer = TextNormalizer(lang=lang) if lang != "" else None
    if lang == "ko":
        regex = re.compile("[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣\s]+")
        lang_filter = lambda text: regex.sub("", text)
    elif lang == "en":
        regex = re.compile("[^a-zA-Z\s]+")
        lang_filter = lambda text: regex.sub("", text).lower()
    elif lang == "jp":
        regex = re.compile(r"[^\w\d']+")
        lang_filter = lambda text: regex.sub("", text).lower()
    else:
        # remove except letters
        regex = re.compile("[^a-zA-Zㄱ-ㅎ|ㅏ-ㅣ|가-힣\s]+")
        lang_filter = lambda text: regex.sub("", text).lower()

    def _process(text: str) -> List[str]:
        # text = text.split(":")[-1]  # remove speaker info
        if normalizer is not None:
            text = normalizer.normalize(text)
            text = lang_filter(text)
        text = base_cleaner(text)
        if metric == "wer":
            return text.split()
        elif metric == "cer":
            text = "".join(text.split())
            return list(text)

    return _process


def get_metric(name: str, lang=None, **kwargs):
    if name in ["cer", "wer"]:
        assert lang is not None
        return ErrorRateStats(name, lang, **kwargs)
    elif "sim" in name:
        return SpeakerSimilarity(name)
    else:
        raise ValueError(f"Metric {name} not supported")


class ErrorRateStats:
    """
    Accumulate error rate statistics for a dataset.
    """

    def __init__(self, name: str = "cer", lang: str = "ko"):
        self.name = name
        self.scores = []
        self.text_processor = process_text(name, lang)

    def append(
        self,
        ids: List[str],
        hyps: List[str],
        refs: List[str],
    ):
        hyps = [self.text_processor(hyp) for hyp in hyps]
        refs = [self.text_processor(ref) for ref in refs]

        scores = wer_details_for_batch(ids, refs, hyps, True)
        self.scores.extend(scores)

    def summarize(self):
        self.summary = wer_summary(self.scores)
        print_wer_summary(self.summary)

        return self.summary

    def write_stats(self, filestream: TextIO):
        print_wer_summary(self.summary, filestream)
        print_alignments(self.scores, filestream)


class SpeakerSimilarity:
    """
    Accumulate speaker similarity statistics for a dataset.
    """

    def __init__(self, name: str):
        self.name = name
        self.ids = []
        self.scores = []

    def append(self, ids: List[str], scores: List[float], *args):
        self.ids.extend(ids)
        self.scores.extend(scores)

    def summarize(self):
        mean = np.mean(self.scores)
        print(f"Mean Speaker Similarity: {mean:.4f}")

        return mean

    def write_stats(self, filestream: TextIO):
        if filestream is None:
            filestream = sys.stdout

        # output header
        print("utt_id,score,mean,std", file=filestream)

        # print aggregate stats
        print(
            f",,{np.mean(self.scores):.4f},{np.std(self.scores):.4f}",
            file=filestream,
        )

        # output in csv format for each utterance and similarity
        for utt_id, score in zip(self.ids, self.scores):
            print(f"{utt_id},{score:.4f},,", file=filestream)
