"""Find all the unique characters in a dataset"""
import argparse
from argparse import RawTextHelpFormatter

from yourtts.TTS.config import load_config
from yourtts.TTS.tts.datasets import load_meta_data

import numpy
import multiprocessing
from yourtts.TTS.tts.utils.text import text2phone
from tqdm.contrib.concurrent import process_map

def compute_phonemes(item):
    try:
        text = item[0]
        language = item[-1]
        ph = text2phone(text, language, use_espeak_phonemes=c.use_espeak_phonemes).split("|")
    except:
        return []
    return list(set(ph))
    
def main():
    global c
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Find all the unique characters or phonemes in a dataset.\n\n"""
        """
    Example runs:

    python TTS/bin/find_unique_chars.py --config_path config.json
    """,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--config_path", type=str, help="Path to dataset config file.", required=True)
    args = parser.parse_args()

    c = load_config(args.config_path)

    # load all datasets
    train_items, eval_items = load_meta_data(c.datasets, eval_split=True)
    items = train_items + eval_items
    print("Num items:", len(items))
    # items = items[:1000]

    phonemes = process_map(compute_phonemes, items, max_workers=multiprocessing.cpu_count(), chunksize=15)
    phones = []
    for ph in phonemes:
        phones.extend(ph)
    phones = set(phones)
    lower_phones = filter(lambda c: c.islower(), phones)
    phones_force_lower = [c.lower() for c in phones]
    phones_force_lower = set(phones_force_lower)

    

    print(f" > Number of unique phonemes: {len(phones)}")
    print(f" > Unique phonemes: {''.join(sorted(phones))}")
    print(f" > Unique lower phonemes: {''.join(sorted(lower_phones))}")
    print(f" > Unique all forced to lower phonemes: {''.join(sorted(phones_force_lower))}")


if __name__ == "__main__":
    main()
