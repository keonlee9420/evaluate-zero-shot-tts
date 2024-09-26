import io
import json
import os
import zipfile
from pathlib import Path
from shutil import copyfile, rmtree

import gdown
import requests

from yourtts.TTS.config import load_config
from yourtts.TTS.utils.generic_utils import get_user_data_dir


class ModelManager(object):
    """Manage TTS models defined in .models.json.
    It provides an interface to list and download
    models defines in '.model.json'

    Models are downloaded under '.TTS' folder in the user's
    home path.

    Args:
        models_file (str): path to .model.json
    """

    def __init__(self, models_file=None, output_prefix=None):
        super().__init__()
        if output_prefix is None:
            self.output_prefix = get_user_data_dir("tts")
        else:
            self.output_prefix = os.path.join(output_prefix, "tts")
        self.url_prefix = "https://drive.google.com/uc?id="
        self.models_dict = None
        if models_file is not None:
            self.read_models_file(models_file)
        else:
            # try the default location
            path = Path(__file__).parent / "../.models.json"
            self.read_models_file(path)

    def read_models_file(self, file_path):
        """Read .models.json as a dict

        Args:
            file_path (str): path to .models.json.
        """
        with open(file_path, "r", encoding="utf-8") as json_file:
            self.models_dict = json.load(json_file)

    def list_langs(self):
        print(" Name format: type/language")
        for model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                print(f" >: {model_type}/{lang} ")

    def list_datasets(self):
        print(" Name format: type/language/dataset")
        for model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                for dataset in self.models_dict[model_type][lang]:
                    print(f" >: {model_type}/{lang}/{dataset}")

    def list_models(self):
        print(" Name format: type/language/dataset/model")
        models_name_list = []
        model_count = 1
        for model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                for dataset in self.models_dict[model_type][lang]:
                    for model in self.models_dict[model_type][lang][dataset]:
                        model_full_name = f"{model_type}--{lang}--{dataset}--{model}"
                        output_path = os.path.join(self.output_prefix, model_full_name)
                        if os.path.exists(output_path):
                            print(f" {model_count}: {model_type}/{lang}/{dataset}/{model} [already downloaded]")
                        else:
                            print(f" {model_count}: {model_type}/{lang}/{dataset}/{model}")
                        models_name_list.append(f"{model_type}/{lang}/{dataset}/{model}")
                        model_count += 1
        return models_name_list

    def download_model(self, model_name):
        """Download model files given the full model name.
        Model name is in the format
            'type/language/dataset/model'
            e.g. 'tts_model/en/ljspeech/tacotron'

        Every model must have the following files:
            - *.pth.tar : pytorch model checkpoint file.
            - config.json : model config file.
            - scale_stats.npy (if exist): scale values for preprocessing.

        Args:
            model_name (str): model name as explained above.

        TODO: support multi-speaker models
        """
        # fetch model info from the dict
        model_type, lang, dataset, model = model_name.split("/")
        model_full_name = f"{model_type}--{lang}--{dataset}--{model}"
        model_item = self.models_dict[model_type][lang][dataset][model]
        # set the model specific output path
        output_path = os.path.join(self.output_prefix, model_full_name)
        output_model_path = os.path.join(output_path, "model_file.pth.tar")
        output_config_path = os.path.join(output_path, "config.json")

        if os.path.exists(output_path):
            print(f" > {model_name} is already downloaded.")
        else:
            os.makedirs(output_path, exist_ok=True)
            print(f" > Downloading model to {output_path}")
            output_stats_path = os.path.join(output_path, "scale_stats.npy")
            output_speakers_path = os.path.join(output_path, "speakers.json")
            output_speakers_id_path = os.path.join(output_path, "speaker_ids.json")

            # download files to the output path
            if self._check_dict_key(model_item, "github_rls_url"):
                # download from github release
                self._download_zip_file(model_item["github_rls_url"], output_path)
            else:
                # download from gdrive
                self._download_gdrive_file(model_item["model_file"], output_model_path)
                self._download_gdrive_file(model_item["config_file"], output_config_path)
                if self._check_dict_key(model_item, "stats_file"):
                    self._download_gdrive_file(model_item["stats_file"], output_stats_path)

            # update the scale_path.npy file path in the model config.json
            if self._check_dict_key(model_item, "stats_file") or os.path.exists(output_stats_path):
                # set scale stats path in config.json
                config_path = output_config_path
                config = load_config(config_path)
                config.audio.stats_path = output_stats_path
                config.save_json(config_path)
            # update the speakers.json file path in the model config.json to the current path
            if os.path.exists(output_speakers_path):
                # set d vector file path in config.json
                config_path = output_config_path
                config = load_config(config_path)
                config.d_vector_file = output_speakers_path
                config.save_json(config_path)
            # update the speaker_ids.json file path in the model config.json to the current path
            if os.path.exists(output_speakers_id_path):
                # set speakers id file path in config.json
                config_path = output_config_path
                config = load_config(config_path)
                # models with model_args
                if "model_args" in config.keys():
                    config.model_args["speakers_file"] = output_speakers_id_path
                else:
                    # other models
                    config.speakers_file = output_speakers_id_path
                config.save_json(config_path)


        return output_model_path, output_config_path, model_item

    def _download_gdrive_file(self, gdrive_idx, output):
        """Download files from GDrive using their file ids"""
        gdown.download(f"{self.url_prefix}{gdrive_idx}", output=output, quiet=False)

    @staticmethod
    def _download_zip_file(file_url, output_folder):
        """Download the github releases"""
        # download the file
        r = requests.get(file_url)
        # extract the file
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(output_folder)
        # move the files to the outer path
        for file_path in z.namelist()[1:]:
            src_path = os.path.join(output_folder, file_path)
            dst_path = os.path.join(output_folder, os.path.basename(file_path))
            copyfile(src_path, dst_path)
        # remove the extracted folder
        rmtree(os.path.join(output_folder, z.namelist()[0]))

    @staticmethod
    def _check_dict_key(my_dict, key):
        if key in my_dict.keys() and my_dict[key] is not None:
            if not isinstance(key, str):
                return True
            if isinstance(key, str) and len(my_dict[key]) > 0:
                return True
        return False
