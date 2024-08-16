import sys
import os
os_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os_pwd)

import torch
from yourtts.TTS.tts.utils.synthesis import synthesis
from yourtts.TTS.utils.audio import AudioProcessor
from yourtts.TTS.tts.models import setup_model
from yourtts.TTS.config import load_config


MODEL_PATH = f'{os_pwd}/yourtts/downloads/best_model.pth.tar'
CONFIG_PATH = f'{os_pwd}/yourtts/downloads/config.json'
TTS_LANGUAGES = f"{os_pwd}/yourtts/downloads/language_ids.json"
TTS_SPEAKERS = f"{os_pwd}/yourtts/downloads/Test-Dataset/new_se.json"
USE_CUDA = True
print(MODEL_PATH, CONFIG_PATH)

class YourTTSModel:

    def __init__(self, model_path=MODEL_PATH):
        # load the config
        self.C = load_config(CONFIG_PATH)

        # load the audio processor
        self.ap = AudioProcessor(**self.C.audio)

        self.C.model_args['d_vector_file'] = TTS_SPEAKERS
        # self.C.model_args['use_speaker_encoder_as_loss'] = False

        self.model = setup_model(self.C)
        self.model.language_manager.set_language_ids_from_file(TTS_LANGUAGES)

        cp = torch.load(model_path, map_location=torch.device('cpu'))
        # remove speaker encoder
        model_weights = cp['model'].copy()
        # for key in list(model_weights.keys()):
        #   if "speaker_encoder" in key:
        #     del model_weights[key]

        self.model.load_state_dict(model_weights)
        self.model.eval()

        if USE_CUDA:
            self.model = self.model.cuda()

        self.model.inference_noise_scale = 0.333 # defines the noise variance applied to the random z vector at inference.
        self.model.inference_noise_scale_dp = 0.333 # defines the noise variance applied to the duration predictor z vector at inference.

    def inference(self, text, ref_path, *args):
        d_vector = self.model.speaker_manager.compute_d_vector_from_clip(ref_path)

        wav, *_ = synthesis(
            self.model,
            text,
            self.C,
            "cuda" in str(next(self.model.parameters()).device),
            self.ap,
            speaker_id=None,
            d_vector=d_vector,
            style_wav=None,
            language_id=0,
            enable_eos_bos_chars=self.C.enable_eos_bos_chars,
            use_griffin_lim=False,
            do_trim_silence=False,
        ).values()

        return torch.from_numpy(wav)[None], None, 16000
