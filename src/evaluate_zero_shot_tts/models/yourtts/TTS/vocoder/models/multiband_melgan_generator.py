import torch

from yourtts.TTS.vocoder.layers.pqmf import PQMF
from yourtts.TTS.vocoder.models.melgan_generator import MelganGenerator


class MultibandMelganGenerator(MelganGenerator):
    def __init__(
        self,
        in_channels=80,
        out_channels=4,
        proj_kernel=7,
        base_channels=384,
        upsample_factors=(2, 8, 2, 2),
        res_kernel=3,
        num_res_blocks=3,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            proj_kernel=proj_kernel,
            base_channels=base_channels,
            upsample_factors=upsample_factors,
            res_kernel=res_kernel,
            num_res_blocks=num_res_blocks,
        )
        self.pqmf_layer = PQMF(N=4, taps=62, cutoff=0.15, beta=9.0)

    def pqmf_analysis(self, x):
        return self.pqmf_layer.analysis(x)

    def pqmf_synthesis(self, x):
        return self.pqmf_layer.synthesis(x)

    @torch.no_grad()
    def inference(self, cond_features):
        cond_features = cond_features.to(self.layers[1].weight.device)
        cond_features = torch.nn.functional.pad(
            cond_features, (self.inference_padding, self.inference_padding), "replicate"
        )
        return self.pqmf_synthesis(self.layers(cond_features))
