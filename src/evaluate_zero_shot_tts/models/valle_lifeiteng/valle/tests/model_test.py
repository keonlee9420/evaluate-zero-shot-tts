# Copyright    2023                             (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import numpy as np
import torch
from icefall.utils import AttributeDict
from torchmetrics.classification import MulticlassAccuracy

from valle.models import NUM_MEL_BINS, get_model


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            cls.devices.append(torch.device("cuda", 0))
            if torch.cuda.device_count() > 1:
                torch.cuda.set_device(1)
                cls.devices.append(torch.device("cuda", 1))

    def test_vallf(self):
        params = AttributeDict()
        params.decoder_dim = 64
        params.nhead = 16
        params.num_decoder_layers = 4

        x = torch.from_numpy(np.random.randint(0, 100, size=[4, 8]))
        x_lens = torch.from_numpy(np.random.randint(4, 8, size=[4]))
        x_lens[-1] = 8

        y = torch.from_numpy(np.random.randint(0, 1000, size=[4, 16, 8]))
        y_lens = torch.from_numpy(np.random.randint(8, 16, size=[4]))
        y_lens[-1] = 16

        params.norm_first = True
        params.add_prenet = False
        params.model_name = "VALL-F"
        model = get_model(params)
        num_param = sum([p.numel() for p in model.parameters()])

        for device in self.devices:
            # VALL-F
            model.to(device)
            x = x.to(device)
            x_lens = x_lens.to(device)
            y = y.to(device)
            y_lens = y_lens.to(device)

            # Training
            codes, loss, metrics = model(x, x_lens, y, y_lens)

            # Inference
            model.eval()
            codes = model.inference(x[-1:], x_lens[-1:], y[-1:])

    def test_valle(self):
        params = AttributeDict()
        params.decoder_dim = 64
        params.nhead = 16
        params.num_decoder_layers = 4

        x = torch.from_numpy(np.random.randint(0, 100, size=[4, 8]))
        x_lens = torch.from_numpy(np.random.randint(4, 8, size=[4]))
        x_lens[-1] = 8

        y = torch.from_numpy(np.random.randint(0, 1000, size=[4, 16, 8]))
        y_lens = torch.from_numpy(np.random.randint(8, 16, size=[4]))
        y_lens[-1] = 16

        params.norm_first = False
        params.add_prenet = True
        params.model_name = "VALL-E"
        model = get_model(params)
        num_param = sum([p.numel() for p in model.parameters()])

        for device in self.devices:
            # VALL-E

            model.to(device)
            x = x.to(device)
            x_lens = x_lens.to(device)
            y = y.to(device)
            y_lens = y_lens.to(device)

            # Training
            codes, loss, metrics = model(x, x_lens, y, y_lens)
            # Inference
            model.eval()
            codes = model.inference(x[-1:], x_lens[-1:], y[-1:])

    def test_topmetric(self):
        metric_top10 = MulticlassAccuracy(1024, top_k=10, average="micro")
        metric_top1 = MulticlassAccuracy(1024, top_k=1, average="micro")
        batch_size, seq_len = 4, 16
        targets = np.random.randint(0, 1000, size=[batch_size, seq_len])
        logits = np.random.random([batch_size, 1024, seq_len]).astype(
            np.float32
        )

        larger_logits = np.clip(logits, -1.0, 1.0)
        smaller_logits = np.clip(logits, -1.0, 1.0)
        for b in range(batch_size):
            for t in range(seq_len):
                assert targets[b, t] >= 0
                larger_logits[b, targets[b, t], t] = 2.0
                smaller_logits[b, targets[b, t], t] = -2.0

        targets = torch.from_numpy(targets)
        larger_logits = torch.from_numpy(larger_logits)
        smaller_logits = torch.from_numpy(smaller_logits)

        for device in self.devices:
            metric_top10.to(device)
            metric_top1.to(device)
            targets = targets.to(device)

            one = metric_top10(larger_logits.to(device), targets)
            assert one.cpu().item() == 1.0, one.cpu().item()

            zero = metric_top1(smaller_logits.to(device), targets)
            assert zero.cpu().item() == 0.0, zero.cpu().item()

            half = metric_top1(
                torch.concat(
                    [smaller_logits.to(device), larger_logits.to(device)], dim=2
                ),
                torch.concat([targets, targets], dim=1),
            )
            assert half.cpu().item() == 0.5, half.cpu().item()

    def test_transformer(self):
        params = AttributeDict()
        params.decoder_dim = 64
        params.nhead = 4
        params.num_decoder_layers = 4

        x = torch.from_numpy(np.random.randint(0, 100, size=[4, 8]))
        x_lens = torch.from_numpy(np.random.randint(4, 8, size=[4]))
        x_lens[-1] = 8

        y = torch.from_numpy(
            np.random.random((4, 16, NUM_MEL_BINS)).astype(np.float32)
        )
        y_lens = torch.from_numpy(np.random.randint(8, 16, size=[4]))
        y_lens[-1] = 16

        params.model_name = "Transformer"
        params.norm_first = False
        params.add_prenet = True

        for device in self.devices:
            # Transformer
            model = get_model(params)
            num_param = sum([p.numel() for p in model.parameters()])

            model.to(device)
            x = x.to(device)
            x_lens = x_lens.to(device)
            y = y.to(device)
            y_lens = y_lens.to(device)

            # Training
            codes, loss, metrics = model(x, x_lens, y, y_lens)
            # Inference
            model.eval()
            codes = model.inference(x[-1:], x_lens[-1:])
            params.add_prenet = False


if __name__ == "__main__":
    unittest.main()
