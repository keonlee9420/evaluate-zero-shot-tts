# 📜 **Abstract**

The author's officially unofficial PyTorch implementation of the evaluation protocol for DiTTo-TTS.

### 🚀 **Our Goal**
The goal of this project is to establish an open-source evaluation protocol by reproducing the evaluation setup and results from large-scale zero-shot TTS literature, contributing to future research by enabling the proposal of new models through reliable and fair evaluation.

### 💡 **Motivation**
Recently, following the pioneering work of **VALL-E** [1], there has been significant research on large-scale zero-shot TTS models (e.g., **NaturalSpeech** series [2,3,4], **Voicebox** [5], **CLaM-TTS** [6], **DiTTo-TTS** [7]). These models have been evaluated against each other using the methods established by VALL-E [1], but there is still no widely agreed-upon, publicly accessible evaluation protocol. To contribute to fair and straightforward performance evaluation and to facilitate future research based on that, we propose an evaluation protocol that can reproduce these evaluations and make it open source.

The following is an excerpt of the model performances listed in **Tables 1 & 2** of the **CLaM-TTS** [6] and **DiTTo-TTS** [7]. The scores reproduced by our project are indicated as **`(·)`**, demonstrating that we have successfully reproduced their evaluation method.

### 📊 **Table 1**: Performances for the English-only ***continuation*** task
| Model          | WER ↓ | CER ↓ | SIM-o ↑ | SIM-r ↑ |
|----------------|-------|-------|---------|---------|
| **Ground Truth**   | 2.2 **`(2.15)`**   | 0.61 **`(0.61)`**  | 0.754 **`(0.7395)`**   | -       |
| **YourTTS [8]**    | 7.57 **`(7.70)`**  | 3.06 **`(3.09)`**  | 0.3928  | -       |
| **VALL-E [1]**     | 3.8   | -     | 0.452   | 0.508   |
| **VALL-E (unofficial)**    | 3.81 **`(3.79)`**  | 1.58 **`(1.55)`**    | 0.2875 **`(0.2870)`**  | 0.3433 **`(0.3428)`**  |
| **Voicebox [5]**   | 2.0   | -     | 0.593   | 0.616   |
| **CLAM-TTS [6]**   | 2.36  | 0.79  | 0.4767  | 0.5128  |
| **DiTTo-en-XL [7]**| 1.78  | 0.48  | 0.5773  | 0.6075  |

### 📊 **Table 2**: Performances for the English-only ***cross-sentence*** task
| Model          | WER ↓  | CER ↓ | SIM-o ↑ | SIM-r ↑ | 
|----------------|--------|-------|---------|---------|
| **YourTTS [8]**    | 7.92 **`(7.85)`** | 3.18 **`(3.21)`**   | 0.3755 **`(0.3728)`** | -       |
| **VALL-E [1]**     | 5.9    | -     | -       | 0.580   |
| **VALL-E (unofficial)**     | 7.63 **`(7.54)`**  | 3.65 **`(3.62)`**    | 0.3031 **`(0.3031)`**  | 0.3700 **`(0.3699)`**  |
| **Voicebox [5]**   | 1.9    | -     | 0.662   | 0.681   |
| **CLAM-TTS [6]**   | 5.11   | 2.87  | 0.4951  | 0.5382  |
| **DiTTo-en-XL [7]**| 2.56   | 0.89  | 0.6270  | 0.6554  |

### ✅ **Supported Model Cards**
- [x] **YourTTS [8]** official [implementation](https://github.com/coqui-ai/TTS) & [checkpoint](https://github.com/Edresson/YourTTS): A state-of-the-art (SOTA) multilingual (English, French, and Portuguese) zero-shot TTS model based on VITS [9], trained on VCTK, LibriTTS, TTS-Portuguese, and M-AILABS French. It is considered one of the conventional SOTA models for zero-shot TTS, and many large-scale TTS models have used it as a baseline.
- [x] **VALL-E [1]** unofficial [implementation](https://github.com/lifeiteng/vall-e) & [checkpoint](https://github.com/lifeiteng/vall-e/issues/58#issuecomment-1483700593) by [@lifeiteng](https://github.com/lifeiteng): The pioneering large-scale neural codec language model for TTS utilizes a pre-trained neural audio codec, EnCodec. It uses both an autoregressive and an additional non-autoregressive model for discrete token generation.

In addition to the provided models, it is also possible and very easy to add a new research model. ***Feel free to submit a pull request***—we look forward to seeing some amazing models added!

# ⚙️ **Installation**

**This implementation has been tested on `torch==2.2.2+cud118` with `Python 3.8.19`, assuming the availability of a single GPU.**

### **YourTTS**
If you only want to evaluate `yourtts`, you can install the dependencies listed in `requirements_inference.txt`. We provide all the necessary checkpoints related to the YourTTS model, including the pre-trained model and inference tools, via [Git LFS](https://git-lfs.com/).

### **VALL-E by [@lifeiteng](https://github.com/lifeiteng)**
To install the requirements for `valle_lifeiteng`, first install the packages listed in `requirements_inference.txt`, and then proceed with the following steps.
```bash
pip install attridict torchmetrics==0.11.1 

# Please set the PyTorch and CUDA driver versions according to your environment (following https://k2-fsa.github.io/k2/installation/from_wheels.html#linux-cuda-example).
pip install k2==1.24.4.dev20240328+cuda11.8.torch2.2.2 -f https://k2-fsa.github.io/k2/cuda.html

pip install lhotse

cd /tmp
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt
export PYTHONPATH=/tmp/icefall:$PYTHONPATH

pip install -U encodec

apt-get install espeak-ng
pip install phonemizer==3.2.1 pypinyin==0.48.0
```

We use a [checkpoint](https://github.com/lifeiteng/vall-e/issues/58#issuecomment-1483700593) trained for 100 epochs with the LibriTTS dataset, shared by [@dohe0342](https://github.com/dohe0342), to generate samples for testing. After obtaining the checkpoint via a request to [@dohe0342](https://github.com/dohe0342), please place it in the following path: `src/models/valle_lifeiteng/ckpt/epoch-100.pt`. If you choose to store it in a different location, make sure to update the "**checkpoint**" value in the `attridict` of the `args` variable in `src/models/valle_lifeiteng/inference.py` to reflect the new path.

### 📏 **Evaluation Metrics**
To conduct the evaluation, set up the environment using `requirements_evaluation.txt`, which includes the necessary dependencies for the models used in metric measurement.

# 📝 **Evaluation**

### 🗂️ **Evalset Preparation**

Following the evaluation setting in **VALL-E** [1], we use a subset of the **LibriSpeech** test-clean dataset. This subset consists of speech clips ranging from 4 to 10 seconds (results in about 2.2 hours), each with a corresponding transcript.

We have constructed the Evalset according to the following structure and criteria, which can be downloaded [here](https://drive.google.com/drive/folders/1TfYCUpccGNOBTnGCEN3qczavMiiFQjJV?usp=sharing).

```
evalset
    ├── librispeech-test-clean
    │   ├── exp_aligned_pl3_r3
    │   │   ├── {FILE_NAME}_wav_c_{TRIAL_ID}.txt
    │   │   │── {FILE_NAME}_wav_c_{TRIAL_ID}.wav
    │   │   ├── {FILE_NAME}_wav_g.txt
    │   │   │── {FILE_NAME}_wav_g.wav
    │   │   ├── {FILE_NAME}_wav_p_{TRIAL_ID}.txt
    │   │   │── {FILE_NAME}_wav_p_{TRIAL_ID}.wav
    │   │   ├── {FILE_NAME}_wav_pg_{TRIAL_ID}.txt
    │   │   │── {FILE_NAME}_wav_pg_{TRIAL_ID}.wav
    │   │   ...
    │   └── exp_base_pl3_r3
    │   │   ...
...
```
- `exp_aligned_pl3_r3` involves word-level prompts cut based on forced-alignment information, as proposed in **CLaM-TTS** [6], to span a maximum of 3 seconds. `exp_base_pl3_r3` involves prompts strictly sliced to 3 seconds. The `pl3` notation indicates that the prompt length is capped at a maximum of 3 seconds.
- To simulate a total of three trials, as commonly done in literature, we sampled three instances for each audio file with the same file name and assigned a *TRIAL_ID* to each. The `r3` notation indicates a total of 3 trials.
- We evaluate two tasks as proposed by **Voicebox** [5]: ***cross-sentence*** prompting (samples marked as `wav_p`), using a 3-second clip from another sample of the same speaker as audio context, and ***continuation*** (samples marked as `wav_c`), using the first 3 seconds of each utterance. For `wav_p`, we randomly selected another utterance from the same speaker and used a cropped segment.
- For each model, we will generate speech that reads the text in `*_wav_g.txt` using the speaker information contained in either `wav_p` or `wav_c`. `wav_pg` refers to the original audio before `wav_p` was cropped, and `wav_g` corresponds to the ground truth audio that the model's output is compared against. *FILE_NAME* is the ID assigned to `wav_g` from the original **LibriSpeech** test-clean subset.

If you are planning a new Evalset, you can easily add it by simply following the same dataset structure.

### 🔄 **Inference with the Evalset**

You can generate the audio for each task and model using the following command:

```bash
python inference.py --dataset_key {DATASET_NAME} --model_name {MODEL_NAME} --task_key {TASK_NAME} 
```

- *DATASET_NAME* can be `librispeech-test-clean`.
- *MODEL_NAME* can be either `yourtts` or `valle_lifeiteng`.
- *TASK_NAME* can be either `cross` or `cont`.

The generated results will be saved in the following path: `samples/{DATASET_NAME}/{MODEL_NAME}/exp_{aligned|base}_pl3_r3_{RUN_ID}`. `RUN_ID` is a unique key generated based on the timestamp. For specific examples, please refer to `scripts/inference_examples.sh`.

### 🛠️ **Metrics Preparation**
**CLaM-TTS** [6] adopted an evaluation method that integrates existing approaches, and we follow this method for reproduction.
- **Speaker Similarity (SIM)**: To evaluate the speaker similarity between the speech prompt and synthesized speech, we employ [WavLM large](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view?usp=sharing) model of [WavLM-TDCNN](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) which outputs the embedding vector representing the speaker's voice attribute. Please download the checkpoint and place it in `src/utils/speaker_verification/ckpt/wavlm_large_finetune.pth`. We also borrow the definition of SIM-o and SIM-r from **Voicebox** [5]. SIM-o measures the similarity between the generated and the original target speeches, while SIM-r measures the similarity concerning the target speech reconstructed from the original speech.
- **WER / CER**: To evaluate the synthesis robustness of models, we perform automatic speech recognition (ASR) on the generated audio and calculate the word error rate (WER) and character error rate (CER) with respect to the original transcriptions. We use an ASR model, specifically the CTC-based [HuBERT-Large](https://huggingface.co/facebook/hubert-large-ls960-ft), which is pre-trained on **LibriLight** and fine-tuned on **LibriSpeech**. For the multilingual cases, we have **OpenAI's** [Whisper](https://github.com/openai/whisper/blob/main/model-card.md) `large-v2` model. We adopt **NVIDIA's** [NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing) for text normalization.

### 📈 **Evaluation with the Metrics**
We follow the procedures of **CLaM-TTS** [6] and **DiTTo-TTS** [7], using the `exp_aligned_pl3_r3` subset for the ***cross-sentence*** task and the `exp_base_pl3_r3` subset for the ***continuation*** task. The evaluation for each task can be executed using the command below.

```bash
python evaluate.py -m {METRIC_NAME} -e {METRIC_MODEL} -t {TASK_NAME} -d {SAMPLE_DIR}
```

- *METRIC_NAME* can be either `wer`, `cer`, `sim_o`, or `sim_r`.
- *METRIC_MODEL* can be either `hubert`, `whisper`, or `wavlmuni`.
- *TASK_NAME* can be either `wav_p`, `wav_c`, or `wav_g`. In `wav_g`, the Ground Truth score from **Table 1** is measured. SIM-o measures the similarity between `wav_pg` and `wav_g`.
- *SAMPLE_DIR* can be `samples/{DATASET_NAME}/{MODEL_NAME}/exp_{aligned|base}_pl3_r3_{RUN_ID}`.

The results will be saved in the following path: `results/{DATASET_NAME}/{MODEL_NAME}`. For specific examples, please refer to `scripts/evaluate_examples.sh`.

# 🔗 **Citation**

Please cite this repository by the "[Cite this repository](https://github.blog/2021-08-19-enhanced-support-citations-github/)" of **About** section (top right of the main page).

# 📚 **References**
1. Wang, C., Chen, S., Wu, Y., Zhang, Z., Zhou, L., Liu, S., ... & Wei, F. (2023). Neural codec language models are zero-shot text to speech synthesizers. arXiv preprint arXiv:2301.02111.
2. Tan, X., Chen, J., Liu, H., Cong, J., Zhang, C., Liu, Y., ... & Liu, T. Y. (2024). Naturalspeech: End-to-end text-to-speech synthesis with human-level quality. IEEE Transactions on Pattern Analysis and Machine Intelligence.
3. Shen, K., Ju, Z., Tan, X., Liu, Y., Leng, Y., He, L., ... & Bian, J. (2023). Naturalspeech 2: Latent diffusion models are natural and zero-shot speech and singing synthesizers. arXiv preprint arXiv:2304.09116.
4. Ju, Z., Wang, Y., Shen, K., Tan, X., Xin, D., Yang, D., ... & Zhao, S. (2024). Naturalspeech 3: Zero-shot speech synthesis with factorized codec and diffusion models. arXiv preprint arXiv:2403.03100.
5. Le, M., Vyas, A., Shi, B., Karrer, B., Sari, L., Moritz, R., ... & Hsu, W. N. (2024). Voicebox: Text-guided multilingual universal speech generation at scale. Advances in neural information processing systems, 36.
6. Kim, J., Lee, K., Chung, S., & Cho, J. (2024). CLaM-TTS: Improving Neural Codec Language Model for Zero-Shot Text-to-Speech. arXiv preprint arXiv:2404.02781.
7. Lee, K., Kim, D. W., Kim, J., & Cho, J. (2024). DiTTo-TTS: Efficient and Scalable Zero-Shot Text-to-Speech with Diffusion Transformer. arXiv preprint arXiv:2406.11427.
8. Casanova, E., Weber, J., Shulby, C. D., Junior, A. C., Gölge, E., & Ponti, M. A. (2022, June). Yourtts: Towards zero-shot multi-speaker tts and zero-shot voice conversion for everyone. In International Conference on Machine Learning (pp. 2709-2720). PMLR.
9. Kim, J., Kong, J., & Son, J. (2021, July). Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech. In International Conference on Machine Learning (pp. 5530-5540). PMLR.

# **Acknowledgement**
Dong Won Kim ([@ddwkim](https://github.com/ddwkim)) for the implementation of metric calculation.