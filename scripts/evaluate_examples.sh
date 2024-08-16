# Ground Truth
python evaluate.py -m wer -e hubert -t wav_g ;
python evaluate.py -m cer -e hubert -t wav_g ;
python evaluate.py -m sim_o -e wavlmuni -t wav_g ;

# YourTTS cross-sentence task
python evaluate.py -m wer -e hubert -t wav_p -d samples/librispeech-test-clean/yourtts/exp_aligned_pl3_r3_20240817023121323011 ;
python evaluate.py -m cer -e hubert -t wav_p -d samples/librispeech-test-clean/yourtts/exp_aligned_pl3_r3_20240817023121323011 ;
python evaluate.py -m sim_o -e wavlmuni -t wav_p -d samples/librispeech-test-clean/yourtts/exp_aligned_pl3_r3_20240817023121323011 ;

# YourTTS continuation task
python evaluate.py -m wer -e hubert -t wav_c -d samples/librispeech-test-clean/yourtts/exp_base_pl3_r3_20240817044136056885 ;
python evaluate.py -m cer -e hubert -t wav_c -d samples/librispeech-test-clean/yourtts/exp_base_pl3_r3_20240817044136056885 ;

# VALL-E cross-sentence task
python evaluate.py -m wer -e hubert -t wav_p -d samples/librispeech-test-clean/valle_lifeiteng/exp_aligned_pl3_r3_20240817043234678390 ;
python evaluate.py -m cer -e hubert -t wav_p -d samples/librispeech-test-clean/valle_lifeiteng/exp_aligned_pl3_r3_20240817043234678390 ;
python evaluate.py -m sim_o -e wavlmuni -t wav_p -d samples/librispeech-test-clean/valle_lifeiteng/exp_aligned_pl3_r3_20240817043234678390 ;
python evaluate.py -m sim_r -e wavlmuni -t wav_p -d samples/librispeech-test-clean/valle_lifeiteng/exp_aligned_pl3_r3_20240817043234678390 ;

# VALL-E continuation task
python evaluate.py -m wer -e hubert -t wav_c -d samples/librispeech-test-clean/valle_lifeiteng/exp_base_pl3_r3_20240817043524859130 ;
python evaluate.py -m cer -e hubert -t wav_c -d samples/librispeech-test-clean/valle_lifeiteng/exp_base_pl3_r3_20240817043524859130 ;
python evaluate.py -m sim_o -e wavlmuni -t wav_c -d samples/librispeech-test-clean/valle_lifeiteng/exp_base_pl3_r3_20240817043524859130 ;
python evaluate.py -m sim_r -e wavlmuni -t wav_c -d samples/librispeech-test-clean/valle_lifeiteng/exp_base_pl3_r3_20240817043524859130 ;