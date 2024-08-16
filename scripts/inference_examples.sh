# YourTTS cross-sentence task
python inference.py --dataset_key librispeech-test-clean --model_name yourtts --task_key cross ;

# YourTTS continuation task
python inference.py --dataset_key librispeech-test-clean --model_name yourtts --task_key cont ;

# VALL-E cross-sentence task
python inference.py --dataset_key librispeech-test-clean --model_name valle_lifeiteng --task_key cross ;

# VALL-E continuation task
python inference.py --dataset_key librispeech-test-clean --model_name valle_lifeiteng --task_key cont ;