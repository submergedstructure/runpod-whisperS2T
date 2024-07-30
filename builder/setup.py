import whisper_s2t

load_model = {"model_identifier": "large-v2", "backend": "CTranslate2", "asr_options": {"word_timestamps": True}}

model = whisper_s2t.load_model(**load_model)