""" Example handler file. """

import runpod
import whisper_s2t
import os
import time
import gc 
import base64
import tempfile
import traceback
import requests

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.


def base64_to_tempfile(base64_data):
    """
    Decode base64 data and write it to a temporary file.
    Returns the path to the temporary file.
    """
    # Decode the base64 data to bytes
    audio_data = base64.b64decode(base64_data)

    # Create a temporary file and write the decoded data
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    with open(temp_file.name, 'wb') as file:
        file.write(audio_data)

    return temp_file.name

def download_file(url):
    """
    Download a file from a URL to a temporary file and return its path.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to download file from URL")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    temp_file.write(response.content)
    temp_file.close()
    return temp_file.name


def one_sentence_per_segment(transcript, end_punct_marks=["?", ".", "!"]):
    if 'word_timestamps' not in transcript[0]:
        print(f"Word Timestamp not available, one utterance can have multiple sentences.")
        return transcript

    new_transcript = []

    all_words = []
    for utt in transcript:
        all_words += utt['word_timestamps']

    curr_utt = []
    for word in all_words:
        curr_utt.append(word)
        if len(word['word']) and word['word'][-1] in end_punct_marks:
            if len(curr_utt):
                new_transcript.append({
                    'text': " ".join([_['word'] for _ in curr_utt]),
                    'start_time': curr_utt[0]['start'],
                    'end_time': curr_utt[-1]['end'],
                    'word_timestamps': curr_utt
                })

                curr_utt = []

    if len(curr_utt):
        new_transcript.append({
            'text': " ".join([_['word'] for _ in curr_utt]),
            'start_time': curr_utt[0]['start'],
            'end_time': curr_utt[-1]['end'],
            'word_timestamps': curr_utt
        })

    return new_transcript

def one_sentence_per_segment_in_each_transcript(transcripts, end_punct_marks=["?", ".", "!"]):
  new_out = []
  for transcript in transcripts:
    new_out.append(one_sentence_per_segment(transcript, end_punct_marks=end_punct_marks))
  return new_out

def handler(job):
    """ Handler function that will be used to process jobs. """
    try:
        job_input = job.get('input', {})
        
        load_model = {}
        
        load_model['model_identifier'] = load_model.get('model_identifier', "large-v2")
        load_model['backend'] = load_model.get('backend', 'CTranslate2')
        
        load_model['asr_options'] = load_model.get('asr_options', {})
        load_model['asr_options']['word_timestamps'] = load_model['asr_options'].get('word_timestamps', True)
        
        model = whisper_s2t.load_model(**load_model)
        
        # convert audio input to tempfiles
        files = []
        for f in job_input.get('files', []):
            if f.get('audio_base_64'):
                files.append(base64_to_tempfile(f.get('audio_base_64')))
            elif f.get('audio_url') and f.get('audio_url').startswith('http'):
                files.append(download_file(f.get('audio_url')))
            else:
                return "No audio input provided"
            
        transcribe_with_vad = job_input.get('transcribe_with_vad', {})
        transcribe_with_vad['lang_codes'] = transcribe_with_vad.get('lang_codes', ['pl', 'en'])
        transcribe_with_vad['tasks'] = transcribe_with_vad.get('tasks', ['transcribe', 'translate'])
        transcribe_with_vad['initial_prompts'] = transcribe_with_vad.get('initial_prompts', ["",""])
        transcribe_with_vad['batch_size'] = transcribe_with_vad.get('batch_size', 32)

        out = model.transcribe_with_vad(files, **transcribe_with_vad)
        out = one_sentence_per_segment_in_each_transcript(out)
        return { "in" : job_input, "out" : out}
    except Exception as e:
        return f"Error: {str(e)}, Args: {e.args}, Traceback: {''.join(traceback.format_tb(e.__traceback__))}"


runpod.serverless.start({"handler": handler})
