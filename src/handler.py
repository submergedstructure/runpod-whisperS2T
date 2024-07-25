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


def handler(job):
    """ Handler function that will be used to process jobs. """
    try:
        job_input = job['input']

    
        model = whisper_s2t.load_model(model_identifier="large-v2", backend='CTranslate2', asr_options={'word_timestamps': True})
        job_input_audio_base_64 = job_input.get('audio_base_64')
        job_input_audio_url = job_input.get('audio_url')
        if job_input_audio_base_64:
            # If there is base64 data
            audio_input = base64_to_tempfile(job_input_audio_base_64)
        elif job_input_audio_url and job_input_audio_url.startswith('http'):
            # If there is an URL
            audio_input = download_file(job_input_audio_url)
        else:
            return "No audio input provided"
        
        language_code = job_input.get('language_code', 'pl')
        files = [audio_input]
        lang_codes = [language_code, "en"]
        tasks = ['transcribe', 'translate']
        initial_prompts = ["W poniższym tekście wszystkie słowa zawierają tylko litery. W tekście nie ma cyfr ani liter niealfabetycznych, takich jak $, % itp., z wyjątkiem znaków interpunkcyjnych: przecinków, kropek, wykrzykników, znaków zapytania, apostrofów i cudzysłowów.

Na przykład:

\"2014\" jest zapisane jako \"dwa tysiące czternaście\".
 \"by 2 children\" jest zapisane jako \"by two children\".
\"13zł\" jest zapisywane jako \"trzynaście złotych\".
\"100%\" jest zapisywane jako \"sto procent\".
\"Około 40-50 p.p.m.\" jest zapisane jako \"Około 40-50 p.p.m.\".

----


",
""]

        out = model.transcribe_with_vad(files,
                                        lang_codes=lang_codes,
                                        tasks=tasks,
                                        initial_prompts=initial_prompts,
                                        batch_size=32)
        return out
    except Exception as e:
        return f"Error: {str(e)}, Args: {e.args}, Traceback: {''.join(traceback.format_tb(e.__traceback__))}"


runpod.serverless.start({"handler": handler})
