ARG SKIP_TENSORRT_LLM=True

FROM shashikg/whisper_s2t:dev

# --- Optional: System dependencies ---
# COPY builder/setup.sh /setup.sh
# RUN /bin/bash /setup.sh && \
#     rm /setup.sh

# Python dependencies
COPY builder/requirements.txt /requirements.txt

RUN python3.10 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Preload Model
#RUN python3.10 -c 'import whisper_s2t; whisper_s2t.load_model(model_identifier="large-v3", backend="CTranslate2", asr_options={"word_timestamps": True})'



# Add src files (Worker Template)
ADD src .

CMD ["python3.10", "-u", "handler.py"]
