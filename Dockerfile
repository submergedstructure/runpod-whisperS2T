ARG SKIP_TENSORRT_LLM=True

FROM shashikg/whisper_s2t:dev


# Python dependencies
COPY builder/requirements.txt /requirements.txt

RUN python3.10 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt


# Add src files (Worker Template)
ADD src .

COPY builder/setup.py /setup.py

RUN CUDNN_PATH=$(python3.10 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))') && \
    echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:'${CUDNN_PATH} >> ~/.bashrc


RUN python3.10 /setup.py && \
    rm /setup.py

CMD ["python3.10", "-u", "handler.py"]
