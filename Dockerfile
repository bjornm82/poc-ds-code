FROM conda/c3i-linux-64

RUN conda update -n base -c defaults conda

COPY conda.yaml /opt/conda.yaml
RUN conda env create -f /opt/conda.yaml

# Probably separate with base image

RUN pip install matplotlib

COPY requirements.txt /opt/requirements.txt
RUN pip install -r /opt/requirements.txt

COPY ./model /opt

RUN chmod 755 /opt/train.py

WORKDIR /opt
