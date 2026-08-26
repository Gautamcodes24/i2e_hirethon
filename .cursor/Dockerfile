FROM python:3.12-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        curl \
        build-essential \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
        sudo \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

RUN useradd -m -s /bin/bash ubuntu \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ubuntu
WORKDIR /home/ubuntu
