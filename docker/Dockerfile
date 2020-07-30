FROM tensorflow/tensorflow:2.3.0-gpu-jupyter
MAINTAINER Louis Tiao <louistiao@gmail.com>

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN apt-get update && apt-get install -y --no-install-recommends \
        git-core \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-xetex \
        cm-super \
        dvipng \
        pandoc \
        msttcorefonts \
        poppler-utils \
        imagemagick \
        ffmpeg \
        graphviz \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* ~/.cache/matplotlib

# TODO: copy to temporary dir rather than unknown current dir
COPY requirements*.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip==20.2 keyrings.alt==3.4.0 && \
    python -m pip install --no-cache-dir -r requirements_dev.txt && \
	python -m pip install --no-cache-dir -r requirements.txt

# Install HpBandSter and HPO Bench
RUN python -m pip install --no-cache-dir Cython==0.29.21 && \
    python -m pip install --no-cache-dir Pyro4 serpent ConfigSpace netifaces h5py
COPY external/HpBandSter /tmp/HpBandSter
COPY external/nas_benchmarks /tmp/nas_benchmarks
# RUN python -m pip install --no-cache-dir hpbandster
RUN python -m pip install --no-deps -e /tmp/HpBandSter && \
    python -m pip install --no-deps -e /tmp/nas_benchmarks

RUN mkdir -p /usr/src/app

COPY . /usr/src/app
WORKDIR /usr/src/app
RUN python -m pip install --no-deps -e .

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]
