FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter
MAINTAINER Louis Tiao <louistiao@gmail.com>

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

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
        bazel \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* ~/.cache/matplotlib

# TODO: copy to temporary dir rather than unknown current dir
COPY requirements*.txt ./
RUN python -m pip install --upgrade pip==20.1.1 keyrings.alt==3.4.0 && \
    python -m pip install --no-cache-dir -r requirements_dev.txt && \
	python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --upgrade https://storage.googleapis.com/jax-releases/cuda101/jaxlib-0.1.50-cp36-none-manylinux2010_x86_64.whl && \
    python -m pip install --upgrade jax

# Install TensorFlow Probability (building and installing the wheel ourselves)
# TODO: take care of temporary dir properly (not this /tmp/whtvr placeholder)
COPY probability /tmp/whtvr/probability
WORKDIR /tmp/whtvr/probability
RUN bazel build --copt=-O3 --copt=-march=native :pip_pkg && \
    PKGDIR=$(mktemp -d) && \
    ./bazel-bin/pip_pkg $PKGDIR && \
    python -m pip install --upgrade $PKGDIR/*.whl

# Install GPFlow (taking care of its dependencies ourselves)
RUN python -m pip install --no-cache-dir multipledispatch tabulate dataclasses && \
    python -m pip install --no-cache-dir --no-deps gpflow==2.0.1
RUN python -m pip install --upgrade GPy==1.9.9 gpyopt==1.2.6

RUN mkdir -p /usr/src/app

COPY . /usr/src/app
WORKDIR /usr/src/app
RUN python -m pip install --no-deps -e .

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]
