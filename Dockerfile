FROM ubuntu:22.10

# build-essential is needed for building the C++ code in the PPAFM package

RUN apt-get update --yes &&                       \
    apt-get install --yes --no-install-recommends \
    build-essential                               \
    python3                                       \
    python-is-python3                             \
    wget &&                                       \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget --no-check-certificate https://bootstrap.pypa.io/get-pip.py && python get-pip.py

RUN mkdir /exec

RUN useradd -ms /bin/bash ppafm-user

COPY ./ ppafm

RUN pip install ppafm/ && pip cache purge

WORKDIR /exec
