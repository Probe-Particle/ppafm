FROM ubuntu:22.10

# build-essential is needed for building the C++ code in the PPAFM package

RUN apt-get update --yes &&                       \
    apt-get install --yes --no-install-recommends \
    build-essential                               \
    python3                                       \
    python-is-python3                             \
    wget &&                                       \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash ppafm-user

RUN mkdir /exec

RUN chown ppafm-user:ppafm-user /exec

USER ppafm-user

ENV PATH="${PATH}:/home/ppafm-user/.local/bin"

WORKDIR /home/ppafm-user

RUN wget --no-check-certificate https://bootstrap.pypa.io/get-pip.py && python get-pip.py

COPY --chown=ppafm-user:ppafm-user . ppafm

RUN pip install --user ppafm/ && pip cache purge

WORKDIR /exec
