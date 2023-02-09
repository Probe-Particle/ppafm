FROM ubuntu:22.10

# build-essential needed for building the C++ code in the PPAFM package

RUN apt-get update --yes &&                       \
    apt-get install --yes --no-install-recommends \
    build-essential                               \
    python3                                       \
    python3-pip                                   \
    python-is-python3 &&                          \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash ppafm-user

RUN mkdir /exec && chown ppafm-user:ppafm-user /exec

USER ppafm-user

WORKDIR /home/ppafm-user

RUN pip install --upgrade --user pip && pip cache purge

COPY --chown=ppafm-user:ppafm-user ./ ppafm

RUN pip install -e ppafm/ && pip cache purge


ENV PATH="${PATH}:/home/ppafm-user/ppafm/"

WORKDIR /exec
