FROM ubuntu:22.10

# build-essential is needed for building the C++ code in the PPAFM package

RUN apt-get update --yes &&                       \
    apt-get install --yes --no-install-recommends \
    build-essential                               \
    python3                                       \
    python3-pip                                   \
    python-is-python3 &&                          \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade --user pip && pip cache purge

RUN useradd -ms /bin/bash ppafm-user

RUN mkdir /exec

RUN chown ppafm-user:ppafm-user /exec

USER ppafm-user

COPY ./ ppafm

RUN pip install ppafm/ && pip cache purge

WORKDIR /exec
