FROM ubuntu:22.10

RUN apt-get update && apt-get install -y \
    build-essential                      \
    python3.6                            \
    python3-distutils                    \
    python3-pip                          \
    python3-apt                          \
    python-is-python3                    \
    vim

RUN useradd -ms /bin/bash ppafm-user

RUN mkdir /exec \
    && chown ppafm-user:ppafm-user /exec

USER ppafm-user

WORKDIR /home/ppafm-user

RUN pip install --upgrade --user pip

COPY --chown=ppafm-user:ppafm-user ./ ppafm

RUN pip install -e ppafm/


ENV PATH="${PATH}:/home/ppafm-user/ppafm/"

WORKDIR /exec
