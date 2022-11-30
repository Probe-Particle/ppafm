FROM ubuntu:22.10

RUN apt-get update && apt-get install -y \
    build-essential                      \
    python3.6                            \
    python3-distutils                    \
    python3-pip                          \
    python3-apt                          \
    vim

ENV PATH="${PATH}:/app/"

COPY ./ /app