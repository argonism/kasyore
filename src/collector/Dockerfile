FROM python:3.10

RUN apt-get update &&\
    apt-get -y install locales &&\
    apt-get -y install libpq-dev python3-dev &&\
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN pip install --upgrade pip

# Poetryのインストール
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.2.2

ENV PATH /root/.local/bin:$PATH

COPY ./ /usr/kasyore

WORKDIR /usr/kasyore

RUN poetry config virtualenvs.create false &&\
    poetry install --only collector --no-dev --no-interaction --no-root
