FROM python:3.12-slim

WORKDIR /srv
COPY backend/requirements*.txt /srv/
RUN pip install --no-cache-dir -r /srv/requirements-celery.txt

COPY evaluation /srv/evaluation
RUN pip install --no-cache-dir -e /srv/evaluation

COPY backend /srv/backend
COPY worker /srv/worker
COPY scripts /srv/scripts

ENV PYTHONPATH=/srv/backend:/srv
WORKDIR /srv/backend
