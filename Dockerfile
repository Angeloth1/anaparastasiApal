FROM python:3.9-slim

LABEL maintainer="p19thom@ionio.gr"


WORKDIR /app

# Αντιγράφουμε το αρχείο requirements.txt στο κοντέινερ
COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt averaged_perceptron_tagger maxent_ne_chunker words


COPY . .

# Ορίζουμε την εντολή που θα εκτελεστεί όταν το κοντέινερ ξεκινήσει
CMD ["python", "txtPrep.py"]
