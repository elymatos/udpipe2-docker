FROM python:3.7.17

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

COPY . ./app

RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip uninstall protobuf -y
RUN pip install protobuf==3.20.1

CMD python3 udpipe2_server.py 8001 --wembedding_server wembedding:8000 --logfile udpipe2_server.log --threads=4 pt_all pt_all-ud-2.12-230717:pt_all:pt models/pt_all-ud-2.12-230717.model pt_bosque https://ufal.mff.cuni.cz/udpipe/2/models#universal_dependencies_210_models
