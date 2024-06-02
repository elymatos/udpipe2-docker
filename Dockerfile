# =====
FROM alpine AS builder
WORKDIR /opt
RUN apk update && \
	apk add --no-cache git
#RUN	git clone --branch udpipe-2 https://github.com/ufal/udpipe.git && \
COPY udpipe2 ./udpipe/
RUN	cd udpipe && \
	rm -r -f wembedding_service && \
    git clone https://github.com/ufal/wembedding_service.git

# =====
#FROM tensorflow/tensorflow:1.15.4-gpu
FROM tensorflow/tensorflow:1.15.4-gpu-py3

RUN du -h --max-depth=1 / || true

WORKDIR /udpipe
COPY --from=builder /opt/udpipe .

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
#RUN apt-get install -y software-properties-common
#RUN add-apt-repository ppa:deadsnakes/ppa
#RUN apt-get update -y
#RUN apt-get install build-essential libpq-dev libssl-dev openssl libffi-dev zlib1g-dev -y
#RUN apt-get install python3-pip python3.7-dev -y
#RUN apt-get install python3.7 -y

#RUN unlink /usr/bin/python3
#RUN ln -s /usr/bin/python3.7 /usr/bin/python3

RUN apt-get update -y
RUN apt-get install libatlas-base-dev -y

RUN pip uninstall numpy -y
RUN apt install python3-numpy -y

RUN python3 -m pip install numpy
RUN pip install --no-cache-dir ufal.chu_liu_edmonds ufal.udpipe

RUN du -h --max-depth=1 / || true

EXPOSE 8001
ENTRYPOINT python3 udpipe2_server.py 8001 --wembedding_server wembedding:8000 --logfile udpipe2_server.log --threads=4 pt_all pt_all-ud-2.12-230717:pt_all:pt models/pt_all-ud-2.12-230717.model pt_bosque https://ufal.mff.cuni.cz/udpipe/2/models#universal_dependencies_210_models