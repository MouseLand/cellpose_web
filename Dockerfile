FROM ubuntu:18.04

# System packages
RUN apt-get update && apt-get install -y \
        bzip2 \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		gfortran \
		git \
		pkg-config \
		software-properties-common \
		unzip \
		wget \
        libgl1-mesa-glx \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*


WORKDIR /home/app

RUN curl -LO https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /home/miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/home/miniconda/bin:${PATH}

#If we add the requirements and install dependencies first, docker can use cache if requirements don't change
ADD requirements.txt /home/app
RUN pip install --no-cache-dir -r requirements.txt

ADD . /home/app

EXPOSE 5000
USER 1001