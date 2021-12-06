FROM ubuntu:latest
RUN apt-get update
RUN apt install -y python3 python3-pip wget
RUN pip3 install --no-cache-dir setuptools
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda install -c conda-forge tensorflow
RUN pip3 install matplotlib pandas seaborn scikit-learn
RUN mkdir src
WORKDIR /src/
COPY . .
RUN pip3 install jupyterlab
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter-lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
EXPOSE 8888
EXPOSE 80
