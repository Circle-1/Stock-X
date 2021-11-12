FROM ubuntu:latest
RUN apt-get update
RUN apt install python3 py3-pip && ln -sf python3 /usr/bin/python
RUN pip3 install --no-cache-dir setuptools
RUN pip3 install --no-cache-dir numpy pandas matplotlib seaborn
RUN pip3 install conda
RUN conda install -c conda-forge tensorflow
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
