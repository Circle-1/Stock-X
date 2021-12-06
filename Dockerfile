FROM python:bullseye
RUN pip3 install --no-cache-dir setuptools
RUN pip3 install tensorflow
RUN pip3 install matplotlib pandas seaborn scikit-learn
WORKDIR /
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
