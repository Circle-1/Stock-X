FROM alpine:latest
RUN apk update
RUN apk add --update --no-cache python3 py3-pip && ln -sf python3 /usr/bin/python3
RUN pip3 -q install pip --upgrade
RUN mkdir src
WORKDIR /src/
COPY . .
RUN pip3 install jupyter
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
EXPOSE 8888
EXPOSE 80
