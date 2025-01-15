FROM python:3.9

ARG MODEL_DATA_REPO_TOKEN
ENV MODEL_DATA_REPO_TOKEN=$MODEL_DATA_REPO_TOKEN
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"
ENV TF_ENABLE_ONEDNN_OPTS="0"
WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

USER user
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
