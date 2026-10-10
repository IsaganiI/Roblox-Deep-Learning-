FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch keeps the image much smaller than the default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY striax/ striax/

# Local defaults. On AWS, override these with s3:// URIs instead of mounting volumes, e.g.
#   STRIAX_MODEL_DIR=s3://my-bucket/models  STRIAX_OUTPUT_DIR=s3://my-bucket/out
ENV STRIAX_DATA_DIR=/app/data \
    STRIAX_MODEL_DIR=/app/models \
    STRIAX_OUTPUT_DIR=/app/out \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "striax"]
CMD ["all"]
