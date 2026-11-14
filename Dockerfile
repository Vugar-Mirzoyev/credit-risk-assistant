# The scoring service (D12). Training is not in here: it needs the 1.6 GB
# extract and the full requirements.txt, while serving needs the committed
# champion and five libraries.
#
#   docker build -t credit-risk-scoring .
#   docker run --rm -p 8000:8000 credit-risk-scoring

# Pinned by digest for the same reason requirements-serve.txt pins versions:
# the tag is reassigned every time the base image is rebuilt, so two builds
# of this file would otherwise stand on different images. A digest also
# stops security patches arriving on their own, which is why Dependabot
# watches this line (.github/dependabot.yml).
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so editing code does not reinstall them.
COPY requirements-serve.txt .
RUN python -m pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY config/config.yaml config/reason_texts.yaml config/
COPY models/champion.json models/

# The service reads files and answers requests; it has no reason to run as root.
RUN useradd --create-home --uid 10001 scorer
USER scorer

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status != 200)"

CMD ["uvicorn", "src.serving.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
