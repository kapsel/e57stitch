# e57stitch web app. Code only: scans, caches, projects and links are
# bind-mounted from the host (see compose.yaml).
#   docker build -t e57stitch .
#   docker build --build-arg VERSION=0.1.0 -t e57stitch:0.1.0 .
FROM python:3.12-slim-bookworm

ARG VERSION=dev
LABEL org.opencontainers.image.title="e57stitch" \
      org.opencontainers.image.description="Register multiple E57 terrestrial laser scans into one aligned point cloud, in a local web app." \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}"

# open3d needs libGL/libgomp at import time even when headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libegl1 libglib2.0-0 libgomp1 libusb-1.0-0 libx11-6 sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md constraints.txt ./
COPY e57stitch ./e57stitch
# constraints.txt pins the dependency versions the release was tested with
RUN pip install --no-cache-dir -c constraints.txt . && python -c "import open3d, pye57, small_gicp, PIL"
COPY tests ./tests

ENV PYTHONUNBUFFERED=1 \
    E57STITCH_VERSION=${VERSION}
EXPOSE 9090
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9090/api/health', timeout=8)" || exit 1
CMD ["python", "-m", "e57stitch", "serve", "--host", "0.0.0.0", "--port", "9090", "--no-open"]
