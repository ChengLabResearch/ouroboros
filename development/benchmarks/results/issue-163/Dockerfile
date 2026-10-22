FROM ouroboros-163-base:44f155af7e0f
RUN rm -rf /app/ouroboros
COPY python/ouroboros /app/ouroboros
COPY python/benchmarks /app/benchmarks
COPY python/pyproject.toml python/poetry.lock /app/campaign-source/
COPY python/test/common/test_memory_identity.py /app/test/common/test_memory_identity.py
ENV PYTHONPATH=/app OUR_ENV=docker OUROBOROS_BUILD_REVISION=e81c4d4743777d80b5b6fa70f8354d9c285ecd0e OUROBOROS_BUILD_ID=issue-163-e81c4d4
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LABEL org.opencontainers.image.revision=e81c4d4743777d80b5b6fa70f8354d9c285ecd0e
WORKDIR /app
ENTRYPOINT []
CMD ["sleep", "infinity"]
