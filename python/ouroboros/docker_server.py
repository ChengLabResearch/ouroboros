from multiprocessing import freeze_support
import os

import uvicorn

from ouroboros.common.server import (
    DOCKER_HOST,
    DOCKER_PORT,
    create_server,
)
from ouroboros.common.server_api import create_api

app = create_server(docker=True)

tasks = {}

create_api(app, docker=True)


def main():
    reload_enabled = os.getenv("OUR_HOT_RELOAD", "").lower() in {"1", "true", "yes", "on"}

    uvicorn.run(
        "ouroboros.docker_server:app" if reload_enabled else app,
        host=DOCKER_HOST,
        port=DOCKER_PORT,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    # Necessary to run multiprocessing in child processes
    freeze_support()

    main()
