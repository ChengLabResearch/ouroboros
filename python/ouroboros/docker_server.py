from ouroboros.common.memory_observer import dispatch_frozen_observer

dispatch_frozen_observer()

from multiprocessing import freeze_support

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
    uvicorn.run(app, host=DOCKER_HOST, port=DOCKER_PORT)


if __name__ == "__main__":
    # Necessary to run multiprocessing in child processes
    freeze_support()

    main()
