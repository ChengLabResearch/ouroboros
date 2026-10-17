from ouroboros.common.memory_observer import dispatch_frozen_observer

dispatch_frozen_observer()

from multiprocessing import freeze_support

import uvicorn

from ouroboros.common.server import create_server
from ouroboros.common.server import HOST, PORT
from ouroboros.common.server_api import create_api

app = create_server()

create_api(app)


def main():
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    # Necessary to run multiprocessing in child processes
    freeze_support()

    main()
