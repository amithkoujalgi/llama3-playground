import asyncio
import logging
import os

import uvicorn

from llama3_playground.server.app import app

if __name__ == "__main__":
    HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    PORT = int(os.getenv("SERVER_PORT", 8070))
    HTTP_GATEWAY_TIMEOUT_SECONDS = int(os.getenv("HTTP_GATEWAY_TIMEOUT_SECONDS", 180))

    # noinspection DuplicatedCode
    logging.info(f"Starting web server on {HOST}:{PORT}")
    config = uvicorn.Config(
        app,
        host=HOST,
        port=PORT,
        timeout_keep_alive=HTTP_GATEWAY_TIMEOUT_SECONDS,
        server_header=False,
    )  # TODO: optimize (5s)
    server_app = uvicorn.Server(config=config)
    app.debug = True
    # noinspection HttpUrlsUsage
    logging.info(f"HTTP gateway timeout is set to {HTTP_GATEWAY_TIMEOUT_SECONDS} seconds.")
    # noinspection HttpUrlsUsage
    logging.info(f"API Docs at: http://{HOST}:{PORT}/docs")
    # noinspection HttpUrlsUsage
    logging.info(f"ReDoc at: http://{HOST}:{PORT}/redoc")
    asyncio.run(server_app.serve())
