from __future__ import annotations

from src.sirtrade.env import load_env_file
from src.sirtrade.health_server import serve_health_server


def main() -> None:
    load_env_file()
    serve_health_server()


if __name__ == "__main__":
    main()
