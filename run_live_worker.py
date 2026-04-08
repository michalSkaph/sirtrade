from __future__ import annotations

from src.sirtrade.env import load_env_file
from src.sirtrade.live_worker import serve_live_worker


def main() -> None:
    load_env_file()
    serve_live_worker()


if __name__ == "__main__":
    main()