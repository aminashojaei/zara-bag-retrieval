import argparse
from pathlib import Path

from zara_bag_retrieval.config import DEFAULT_DATASET, DEFAULT_RESULTS
from zara_bag_retrieval.service import VisualSearchService
from zara_bag_retrieval.ui import build_demo


def main():
    parser = argparse.ArgumentParser(description="Zara Bag Retrieval")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app_service = VisualSearchService(args.dataset, args.results, device=args.device)
    for name in app_service.available:
        app_service.store.load(name)
        print(f"Ready: {name}")
    demo = build_demo(app_service)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, show_error=True,
        allowed_paths=[str(args.dataset.resolve())]
    )


if __name__ == "__main__":
    main()
