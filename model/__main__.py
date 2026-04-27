from __future__ import annotations

import sys


def _print_missing_dependency_help(error: ModuleNotFoundError) -> None:
    package = error.name or "required package"
    print(
        "Missing dependency while starting the model pipeline.\n"
        f"Package not found: {package}\n\n"
        "Install the project dependencies with:\n"
        "  python3 -m pip install -r requirements.txt\n\n"
        "If you need a GPU-enabled PyTorch build, install torch/torchvision first\n"
        "from the appropriate PyTorch wheel index, then rerun the command above.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    try:
        from model.cli import main
    except ModuleNotFoundError as error:
        _print_missing_dependency_help(error)
        raise SystemExit(1) from error

    main()
