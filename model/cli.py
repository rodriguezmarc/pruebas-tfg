from __future__ import annotations

from model.workflows.full import run_pipeline


def main() -> None:
    num = run_pipeline()
    print(f"Generated {num} images in results/generated.")


if __name__ == "__main__":
    main()
