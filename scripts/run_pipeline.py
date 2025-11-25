from __future__ import annotations

import subprocess


def main():
    subprocess.run(["python", "-m", "src.entrypoints.cli", "Sample math problem"], check=True)


if __name__ == "__main__":
    main()

