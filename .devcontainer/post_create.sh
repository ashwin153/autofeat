#!/bin/bash
set -euo pipefail

poetry config cache-dir $(pwd)/.cache
poetry config virtualenvs.in-project true
poetry completions bash >> ~/.bash_completion
poetry install
