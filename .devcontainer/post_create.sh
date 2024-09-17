#!/bin/bash
set -euo pipefail

poetry config cache-dir $(pwd)/.cache
poetry config virtualenvs.in-project true
poetry completions bash >> ~/.bash_completion
poetry install

git config --global --add --bool push.autoSetupRemote true
git config --global --add safe.directory /workspaces/autofeat

pre-commit install --install-hooks
