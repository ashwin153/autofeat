#!/bin/bash
set -euo pipefail

# git
git config --global --add --bool push.autoSetupRemote true
git config --global --add safe.directory /workspaces/autofeat

# ipython
mkdir -p ~/.ipython/profile_default/startup
cat <<EOF >~/.ipython/profile_default/startup/01-autoreload.py
import IPython
ipython = IPython.get_ipython()
ipython.run_line_magic(magic_name="load_ext", line="autoreload")
ipython.run_line_magic(magic_name="autoreload", line="2")
EOF

# poetry
poetry config cache-dir $(pwd)/.cache
poetry config virtualenvs.in-project true
poetry completions bash >> ~/.bash_completion
poetry install

# pre-commit
pre-commit install --install-hooks

# streamlit
mkdir ~/.streamlit
cat >~/.streamlit/config.toml <<EOF
[browser]
gatherUsageStats = false
EOF
