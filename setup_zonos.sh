#!/bin/bash
set -e

echo -e "\n====== Zonos Setup (CPU-Only Mode) ======\n"

# Install uv if missing
if ! command -v uv &> /dev/null; then
  echo "uv not found, installing..."
  pip install --user -U uv
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
  echo "Creating .venv virtual environment"
  uv venv
fi

# Point shell type detection
USER_SHELL="$(basename "$SHELL")"

case "$USER_SHELL" in
  fish)
    VENV_ACTIVATE=".venv/bin/activate.fish"
    ;;
  zsh|bash)
    VENV_ACTIVATE=".venv/bin/activate"
    ;;
  *)
    VENV_ACTIVATE=".venv/bin/activate"
    ;;
esac

# Do NOT automatically 'source' here; just info display.
echo "Detected shell: $USER_SHELL"
echo "Virtualenv activation script: $VENV_ACTIVATE"

# Sync deps
# Use Bash syntax to avoid problems under Fish invocation
bash -c "
  source .venv/bin/activate
  echo 'Syncing ONLY core transformer dependencies (no extras)...'
  uv sync

  echo 'Installing Zonos core package (editable, no extras)...'
  uv pip install --no-build-isolation -e .

  echo '-- Removing leftover hybrid/GPU backend packages --'
  uv pip uninstall -y flash-attn mamba-ssm triton || true
  pip uninstall -y flash-attn mamba-ssm triton || true

  echo '-- Installing runtime UI & language dependencies explicitly --'
  uv pip install -U \
    PyQt5 \
    sentencepiece \
    langcodes \
    kanjize \
    einops
"

echo -e "\n====== Setup Complete (CPU Transformer Mode ONLY) ======\n"

echo "To activate and run:"
echo

echo "  # For Bash / zsh:"
echo "  source .venv/bin/activate"
echo "  python gui.py"
echo
echo "  # For Fish shell:"
echo "  source .venv/bin/activate.fish"
echo "  python gui.py"
echo

