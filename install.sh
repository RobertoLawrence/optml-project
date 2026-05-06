sudo apt-get update
sudo apt-get install screen

VENV_PIP="$(dirname "$0")/venv/bin/pip"

"$VENV_PIP" install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
"$VENV_PIP" install transformers datasets wandb omegaconf jaxtyping kornia
"$VENV_PIP" install -U Pillow
"$VENV_PIP" uninstall --yes flash_attn
