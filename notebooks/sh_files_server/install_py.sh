# Install Python
sudo apt update
sudo apt install python3-pip -y # -y: just continue yes
sudo apt install python3.10 python3.10-venv

# Install poetry
curl -sSL https://install.python-poetry.org | python3 -
    #PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Check Poetry version
poetry --version

# Generate cache folder for huggingface models
mkdir -p ~/.cache/huggingface
echo 'export HF_HOME="$HOME/.cache/huggingface"' >> ~/.bashrc
source ~/.bashrc

# pyenv + poetry setup
curl https://pyenv.run | bash
pyenv install 3.10.13
pyenv local 3.10.13
poetry env use ~/.pyenv/versions/3.10.13/bin/python

# Login to huggingface
poetry run hf auth login

# test python
poetry run python3 streamlit_test/pages/3_huggingface.py