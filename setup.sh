## ----- DONE BY PRIYAM PAL -----

#!/bin/bash

# Exit on error
set -e

# Ensure Conda is available in this script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create the Conda environment if it doesn't exist
if ! conda info --envs | grep -q "social_media_content_env"; then
    echo "Creating Conda environment 'social_media_content_env' with Python 3.12.4..."
    conda create -y -n social_media_content_env python=3.12.4
else
    echo "Conda environment 'social_media_content_env' already exists."
fi

# # Activate the Conda environment
# echo "Activating Conda environment 'social_media_content_env'..."
# conda init bash
# source ~/.bashrc   
# conda activate social_media_content_env

# # Start a new interactive shell with the activated environment
# exec bash --rcfile <(echo "source activate social_media_content_env")

# Activate the Conda environment
echo "Activating Conda environment 'social_media_content_env'..."
conda activate social_media_content_env

# Start a new interactive shell with the activated environment
# exec bash --rcfile <(echo "source activate social_media_content_env")

# Installing the ChromeDriver
brew install --cask chromedriver

# Installing the Requirements.txt
pip install -r requirements.txt
