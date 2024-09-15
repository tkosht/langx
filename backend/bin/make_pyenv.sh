#!/bin/sh

ENV_NAME="lab"
if [ "$1" != "" ]; then
    ENV_NAME="$1"
fi

# pyenv virtualenv miniconda3-latest $ENV_NAME
# pyver="3.7.1"
# pyver="3.6.5"
# pyver="3.6.8"
pyver="3.10.10"
rm -rf ~/pj/$ENV_NAME
pyenv install -s $pyver
pyenv virtualenv $pyver $ENV_NAME

mkdir -p ~/pj/$ENV_NAME
cd ~/pj/$ENV_NAME
pyenv local $ENV_NAME
