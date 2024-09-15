#!/bin/sh


sudo apt install -y git gcc make openssl libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev

# sudo apt install -y zlib1g-dev

pyenv_root="$HOME/.pyenv"
if [ "$1" = "-f" ]; then
    rm -rf $pyenv_root
else
    if [ -e "$pyenv_root" ]; then
        echo "Already Exists: $pyenv_root"
        exit 2
    fi
fi
git clone https://github.com/yyuu/pyenv.git $pyenv_root
git clone https://github.com/yyuu/pyenv-virtualenv.git $pyenv_root/plugins/pyenv-virtualenv

bashrc='
# added follows by custom pyenv installer
# BEGIN #
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
# END #
'

echo "$bashrc" >> $HOME/.bashrc
exec $SHELL -l

