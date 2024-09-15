#!/usr/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d/../

# # cleanup previous virtual env -> use `make clean`
# rm -rf .venv/
# rm -f poetry.lock

export PATH="$HOME/.local/bin:$PATH" 

# for llama-cpp-python
# export CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
# export CMAKE_ARGS="-DLLAMA_CUBLAS=ON"
export CMAKE_ARGS="-DGGML_CUDA=ON"
export FORCE_CMAKE=1

if [ -d ".venv" ]; then
    poetry update
else
    poetry config virtualenvs.in-project true
    poetry install
fi
poetry add matplotlib
poetry run sed -i -e 's/^#font.family:\s*sans-serif/#font.family: IPAexGothic/' $(poetry run python -c 'import matplotlib as m; print(m.matplotlib_fname())')

