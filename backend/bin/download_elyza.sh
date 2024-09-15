#!/usr/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../data

wget https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-gguf/resolve/main/ELYZA-japanese-Llama-2-7b-fast-instruct-q8_0.gguf
wget https://huggingface.co/mmnga/ELYZA-japanese-CodeLlama-7b-instruct-gguf/resolve/main/ELYZA-japanese-CodeLlama-7b-instruct-q8_0.gguf
wget https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-13b-fast-instruct-gguf/resolve/main/ELYZA-japanese-Llama-2-13b-fast-instruct-q8_0.gguf
