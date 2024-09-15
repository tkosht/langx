#!/usr/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../

. ../.env

mkdir -p data/
cd data/



# huggingface-cli login --token $HUGGINGFACE_TOKEN


# cf. https://github.com/ggerganov/llama.cpp
# cf. https://huggingface.co/TheBloke/Llama-2-7B-GGML
# cf. https://huggingface.co/TheBloke/Llama-2-13B-GGML

# wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q5_K_M.bin
# 
# wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q4_K_M.bin
# wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve/main/llama-2-13b-chat.ggmlv3.q5_K_M.bin

# wget https://huggingface.co/TheBloke/Llama-2-70B-chat-GGML/resolve/main/llama-2-70b-chat.ggmlv3.q4_K_M.bin
# wget https://huggingface.co/TheBloke/Llama-2-70B-chat-GGML/resolve/main/llama-2-70b-chat.ggmlv3.q5_K_M.bin
wget https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GGUF/resolve/main/llama-2-7b-32k-instruct.Q8_0.gguf

