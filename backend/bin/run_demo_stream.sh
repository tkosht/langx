#/usr/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d/../

export SHOW_INFO=False
streamlit run app/langchain/executable/demo_stream.py --server.port 8501

