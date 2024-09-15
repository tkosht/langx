#!/usr/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d/../
. ../../../.env


query="情報通信白書のポイントは？"
if [ "$1" != "" ]; then
    query=$1
fi

curl -X POST 'https://api.dify.ai/v1/chat-messages' \
--header 'Authorization: Bearer '$DIFY_API_KEY_ANALYSIS \
--header 'Content-Type: application/json' \
--data-raw '{
    "inputs": {},
    "query": "'$query'",
    "response_mode": "blocking",
    "conversation_id": "",
    "user": "tkosht",
    "files": []
}'
