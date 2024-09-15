curl -sS -X POST "http://localhost:8000/post?type=aaa" \
    -H  "accept: application/json" \
    -H  "Content-Type: application/json" \
    -d '{
"mode":"pretrain",
"color":"red"
}' \
| jq .
