#!/bin/sh

url_list="
http://localhost:8000/
http://localhost:8000/hello
http://localhost:8000/front/static/a.html
http://localhost:8000/static/a.html
"

for url in $url_list
do
    echo "===== [$url] ====="
    curl -sSL $url
    echo
done
