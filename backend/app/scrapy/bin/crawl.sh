#!/usr/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../

mkdir -p data/
rm -f data/news.db data/news.json
scrapy crawl news --loglevel=INFO -O data/news.json
# scrapy crawl news --loglevel=INFO

