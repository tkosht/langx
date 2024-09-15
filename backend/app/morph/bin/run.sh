#!/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../

# export MECABRC="/etc/mecabrc"

echo "`date +'%Y/%m/%d %T'` - Start" | tee .run.log
# unbuffer python classify.py $* | tee -a .run.log
python classify.py $* | tee -a .run.log
echo "`date +'%Y/%m/%d %T'` - End" | tee -a .run.log

