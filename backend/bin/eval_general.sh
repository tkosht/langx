#!/usr/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../

# cat /dev/null > log/app.log
PYTHONPATH=. python app/general/executable/eval.py $*
