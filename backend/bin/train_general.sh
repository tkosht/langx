#!/usr/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../


cat /dev/null > log/app.log
if [ "$1" = "dryrun" ]; then
    shift
    export CUDA_LAUNCH_BLOCKING=1
    PYTHONPATH=. python app/general/executable/train.py --max-epoch=1 --max-batches=2 --no-save-on-exit $*
else
    PYTHONPATH=. python app/general/executable/train.py $*
fi
