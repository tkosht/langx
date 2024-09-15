#!/usr/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d/../

while :
do
    date +'%Y/%m/%d %T'
    echo "--------------------"
    $*
    sleep 1
    clear
done

