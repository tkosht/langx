#!/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d/../

pj_name="frontend"
rm -rf $pj_name
vue init webpack $pj_name
