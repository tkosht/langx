#!/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d/../

git checkout src .eslintrc.js
git status | egrep deleted | awk '{print $NF}' | xargs git checkout
