#!/usr/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d/../

plugins_dir="data/neo4j/plugins/"
sudo mkdir -p $plugins_dir
sudo chown -R $(id -un):$(id -gn) $plugins_dir
cd $plugins_dir
wget https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/5.17.1/apoc-5.17.1-extended.jar

