#!/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d/../frontend

npm install --save vue vue-router bootstrap-vue bootstrap
npm install --save axios vue-axios

npm install --save-dev webpack-dev-server
npm install --save-dev prettier
npm install --save-dev eslint-plugin-prettier
npm install --save-dev @vue/eslint-config-prettier
npm install --save-dev jest @vue/test-utils
npm install --save-dev vue-jest
npm install --save-dev babel-jest
npm install --save-dev chromedriver
npm install --save-dev geckodriver
npm install --save-dev selenium-server
npm install --save-dev source-map-support
ncu -u chromedriver

npm install
npm audit fix
npm ddp
