#!/usr/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d/../

sh bin/gpg.sh

sudo apt-get install -y libgbm-dev x11vnc xvfb \
    && CHROMEDRIVER_VERSION=$(curl -sS https://googlechromelabs.github.io/chrome-for-testing/LATEST_RELEASE_STABLE) \
    && curl -sSL -o /tmp/chromedriver-linux64.zip https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/$CHROMEDRIVER_VERSION/linux64/chromedriver-linux64.zip \
    && cd /tmp && unzip chromedriver-linux64.zip && sudo mv chromedriver-linux64/chromedriver /usr/local/bin/ \
    && curl -sSL -o /tmp/chrome-linux64.zip https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/$CHROMEDRIVER_VERSION/linux64/chrome-linux64.zip \
    && cd /tmp && unzip chrome-linux64.zip && sudo mv chrome-linux64 /opt

export PATH $PATH:/opt/chrome-linux64
