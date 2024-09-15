#!/usr/bin/sh

export PATH=$PATH:/opt/chrome-linux64
# export VDPAU_DRIVER=nvidia

# url="https://www.google.co.jp"
# url="https://twitter.com/XDevelopers/status/1722314289233477821"
# url=""
# url="https://www.yahoo.co.jp"
# url="https://t.co/x4oqMTqpfK"
url="https://twitter.com/ai_database/status/1756147762822058066"
if [ "$1" != "" ]; then
    url="$1"
fi

chrome \
    --no-sandbox \
    --disable-dev-shm-usage \
    --enable-chrome-browser-cloud-management \
    --start-maximized \
    $url

    # --disable-gpu \
