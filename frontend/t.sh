/usr/bin/google-chrome \
    --headless --disable-gpu \
    --no-sandbox \
    --user-data-dir=./user_data
    # --remote-debugging-address=0.0.0.0 \
exit $?
