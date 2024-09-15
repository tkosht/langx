wget -qO- https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor > /tmp/packages.google.gpg
sudo install -o root -g root -m 644 /tmp/packages.google.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/packages.google.gpg] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'

sudo apt-get install apt-transport-https
sudo apt-get update
sudo apt-get install -y google-chrome-stable


