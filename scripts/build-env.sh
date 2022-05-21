#/bin/bash

sudo yum install vim git cmake make numactl gcc-8.5.0 gcc-c++-8.5.0 -y

pip3 install virtualenv==16.7.7

virtualenv -p /usr/bin/python3.6 venv-py3.6-torch1.9.0
source ./venv-py3.6-torch1.9.0/bin/activate
pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

