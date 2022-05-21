# Neural Boost

Neural Boost targeting to boost inference performance.

It could boost operaters inference performance by 3-8x:
1) Repalce nn.LSTM by nb.LSTM
2) Replace nn.GRU  by nb.GRU

Package: https://pypi.org/project/neural-boost/

## How to test

```Bash
$ git clone https://github.com/neural-boost/neural-boost.git
$ bash scripts/build-env.sh
$ source venv-py3.6-torch1.9.0/bin/activate
$ bash run.sh
```

## How to use

```Bash
$ pip3 install neural-boost -i https://pypi.org/simple
```

```Python
from neural_boost import nb

nb.LSTM(...)
nb.GRU(...)
```

