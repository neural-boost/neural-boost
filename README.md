# Neural Boost

Neural Boost targeting to boost inference performance.

It could boost operaters inference performance by 3-8x:  
1) Repalce nn.LSTM by nb.LSTM  
2) Replace nn.GRU  by nb.GRU  

Package: https://pypi.org/project/neural-boost/

## How to install

```Bash
$ pip3 install neural-boost -i https://pypi.org/simple
```

## How to use

```Python
from neural_boost import nb

nb.LSTM(...)
nb.GRU(...)

# To see ./tests/* files
```

