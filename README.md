# Neural Boost

Neural Boost targeting to boost inference performance.

It could boost operaters inference performance by `+3-14x`:
1) Repalce nn.LSTM by nb.LSTM
2) Replace nn.GRU  by nb.GRU

Package: https://pypi.org/project/neural-boost/

## Benchmark

**No Accuacy Loss ! No Quantization ! No Sparsity !**

nn.LSTM and nn.GRU are from PyTorch.  
nb.LSTM and nb.GRU are from Neural Boost.  
Tested on single core(Aliyun hfg7).

### LSTM

Performance data is latency and the lower the better.

#### Single-Batch Input

| Inputs | input_size | hidden_size | num_layers | batch_size | seq length |
| -- | -- | -- | -- | -- | -- |
| Values | 213 | 51 | 1 | 1 | 61 |

| Inputs |  | one sequence | | |
| -- | -- | -- | -- | -- |
| batch_first | bidirectional | nn.LSTM | nb.LSTM | Speedup |
| True | True | 4.416 | 0.408 | **+10.824x** |
| False | True | 4.414 | 0.409 | **+10.792x** |
| True | False | 2.209 | 0.221 | **+9.995x** |
| False | False | 2.208 | 0.209 | **+10.565x** |

#### Multi-Batch Inputs

| Inputs | input_size | hidden_size | num_layers | batch_size | seq length |
| -- | -- | -- | -- | -- | -- |
| Values | 213 | 51 | 1 | 17 | 61 |

| Inputs |  | batched sequences | | |
| -- | -- | -- | -- | -- |
| batch_first | bidirectional | nn.LSTM | nb.LSTM | Speedup |
| True | True | 8.579 | 2.671 | **+3.212x** |
| False | True | 8.375 | 2.651 | **+3.159x** |
| True | False | 4.152 | 1.348 | **+3.080x** |
| False | False | 4.064 | 1.341 | **+3.031x** |

| Inputs |  | packed variable length sequences | | |
| -- | -- | -- | -- | -- |
| batch_first | bidirectional | nn.LSTM | nb.LSTM | Speedup |
| True | True | 7.987 | 2.303 | **+3.468x** |
| False | True | 8.262 | 2.455 | **+3.365x** |
| True | False | 3.973 | 1.215 | **+3.270x** |
| False | False | 3.955 | 1.255 | **+3.151x** |

### GRU

Performance data is latency and the lower the better.

#### Single-Batch Input

| Inputs | input_size | hidden_size | num_layers | batch_size | seq length |
| -- | -- | -- | -- | -- | -- |
| Values | 213 | 51 | 1 | 1 | 61 |

| Inputs |  | one sequence | | |
| -- | -- | -- | -- | -- |
| batch_first | bidirectional | nn.GRU | nb.GRU | Speedup |
| True | True | 4.635 | 0.323 | **+14.350x** |
| False | True | 4.621 | 0.321 | **+14.396x** |
| True | False | 2.341 | 0.167 | **+14.018x** |
| False | False | 2.336 | 0.165 | **+14.158x** |

#### Multi-Batch Inputs

| Inputs | input_size | hidden_size | num_layers | batch_size | seq length |
| -- | -- | -- | -- | -- | -- |
| Values | 213 | 51 | 1 | 17 | 61 |

| Inputs |  | batched sequences | | |
| -- | -- | -- | -- | -- |
| batch_first | bidirectional | nn.GRU | nb.GRU | Speedup |
| True | True | 8.275 | 1.974 | **+4.192x** |
| False | True | 8.101 | 1.965 | **+4.123x** |
| True | False | 4.005 | 1.001 | **+4.001x** |
| False | False | 3.918 | 0.993 | **+3.946x** |

| Inputs |  | packed variable length sequences | | |
| -- | -- | -- | -- | -- |
| batch_first | bidirectional | nn.GRU | nb.GRU | Speedup |
| True | True | 7.788 | 1.698 | **+4.587x** |
| False | True | 7.987 | 1.804 | **+4.427x** |
| True | False | 3.868 | 0.896 | **+4.317x** |
| False | False | 3.805 | 0.924 | **+4.118x** |

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
## How to connect

If you have any questions and issues, please send email to me(neural_boost@163.com).
