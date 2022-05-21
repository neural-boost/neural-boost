#!/bin/bash

# test
numactl -C 2 -l python tests/test_gru.py
numactl -C 2 -l python tests/test_gru.py --BM true

numactl -C 2 -l python tests/test_lstm.py
numactl -C 2 -l python tests/test_lstm.py --BM true

