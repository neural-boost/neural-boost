import torch
from torch import Tensor
import torch.nn.modules.rnn as rnn
import torch.nn.utils.rnn as utils
from torch.nn.utils.rnn import PackedSequence
from typing import Optional


class GRU(rnn.GRU):
    def __init__(self, *args, **kwargs):
        super(GRU, self).__init__(*args, **kwargs)
        self._rnn = torch.classes.neural_boost.GRU()
        self._rnn.init(self.input_size, self.hidden_size, self.num_layers,
                       self.batch_first, self.bidirectional)
        self.dropout = None
        self.training = False

    def forward(self, input, hx: Optional[Tensor] = None):  # noqa: F811
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size, self.hidden_size,
                             dtype=input.dtype, device=input.device)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = self._rnn.gru_batch(input, hx, self._flat_weights, self.bias, self.num_layers,
                             self.bidirectional, self.batch_first)
        else:
            result = self._rnn.gru_packed(input, batch_sizes, hx, self._flat_weights, self.bias,
                             self.num_layers, self.bidirectional)
        output = result[0]
        hidden = result[1]

        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)
