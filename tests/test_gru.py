import torch
from torch._C import Size
import torch.nn as nn
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import time
import argparse
import random

import sys
sys.path.append(".")
from neural_boost import nb as nb


parser = argparse.ArgumentParser()
parser.add_argument('--BM',
                        help='Only open benchmark',
                        type=bool,
                        default=False)
args = parser.parse_args()

random.seed(2022)
torch.manual_seed(2022)
torch.set_printoptions(precision=4)

def generate_packed_batch_sequences(batch_size, sequence_length, input_size):
    def sort_by_seq_lens(batch, sequences_lengths, descending=True):
        sorted_seq_lens, sorting_index =\
            sequences_lengths.sort(0, descending=descending)
        sorted_batch = batch.index_select(0, sorting_index)
        idx_range = torch.arange(0, sequences_lengths.size(0))
        _, reverse_mapping = sorting_index.sort(0, descending=False)
        restoration_index = idx_range.index_select(0, reverse_mapping)
        return sorted_batch, sorted_seq_lens, sorting_index, restoration_index

    sequences = [torch.randn(sequence_length - random.randint(0, batch_size), input_size) for _ in range(batch_size)]
    # sequences = [torch.randn(sequence_length - (1 if i == 0 else 0), input_size) for i in range(batch_size)]
    # print("sequences: ", sequences)
    paded_batch = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    paded_lengths = torch.count_nonzero(paded_batch[:,:,0], dim=1)
    # print("paded_batch: ", list(paded_batch.size()))
    # print("paded_batch: ", paded_batch)
    # print("paded_lengths: ", paded_lengths)
    sorted_batch, sorted_lengths, _, restoration_idx = \
            sort_by_seq_lens(paded_batch, paded_lengths)
    # print("sorted_batch: ", list(sorted_batch.size()))
    # print("sorted_batch: ", sorted_batch)
    # print("sorted_lengths: ", sorted_lengths)
    # print("restoration_idx: ", restoration_idx)
    packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
            sorted_lengths, batch_first=True)
    # print("packed_batch: ", packed_batch)
    # print("packed_batch[0]: ", list(packed_batch[0].size()))
    # print("packed_batch[0]: ", packed_batch[0])
    # print("packed_batch[1]: ", list(packed_batch[1].size()))
    # print("packed_batch[1]: ", packed_batch[1])
    return packed_batch, restoration_idx

def restoration(outputs, restoration_idx):
    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
    # print("outputs: ", list(outputs.size()))
    # print("outputs: ", outputs)
    reordered_outputs = outputs.index_select(0, restoration_idx)
    # print("reordered_outputs: ", list(reordered_outputs.size()))
    # print("reordered_outputs: ", reordered_outputs)
    return reordered_outputs


def unit_test(rnn_cell, rnn, irnn, input, h0, batch_first, bidirectional, seq_input, res_idx):
    print("\n------------------- nn.GRUCell -------------------")
    # run GRUCell forward model
    hn_f = h0[0]

    if batch_first == True:
        seq_length = input.size()[1]
        merge_dim = 1
    else:
        seq_length = input.size()[0]
        merge_dim = 0

    for i in range(seq_length):
        if batch_first == True:
            hn_f = rnn_cell(input[:,i,:], hn_f)
        else:
            hn_f = rnn_cell(input[i,:,:], hn_f)
        # print("hn_f: ", hn_f)
        if i == 0:
            cell_output_f = hn_f
            cell_output_f = cell_output_f.unsqueeze(merge_dim)
        else:
            temp = hn_f.unsqueeze(merge_dim)
            cell_output_f = torch.cat([cell_output_f, temp], dim=merge_dim)

    if bidirectional == True:
        print("cell_output_f: ", list(cell_output_f.size()))
        hn_f = hn_f.unsqueeze(0)
        print("hn_f: ", list(hn_f.size()))
        # print(cell_output_f)
    else:
        cell_output = cell_output_f
        print("cell_output: ", list(cell_output.size()))
        cell_hn = hn_f.unsqueeze(0)
        print("cell_hn: ", list(cell_hn.size()))
        # print(cell_output)
        # print(cell_hn)

    if bidirectional == True:
        # run GRUCell forward reverse model
        hn_b = h0[1]
        for i in range(seq_length):
            if batch_first == True:
                hn_b = rnn_cell(input[:,seq_length-i-1,:], hn_b)
            else:
                hn_b = rnn_cell(input[seq_length-i-1,:,:], hn_b)
            # print("hn_b: ", hn_b)
            if i == 0:
                cell_output_b = hn_b
                cell_output_b = cell_output_b.unsqueeze(merge_dim)
            else:
                temp = hn_b.unsqueeze(merge_dim)
                cell_output_b = torch.cat([temp, cell_output_b], dim=merge_dim)
        print("cell_output_b: ", list(cell_output_b.size()))
        hn_b = hn_b.unsqueeze(0)
        print("hn_b: ", list(hn_b.size()))
        # print(cell_output_b)

        cell_output = torch.cat([cell_output_f, cell_output_b], dim=2)
        print("cell_output: ", list(cell_output.size()))
        cell_hn = torch.cat([hn_f, hn_b], dim=0)
        print("cell_hn: ", list(cell_hn.size()))
        # print(cell_output)
        # print(cell_hn)


    print("\n------------------- nn.GRU -------------------")
    print("rnn_cell.weight_ih: ", rnn_cell.weight_ih.size())
    print("rnn_cell.weight_hh: ", rnn_cell.weight_hh.size())
    print("rnn_cell.bias_ih: ", rnn_cell.bias_ih.size())
    print("rnn_cell.bias_hh: ", rnn_cell.bias_hh.size())
    print(rnn._flat_weights_names)
    rnn._flat_weights[0] = rnn_cell.weight_ih
    rnn._flat_weights[1] = rnn_cell.weight_hh
    rnn._flat_weights[2] = rnn_cell.bias_ih
    rnn._flat_weights[3] = rnn_cell.bias_hh
    if bidirectional == True:
        rnn._flat_weights[4] = rnn_cell.weight_ih
        rnn._flat_weights[5] = rnn_cell.weight_hh
        rnn._flat_weights[6] = rnn_cell.bias_ih
        rnn._flat_weights[7] = rnn_cell.bias_hh

    # run GRU model
    output, hn = rnn(input, h0)
    print("output: ", list(output.size()))
    print("hn: ", list(hn.size()))
    # print(output)
    # print(hn)

    print(torch.all((cell_output - output) < 0.000001))
    print(torch.all((cell_hn - hn) < 0.000001))

    print("\n------------------- nb.GRU -------------------")
    irnn._flat_weights[0] = rnn_cell.weight_ih
    irnn._flat_weights[1] = rnn_cell.weight_hh
    irnn._flat_weights[2] = rnn_cell.bias_ih
    irnn._flat_weights[3] = rnn_cell.bias_hh
    if bidirectional == True:
        irnn._flat_weights[4] = rnn_cell.weight_ih
        irnn._flat_weights[5] = rnn_cell.weight_hh
        irnn._flat_weights[6] = rnn_cell.bias_ih
        irnn._flat_weights[7] = rnn_cell.bias_hh

    ioutput, ihn = irnn(input, h0)
    print("ioutput: ", list(ioutput.size()))
    print("ihn: ", list(ihn.size()))
    # print(ioutput)
    # print(ihn)

    print(torch.all((ioutput - output) < 0.000001))
    print(torch.all((ihn - hn) < 0.000001))

    print("\n------------------- nb.GRU -------------------")
    ioutput2, ihn2 = irnn(input)
    print("ioutput2: ", list(ioutput2.size()))
    print("ihn2: ", list(ihn2.size()))
    # print(ioutput2)
    # print(ihn2)

    print(torch.all((ioutput2 - output) < 0.000001))
    print(torch.all((ihn2 - hn) < 0.000001))

    print("\n------------------- nn.GRU sequences -------------------")
    output, hn = rnn(seq_input, None)
    output = restoration(output, res_idx)
    print("output: ", list(output.size()))
    print("hn: ", list(hn.size()))
    # print(output)
    # print(hn)

    print("\n------------------- nb.GRU sequences -------------------")
    ioutput, ihn = irnn(seq_input, None)
    ioutput = restoration(ioutput, res_idx)
    print("ioutput: ", list(ioutput.size()))
    print("ihn: ", list(ihn.size()))
    # print(ioutput)
    # print(ihn)

    print(torch.all((ioutput - output) < 0.000001))
    print(torch.all((ihn - hn) < 0.000001))


def benchmark(rnn, irnn, input, h0, seq_input):
    loops = 100
    print("\n------------------- Perf nn.GRU -------------------")
    output, hn = rnn(input, h0)
    start_time = time.time()
    for _ in range(loops):
        output, hn = rnn(input, h0)
    end_time = time.time()
    total = (end_time - start_time) * 1000
    latency = total / loops
    print("nn.GRU latency: ", latency, " ms")

    print("\n------------------- Perf nb.GRU -------------------")
    output, hn = irnn(input, h0)
    start_time = time.time()
    for _ in range(loops):
        output, hn = irnn(input, h0)
    end_time = time.time()
    total = (end_time - start_time) * 1000
    latency = total / loops
    print("nb.GRU latency: ", latency, " ms")

    print("\n------------------- Perf nn.GRU sequences -------------------")
    output, hn = rnn(seq_input, None)
    start_time = time.time()
    for _ in range(loops):
        output, hn = rnn(seq_input, None)
    end_time = time.time()
    total = (end_time - start_time) * 1000
    latency = total / loops
    print("nn.GRU sequences latency: ", latency, " ms")

    print("\n------------------- Perf nb.GRU sequences -------------------")
    output, hn = irnn(seq_input, None)
    start_time = time.time()
    for _ in range(loops):
        output, hn = irnn(seq_input, None)
    end_time = time.time()
    total = (end_time - start_time) * 1000
    latency = total / loops
    print("nb.GRU sequences latency: ", latency, " ms")

    # # jit script model
    # script_irnn = torch.jit.script(irnn)
    # torch.jit.save(script_irnn, 'GRU.torchscript.pt')
    # script_irnn = torch.jit.load('GRU.torchscript.pt')

    # output, hn = script_irnn(input, h0)
    # start_time = time.time()
    # for _ in range(loops):
    #     output, hn = script_irnn(input, h0)
    # end_time = time.time()
    # total = (end_time - start_time) * 1000
    # latency = total / loops
    # print("scripted nb.GRU latency: ", latency, " ms")


def generate_inputs(batch_size, sequence_length, input_size, hidden_size, num_layers, batch_first, bidirectional):
    print("\n------------------- input data -------------------")
    D = 2 if bidirectional == True else 1
    output_size = hidden_size

    # data
    if batch_first == True:
        input = torch.randn(batch_size, sequence_length, input_size)
    else:
        input = torch.randn(sequence_length, batch_size, input_size)
    h0 = torch.zeros(D*num_layers, batch_size, output_size)
    print("input : ", list(input.size()))
    print("hidden: ", list(h0.size()))

    seq_input, res_idx = generate_packed_batch_sequences(batch_size, sequence_length, input_size)
    print("sequences_input[0]: ", list(seq_input[0].size()))
    print("sequences_input[1]: ", list(seq_input[1].size()))
    return input, h0, seq_input, res_idx


def create_GRU(input_size, hidden_size, num_layers, batch_first, bidirectional):
    rnn_cell = nn.GRUCell(input_size, hidden_size)
    rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first, bidirectional=bidirectional)
    irnn = nb.GRU(input_size, hidden_size, num_layers, batch_first=batch_first, bidirectional=bidirectional)
    return rnn_cell, rnn, irnn


def run(input_size, hidden_size, num_layers, batch_first, bidirectional, batch_size, sequence_length, BM):
    # GRU
    print("input_size: ", input_size)
    print("hidden_size: ", hidden_size)
    print("num_layers: ", num_layers)
    print("bidirectional: ", bidirectional)
    print("batch_first: ", batch_first)
    rnn_cell, rnn, irnn = create_GRU(input_size, hidden_size, num_layers, batch_first, bidirectional)

    # inputs
    print("batch_size: ", batch_size)
    print("seq length: ", sequence_length)
    input, h0, seq_input, res_idx = generate_inputs(batch_size, sequence_length, input_size, hidden_size, num_layers, batch_first, bidirectional)

    if BM == False:
        unit_test(rnn_cell, rnn, irnn, input, h0, batch_first, bidirectional, seq_input, res_idx)
    else:
        benchmark(rnn, irnn, input, h0, seq_input)


print("\n\n########## batch_first: True, bidirectional: True ##########\n")
run(213, 51, 1, True, True, 17, 61, args.BM)
print("\n\n########## batch_first: False, bidirectional: True ##########\n")
run(213, 51, 1, False, True, 17, 61, args.BM)
print("\n\n########## batch_first: True, bidirectional: False ##########\n")
run(213, 51, 1, True, False, 17, 61, args.BM)
print("\n\n########## batch_first: False, bidirectional: False ##########\n")
run(213, 51, 1, False, False, 17, 61, args.BM)
