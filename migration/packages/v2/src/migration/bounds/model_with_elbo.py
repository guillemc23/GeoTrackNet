from __future__ import absolute_import, division, print_function

import tensorflow as tf

import migration.nested_utils as nested

# called with:
    # (model,
    # (inputs, targets),  each with (99, 32, 702)  // (longest time series, batch size, attributes) (t, B, A)
    # lengths, # (32,) , batch size length, length of each time series
    # num_samples=1)

def call(cell,
         inputs,
         seq_lengths,
         num_samples=1,
         parallel_iterations=30,
         swap_memory=True):
    
