from __future__ import absolute_import, division, print_function

import tensorflow as tf

import migration.nested_utils as nested


# called with:
    # (model,
    # (inputs, targets),  each with (99, 32, 702)  // (longest time series, batch size, attributes) (t, B, A)
    # lengths, # (32,) , batch size length, length of each time series
    # num_samples=1)
def elbo(cell,
         inputs,
         seq_lengths,
         parallel_iterations=30,
         swap_memory=True):
    batch_size = tf.shape(input=seq_lengths)[0]
    max_seq_len = tf.reduce_max(input_tensor=seq_lengths)
    seq_mask = tf.transpose(
            a=tf.sequence_mask(seq_lengths, maxlen=max_seq_len, dtype=tf.float32),
            perm=[1, 0])
    # not accessed
    # if num_samples > 1:
    #     inputs, seq_mask = nested.tile_tensors([inputs, seq_mask], [1, num_samples])
    # not accessed

    # tensorarray of len t, elem (B, A)
    inputs_ta, mask_ta = nested.tas_for_tensors([inputs, seq_mask], max_seq_len)

    t0 = tf.constant(0, tf.int32)
    init_states = cell.zero_state(batch_size, tf.float32)
    # init_inputs, init_mask = nested.read_tas([inputs_ta, mask_ta], t0)
    log_weights_acc = tf.zeros([1, batch_size], dtype=tf.float32)

    def while_predicate(t, *unused_args):
        return t < max_seq_len

    def while_step(t, rnn_state, log_weights_acc):
        """Implements one timestep of IWAE computation."""
        cur_inputs, cur_mask = nested.read_tas([inputs_ta, mask_ta], t)
        # Run the cell for one step.
        log_q_z, log_p_z, log_p_x_given_z, _, new_state, _\
                                                     = cell(cur_inputs,
                                                            rnn_state,
                                                            cur_mask,
                                                            )
        # Compute the incremental weight and use it to update the current
        # accumulated weight.
        log_alpha = (log_p_x_given_z + log_p_z - log_q_z) * cur_mask
        log_alpha = tf.reshape(log_alpha, [1, batch_size])
        log_weights_acc += log_alpha 
        # Calculate the effective sample size.
        # Update the  Tensorarrays and accumulators.
        new_log_weights_acc = log_weights_acc
        return t + 1, new_state, new_log_weights_acc
    
    _, _, final_log_weights = tf.while_loop(cond=while_predicate,
                                    body=while_step,
                                    loop_vars=(t0, init_states, log_weights_acc),
                                    parallel_iterations=parallel_iterations,
                                    swap_memory=swap_memory)
    
    ## Here log_weights is acc log_weights
    log_p_hat = (tf.reduce_logsumexp(input_tensor=final_log_weights, axis=0) -
                                 tf.math.log(tf.cast(1, dtype=tf.float32)))
    return log_p_hat
