from __future__ import absolute_import, division, print_function

import tensorflow as tf

import migration.nested_utils as nested


def elbo(cell,
         inputs,
         seq_lengths,
         num_samples=1,
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
    init_states = cell.zero_state(batch_size * num_samples, tf.float32)
    init_inputs, init_mask = nested.read_tas([inputs_ta, mask_ta], t0)
    ta_names = ['log_weights', 'log_ess']
    tas = [tf.TensorArray(tf.float32, max_seq_len, name='%s_ta' % n)
             for n in ta_names]
    log_weights_acc = tf.zeros([num_samples, batch_size], dtype=tf.float32)
    kl_acc = tf.zeros([num_samples * batch_size], dtype=tf.float32)
    accs = (log_weights_acc, kl_acc)

    def while_predicate(t, *unused_args):
        return t < max_seq_len

    def while_step(t, rnn_state, tas, accs):
        """Implements one timestep of IWAE computation."""
        log_weights_acc, kl_acc = accs
        cur_inputs, cur_mask = nested.read_tas([inputs_ta, mask_ta], t)
        # Run the cell for one step.
        log_q_z, log_p_z, log_p_x_given_z, kl, new_state, new_rnn_out\
                                                     = cell(cur_inputs,
                                                            rnn_state,
                                                            cur_mask,
                                                            )
        # Compute the incremental weight and use it to update the current
        # accumulated weight.
        kl_acc += kl * cur_mask
        log_alpha = (log_p_x_given_z + log_p_z - log_q_z) * cur_mask
        log_alpha = tf.reshape(log_alpha, [num_samples, batch_size])
        log_weights_acc += log_alpha 
        # Calculate the effective sample size.
        ess_num = 2 * tf.reduce_logsumexp(input_tensor=log_weights_acc, axis=0)
        ess_denom = tf.reduce_logsumexp(input_tensor=2 * log_weights_acc, axis=0)
        log_ess = ess_num - ess_denom
        # Update the  Tensorarrays and accumulators.
        ta_updates = [log_weights_acc, log_ess]
        new_tas = [ta.write(t, x) for ta, x in zip(tas, ta_updates)]
        new_accs = (log_weights_acc, kl_acc)
        return t + 1, new_state, new_tas, new_accs
    
    _, _, tas, accs = tf.while_loop(cond=while_predicate,
                                    body=while_step,
                                    loop_vars=(t0, init_states, tas, accs),
                                    parallel_iterations=parallel_iterations,
                                    swap_memory=swap_memory)
    
    ## Here log_weights is acc log_weights
    log_weights, log_ess = [x.stack() for x in tas]
    final_log_weights, kl = accs
    log_p_hat = (tf.reduce_logsumexp(input_tensor=final_log_weights, axis=0) -
                                 tf.math.log(tf.cast(num_samples, dtype=tf.float32)))
    kl = tf.reduce_mean(input_tensor=tf.reshape(kl, [num_samples, batch_size]), axis=0)
    log_weights = tf.transpose(a=log_weights, perm=[0, 2, 1])
    return log_p_hat, kl, log_weights, log_ess
