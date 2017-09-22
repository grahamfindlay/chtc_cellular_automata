#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Adapted from Larissa's cellular_automata_20170830.py

import argparse
import logging
import pickle
import pyphi
import numpy as np

def get_cm(n):
  '''Return selfloops and nearest-neighbor connections for an n-node ring.'''
  I = np.eye(n)
  return I + np.roll(I, 1, 1) + np.roll(I, -1, 1)


def get_state(z, tpm):
    # Find first states in tpm with z 1's, since in the CA's all nodes are
    # identical; it doesn't matter which state with z 1's is chosen.
    # BUT: not all states with the same number of 1's have the same phi!
    # Depends on 10101 or 11100.
    sum_rows_tpm = np.sum(tpm, axis=1)
    rows_with_z_ones = np.nonzero(sum_rows_tpm == z)[0]
    if not rows_with_z_ones.size:
        return None
    return tuple(map(int, tpm[rows_with_z_ones[0]]))


def get_tpm_from_rule(rule, n, z):
    Arule = list(map(int, bin(rule)[2:].zfill(8)))
    # Preallocate TPM
    num_states = 2**n
    tpm = np.zeros((num_states, n))
    state = np.zeros((num_states, n))

    for i in range(num_states):
        # If I don't need state anymore than I don't need to make it a matrix
        # actually
        state[i, :] = pyphi.convert.loli_index2state(i, n)
        for j in range(n):
            nn = np.array([
                (j - 1) % n,
                j,
                (j + 1) % n
            ])
            NN = int(''.join(list(map(str, list(map(int, state[i, nn]))))), 2)
            # The 7 here is hard coded for the ECA with nearest neighbor and self connections
            tpm[i, j] = Arule[7 - NN]

    return tpm


def evaluate_rule(rule, n, z):
    description = 'r{}_n{}_z{}'.format(rule, n, z)

    # Setup the pyphi log
    pyphi.config.LOG_FILE = description + '.pyphi_log'
    pyphi.config.configure_logging()
    log = logging.getLogger(pyphi.config.__name__)
    log.info('PyPhi v%s', pyphi.config.__about__.__version__)
    log.info('Current PyPhi configuration:\n %s', pyphi.config.get_config_string())

    # Get the CM, TPM, and state.
    cm = get_cm(n)
    tpm = get_tpm_from_rule(rule, n, z)
    current_state = get_state(z, tpm)


    # Build network and compute main complex
    if not current_state:
      print("No state with {} ones".format(z))
      main_complex = None
    else:
      network = pyphi.Network(tpm, connectivity_matrix=cm)
      main_complex = pyphi.compute.main_complex(network, current_state)

    # Save the results in a pickled file for analysis with Python.
    with open(description + '.pkl', 'wb') as f:
        pickle.dump(main_complex, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rule", help="rule number", type=int)
    parser.add_argument("n", help="size of the automaton", type=int)
    parser.add_argument("z", help="number of nodes in the ON state", type=int)
    args = parser.parse_args()

    evaluate_rule(args.rule, args.n, args.z)
