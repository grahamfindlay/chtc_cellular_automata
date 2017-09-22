#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging
import functools
import os
import pickle
import scipy.io
import numpy as np
import random
from joblib import Parallel, delayed
import pyphi
from pyphi import convert
from time import time

formatter = logging.Formatter(
    fmt='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

# Global settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rules 30, 57, 105 and 54 are pretty cool.
number_of_nodes = 5

# Parallel computation settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PARALLEL_VERBOSITY = 20
# Use all but three processors.
NUMBER_OF_CORES = 6
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Flag to indicate whether we're debugging the script.
DEBUG = False

# Flag to indicate whether we're computing only the unique rules, or every
# rule.
ONLY_UNIQUE = True

# Only compute rule 105 if debugging.
if DEBUG:
    rules_to_compute = [232]
else:
    if ONLY_UNIQUE:
        rules_to_compute = range(88)
    else:
        rules_to_compute = range(256)

# The unique equivalence classes of rules.
unique_rules = functools.reduce(lambda x, y: list(x) + list(y), [
    range(0, 16), [18, 19], range(22, 31), range(32, 39), range(40, 47),
    [50, 51, 54], range(56, 59), [60, 62], range(72, 75), range(76, 79),
    [90, 94], range(104, 107), [108, 110, 122], range(126, 144, 2), [146],
    range(150, 158, 2), range(160, 166, 2), range(168, 174, 2),
    [178, 184, 200, 204, 232]
])

number_of_states = number_of_nodes + 1
# Flag to indicate whether we calculate all complexes and find the main
# complex, or only calculate the full system.
calculate_all_complexes = True
# This is the name of the folder (directory) where results will be saved.
matlab_results_dir = 'matlab_results/' + str(number_of_nodes) + '_nodes_PD2'
python_results_dir = 'python_results/' + str(number_of_nodes) + '_nodes'
# Make the results directory if it doesn't exist.
os.makedirs(matlab_results_dir, exist_ok=True)
os.makedirs(python_results_dir, exist_ok=True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

cm_filename = 'connectivity_mat' + str(number_of_nodes) + 'nodes'
# Get the connectivity matrix from the matfile.
connectivity_matrix = scipy.io.loadmat(cm_filename)['connectivity_matrix']


def get_random_state(z):
    base_state = [1] * z + [0] * (number_of_nodes - z)
    random.shuffle(base_state)
    return base_state


def get_state(z, tpm):
    # Find first states in tpm with z 1's, since in the CA's all nodes are
    # identical; it doesn't matter which state with z 1's is chosen.
    # BUT: not all states with the same number of 1's have the same phi!
    # Depends on 10101 or 11100.
    sum_rows_tpm = np.sum(tpm, axis=1)
    rows_with_z_ones = np.nonzero(sum_rows_tpm == z)[0]
    if not rows_with_z_ones.size:
        return None
    base_state = tuple(map(int, tpm[rows_with_z_ones[0]]))
    #past_state = convert.loli_index2state(rows_with_z_ones[0], number_of_nodes)
    return base_state


def make_tpm_from_rule(rule):
    Arule = list(map(int, bin(rule)[2:].zfill(8)))
    # Preallocate TPM
    num_states = 2**number_of_nodes
    tpm = np.zeros((num_states, number_of_nodes))
    state = np.zeros((num_states, number_of_nodes))

    for i in range(num_states):
        # If I don't need state anymore than I don't need to make it a matrix
        # actually
        state[i, :] = convert.loli_index2state(i, number_of_nodes)
        for j in range(number_of_nodes):
            nn = np.array([
                (j - 1) % number_of_nodes,
                j,
                (j + 1) % number_of_nodes
            ])
            NN = int(''.join(list(map(str, list(map(int, state[i, nn]))))), 2)
            # The 7 here is hard coded for the ECA with nearest neighbor and self connections
            tpm[i, j] = Arule[7 - NN]

    return tpm


def mice2dict(mice):
    """Convert a PyPhi Mice to a dictionary suitable for conversion to a
    Matlab structure with scipy."""
    #import pdb; pdb.set_trace()
    return {
        'phi': mice.phi,
        'purview': mice.purview,
        'partition': (
            ((mice.mip.partition[0].mechanism,
              mice.mip.partition[0].purview),
             (mice.mip.partition[1].mechanism,
              mice.mip.partition[1].purview))
            if mice.mip.partition is not None else 'None'
        ),
        'repertoire': (
            mice.repertoire.flatten(order = 'F') if
            mice.repertoire is not None else 'None'
        ),
        'partitioned_repertoire': (
            mice.partitioned_repertoire.flatten(order = 'F') if 
            mice.partitioned_repertoire is not None else 'None'
        ),
    }


def concept2dict(c):
    """Convert a PyPhi Concept to a dictionary suitable for conversion to a
    Matlab structure with scipy."""
    return {
        'phi': c.phi,
        'mechanism': c.mechanism,
        'cause': mice2dict(c.cause) if c.cause is not None else 'None',
        'effect': mice2dict(c.effect) if c.effect is not None else 'None'
    }


def bigmip2dict(mip, time):
    """Convert a BigMip to a dictionary suitable for conversion to a Matlab
    structure with scipy."""
    if mip is None:
        return np.array([])

    matlab_data = {
        'PhiMip': mip.phi,
        'main_complex': convert.nodes2indices(mip.subsystem.nodes),
        'MIP1': mip.cut.from_nodes,
        'MIP2': mip.cut.to_nodes,
        'current_state': mip.subsystem.state,
        'num_concepts': len(mip.unpartitioned_constellation),
        'calculation_time': time,
        'sum_small_phi': sum(c.phi for c in mip.unpartitioned_constellation),
        'partition_only_concepts': [
            concept2dict(c) for c in mip.partitioned_constellation if
            c.mechanism not in
            [
                upc.mechanism for upc in
                mip.unpartitioned_constellation
            ]
        ],
        'concepts': [concept2dict(c) for c in
                     mip.unpartitioned_constellation]
    }
    return matlab_data


def compute_phi_data(network, current_state):
    print("\n[CA] Calculating Phi...")
    print(''.center(40, '~'))
    print("[CA] Connectivity matrix:\n", network.connectivity_matrix)
    print("[CA] Current state:\n", current_state)
    
    if current_state is None: 
        return
    #import pdb; pdb.set_trace()
    main_complex = None
    elapsed = 0
    try:
        # Calculate!
        if not calculate_all_complexes:
            subsystem = pyphi.subsystem.Subsystem(network, current_state, range(network.size))
            tic = time()
            main_complex = pyphi.compute.big_mip(subsystem)
            toc = time()
        else:
            tic = time()
            main_complex = pyphi.compute.main_complex(network, current_state)
            toc = time()

            print("[CA] Found main_complex:")
            print("[CA]\tNodes:", main_complex.subsystem.nodes)
            print("[CA]\tPhi:", main_complex.phi)
            #import pdb; pdb.set_trace()
        elapsed = toc - tic
        print('\n[CA] Elapsed time:', elapsed)

    # The except shouldn't happen since the current_state is taken from the TPM
    except pyphi.validate.state_reachable(pyphi.subsystem.Subsystem(network, current_state, range(network.size))):
        print("[CA] State unreachable.")
        pass

    return main_complex, elapsed


def evaluate_rule(rule):

    if ONLY_UNIQUE and not DEBUG:
        unique_rule_index = rule
        rule = unique_rules[rule]

    rule_string = ''.join(['nodes_', str(number_of_nodes), '_rule_',
                           str(rule)])

    log = logging.getLogger(rule_string)
    handler = logging.FileHandler('logs/' + rule_string + '.log')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    if ONLY_UNIQUE and not DEBUG:    
        log.info("\n\n[CA] " + (" Rule " + str(rule) + " (Unique Rule #" +
                             str(unique_rule_index) + ") ").center(60, '='))
    else:
        log.info("\n\n[CA] " + (" Rule " + str(rule) + " ").center(60, '='))

    # Make the TPM and past and current states for this rule.
    tpm = make_tpm_from_rule(rule)
    net = pyphi.Network(tpm, connectivity_matrix=connectivity_matrix)
    current_states = [get_state(z, tpm) for z in range(number_of_states)]
    #import pdb; pdb.set_trace()
    # Compute results for this rule.
    tic = time()
    results = {
        'tpm': tpm,
        'connectivity_matrix': connectivity_matrix,
        'state': tuple(filter(lambda x: x is not None, [
            compute_phi_data(net, current) for current in current_states
        ]))
    }

    toc = time()
    elapsed = toc - tic
    #import pdb; pdb.set_trace()

    # Get the results in a form suitable for saving in a matfile.
    matlab_results = {
        'rule': rule,
        'tpm': tpm,
        'connectivity_matrix': connectivity_matrix,
        'state': [
            bigmip2dict(mip, t) for mip, t in results['state']
        ],
        'phi_partition_type': pyphi.config.PARTITION_TYPE,
        'phi_measure': pyphi.config.MEASURE,
        'pick_smallest_purview': pyphi.config.PICK_SMALLEST_PURVIEW,
        'sum_small_phi_as_big_phi': pyphi.config.USE_SMALL_PHI_DIFFERENCE_FOR_CONSTELLATION_DISTANCE
    }

    log.info('\n[CA] Total time elapsed: ' + str(elapsed))

    # Save the matlab results in a matfile for analysis with Matlab.
    matfile_filename = os.path.join(matlab_results_dir, rule_string)
    scipy.io.savemat(matfile_filename, matlab_results, do_compression=True)

    # Save the results in a pickled file for analysis with Python.
    # pickle_filename = os.path.join(python_results_dir, rule_string + '.pkl')
    # with open(pickle_filename, 'wb') as f:
    #     pickle.dump(results, f)


# Run everything if this file is being executed.
if __name__ == "__main__":
    tic = time()

    Parallel(n_jobs=(NUMBER_OF_CORES), verbose=PARALLEL_VERBOSITY)(
        delayed(evaluate_rule)(rule) for rule in rules_to_compute)

    toc = time()
    elapsed = toc-tic
    print("Finished in" + str(elapsed) + "\n")
