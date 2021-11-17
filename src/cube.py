#! /usr/bin/env python3
import time
import numpy as np
import multiprocessing as mp
from typing import Optional, Union, Tuple
import dataprep as dp
from cubeutils import (
    get_fake_strata,
    prepare_output,
    get_active_indices,
    flight_phase,
    landing_phase,
)


def sample_cube(
    data: np.ndarray,
    init_probs: np.ndarray,
    is_pop_size_fixed: bool = False,
    is_sample_size_fixed: bool = False,
    seed: Optional[int] = None,
    use_internal_timing: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Use the cube method to select a balanced sample.
    The method selects a sample such that the Horvitz-Thompson
    estimators for all the auxiliary variables are satisfied exactly
    or approximately.

    Paper Reference: A Fast Algorithm for Balanced Sampling (Chauvet and Tille 2006)

    :param data: a matrix with dimensions population size x number of auxiliary variables
    :param init_probs: a vector of inclusion probabilities
    :param is_pop_size_fixed: whether the algorithm should fix the population size
    :param is_sample_size_fixed: whether the algorithm should fix the sample size
    :param seed: random number generator seed for replication
    :returns: a vector of 0s and 1s for selected sample
    """

    start = time.time()
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    prepared_data, prepared_init_probs = dp.prepare_inputs(
        data, init_probs, is_pop_size_fixed, is_sample_size_fixed
    )

    time_part1 = time.time() - start
    flight_probs = flight_phase(prepared_data, prepared_init_probs, rng)
    sample = landing_phase(prepared_data, flight_probs, rng)
    sample = prepare_output(init_probs, sample)
    time_part2 = (time.time() - start) - time_part1

    if use_internal_timing:
        return (sample, (time_part1, time_part2))
    else:
        return sample


def sample_cube_parallel(
    data: np.ndarray,
    init_probs: np.ndarray,
    num_proc: int,
    num_strata: int = 0,
    is_pop_size_fixed: bool = False,
    is_sample_size_fixed: bool = False,
    seed: Optional[int] = None,
    use_internal_timing: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[int, int, int]]]:
    """
    Use a parallel cube method to select a balanced sample.
    The method selects a sample such that the Horvitz-Thompson
    estimators for all the auxiliary variables are satisfied exactly
    or approximately.

    Paper Reference: Stratified Balanced Sampling (Chauvet 2009), Algorithm 1

    :param data: a matrix with dimensions population size x number of auxiliary variables
    :param init_probs: a vector of inclusion probabilities
    :param num_proc: number of processors to use
    :param num_strata: number of fake strata to create; if unspecified, num_strata = num_proc
    :param is_pop_size_fixed: whether the algorithm should fix the population size
    :param is_sample_size_fixed: whether the algorithm should fix the sample size
    :param seed: random number generator seed for replication
    :param use_internal_timing: whether to return runtimes of the algorithm
    :returns: a vector of 0s and 1s for selected sample if use_internal_timing=False.
    If use_internal_timing=True, returns a tuple containing the vector of 0s and 1s
    and a 3-tuple of runtimes.
    """

    start = time.time()
    if num_proc <= 0:
        raise ValueError(f"sample_cube_parallel: non-positive value for num_proc: {num_proc}")
    if num_strata <= 0:
        num_strata = num_proc

    if seed is not None:
        seed_sequence = np.random.SeedSequence(seed)
        seq_rng = np.random.default_rng(seed)
    else:
        seed_sequence = np.random.SeedSequence()
        seq_rng = np.random.default_rng()

    prepared_data, prepared_init_probs = dp.prepare_inputs(
        data, init_probs, is_pop_size_fixed, is_sample_size_fixed
    )

    num_active_units = len(prepared_init_probs)
    fake_strata = get_fake_strata(num_active_units, num_strata, seq_rng)
    par_rngs = [np.random.default_rng(s) for s in seed_sequence.spawn(num_strata)]

    # Step 1
    flight_probs = np.zeros(num_active_units)
    time_part1 = time.time() - start
    with mp.Pool(processes=num_proc) as pool:
        stratified_flight_probs = pool.starmap(
            flight_phase,
            [
                (
                    prepared_data[:, fake_strata == h],
                    prepared_init_probs[fake_strata == h],
                    par_rngs[h],
                )
                for h in np.arange(num_strata)
            ],
        )
    for h in np.arange(num_strata):
        flight_probs[fake_strata == h] = stratified_flight_probs[h]

    time_part2 = (time.time() - start) - time_part1

    # Step 2,3
    active_indices = get_active_indices(flight_probs)
    prepared_flight_probs = flight_probs[active_indices]
    prepared_data = prepared_data[:, active_indices]

    flight_probs_round_2 = flight_phase(prepared_data, prepared_flight_probs, seq_rng)
    sample = landing_phase(prepared_data, flight_probs_round_2, seq_rng)
    sample = prepare_output(flight_probs, sample)
    sample = prepare_output(init_probs, sample)

    time_part3 = (time.time() - start) - time_part2

    if use_internal_timing:
        return (sample, (time_part1, time_part2, time_part3))
    else:
        return sample


def stratified_sample_cube(
    data: np.ndarray,
    init_probs: np.ndarray,
    strata: np.ndarray,
    is_pop_size_fixed: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Use the stratified cube method to select a stratified balanced sample.

    Paper Reference: Stratified Balanced Sampling (Chauvet 2009)

    :param data: a matrix with dimensions population size x number of auxiliary variables
    :param init_probs: a vector of inclusion probabilities
    :param strata: the strata that the balanced sample should respect
    :param is_pop_size_fixed: whether the algorithm should fix the population size
    :param seed: random number generator seed for replication
    :returns: a vector of 0s and 1s for selected sample
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # make pop size constraint an additional aux var
    num_aux_vars = data.shape[1] + is_pop_size_fixed

    prepared_data, prepared_init_probs, active_strata = dp.prepare_inputs(
        data, init_probs, is_pop_size_fixed, True, strata
    )  # is_sample_size_fixed is always True

    # Adding 1 because strata ranges from 0 to num_strata - 1
    num_active_strata = max(active_strata) + 1
    num_active_units = len(prepared_init_probs)
    flight_probs = np.zeros(num_active_units)

    for h in np.arange(num_active_strata):
        strata_prepared_data, strata_init_probs = (
            prepared_data[:, active_strata == h],
            prepared_init_probs[active_strata == h],
        )
        flight_probs[active_strata == h] = flight_phase(
            strata_prepared_data, strata_init_probs, rng
        )

    prepared_data, prepared_flight_probs = dp.prepare_inputs_for_strat_fp_2(
        prepared_data,
        flight_probs,
        strata,
        num_aux_vars,
        is_sample_size_in_strata_fixed=True,
    )

    flight_probs_round_2 = flight_phase(prepared_data, prepared_flight_probs, rng)
    sample = landing_phase(prepared_data, flight_probs_round_2, rng)
    sample = prepare_output(flight_probs, sample)
    sample = prepare_output(init_probs, sample)

    return sample


def parallel_stratified_sample_cube(
    data: np.ndarray,
    init_probs: np.ndarray,
    strata: np.ndarray,
    num_proc: int,
    is_pop_size_fixed: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Use the parallel stratified cube method to select a stratified balanced sample.

    Paper Reference: Stratified Balanced Sampling (Chauvet 2009)

    :param data: a matrix with dimensions population size x number of auxiliary variables
    :param init_probs: a vector of inclusion probabilities
    :param strata: the strata that the balanced sample should respect
    :param num_proc: number of processors to use
    :param is_pop_size_fixed: whether the algorithm should fix the population size
    :param seed: random number generator seed for replication
    :returns: a vector of 0s and 1s for selected sample
    """

    if seed is not None:
        seed_sequence = np.random.SeedSequence(seed)
        seq_rng = np.random.default_rng(seed)
    else:
        seed_sequence = np.random.SeedSequence()
        seq_rng = np.random.default_rng()

    # make pop size constraint an additional aux var
    num_aux_vars = data.shape[1] + is_pop_size_fixed

    prepared_data, prepared_init_probs, active_strata = dp.prepare_inputs(
        data, init_probs, is_pop_size_fixed, True, strata
    )  # is_sample_size_fixed is always True

    # Adding 1 because strata ranges from 0 to num_strata - 1
    num_active_strata = max(active_strata) + 1
    num_active_units = len(prepared_init_probs)
    par_rngs = [np.random.default_rng(s) for s in seed_sequence.spawn(num_active_strata)]

    flight_probs = np.zeros(num_active_units)
    with mp.Pool(processes=num_proc) as pool:
        stratified_flight_probs = pool.starmap(
            flight_phase,
            [
                (
                    prepared_data[:, active_strata == h],
                    prepared_init_probs[active_strata == h],
                    par_rngs[h],
                )
                for h in np.arange(num_active_strata)
            ],
        )
    for h in np.arange(num_active_strata):
        flight_probs[active_strata == h] = stratified_flight_probs[h]

    prepared_data, prepared_flight_probs = dp.prepare_inputs_for_strat_fp_2(
        prepared_data,
        flight_probs,
        active_strata,
        num_aux_vars,
        is_sample_size_in_strata_fixed=True,
    )

    flight_probs_round_2 = flight_phase(prepared_data, prepared_flight_probs, seq_rng)
    sample = landing_phase(prepared_data, flight_probs_round_2, seq_rng)
    sample = prepare_output(flight_probs, sample)
    sample = prepare_output(init_probs, sample)

    return sample
