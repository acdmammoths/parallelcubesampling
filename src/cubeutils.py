import numpy as np
from typing import Tuple
import scipy.linalg


tol = 1e-12


def flight_phase(
    prepared_data: np.ndarray, prepared_probs: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Attempt to find an exactly balanced sample with an optimized algorithm that
    satisfies the balancing equations given by the auxiliary variables
    in prepared_data starting at the inclusion probabilities prepared_probs.

    Paper Reference: A Fast Algorithm for Balanced Sampling (Chauvet and Tille 2006)

    :param prepared_data: a matrix of scaled, active, and transposed data
    :param prepared_probs: a vector of active inclusion probabilities
    :param rng: a random number generator
    :returns: a vector of inclusion probabilties
    """

    # Prepare the initial state for the fast flight phase.
    # We skip 1a,b because the inputs already arrive in the right form
    # 1c
    subset_size = prepared_data.shape[0] + 1
    subset_probs = prepared_probs[:subset_size]
    # 1d
    subset_columns = np.arange(subset_size)
    # 1e
    subset_prepared_data = prepared_data[:, :subset_size]
    # 1f
    next_col_index = subset_size
    # end 1f
    prepared_data_copy = prepared_data.copy()
    prepared_probs_copy = prepared_probs.copy()

    num_aux_vars, pop_size = prepared_data_copy.shape
    if pop_size > subset_size:
        while next_col_index < pop_size:
            subset_probs = flight_step(subset_prepared_data, subset_probs, rng)
            # 2d
            i = 0
            while i in range(subset_size) and next_col_index < pop_size:
                if subset_probs[i] < tol or subset_probs[i] > (1 - tol):
                    prepared_probs_copy[subset_columns[i]] = subset_probs[i]
                    subset_columns[i] = next_col_index
                    subset_probs[i] = prepared_probs_copy[next_col_index]
                    subset_prepared_data[:, i] = prepared_data_copy[:, next_col_index]
                    next_col_index = next_col_index + 1
                i = i + 1
            # end 2d
        if get_num_not_selected(prepared_probs_copy) == num_aux_vars + 1:
            subset_probs = flight_step(subset_prepared_data, subset_probs, rng)
        # 3a
        prepared_probs_copy[subset_columns] = subset_probs
        # end 3a
        active_indices = get_active_indices(prepared_probs)
        prepared_data[:, active_indices] = prepared_data_copy
        prepared_probs[active_indices] = prepared_probs_copy
    return prepared_probs


def flight_step(
    subset_prepared_data: np.ndarray, subset_probs: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Randomly assigns one element inclusion probability 0 or 1.
    This is the step that decides whether an element is in or not in
    the balanced sample. Two important steps occur inside, finding a
    vector in the kernel of the support, and finding the step sizes in
    opposite directions, which would all -- after the update -- round off
    at least one of the inclusion probabilities and keep the vector in
    the intersection of the hypercube and the support.

    Paper Reference: A Fast Algorithm for Balanced Sampling (Chauvet and Tille 2006)

    :param subset_prepared_data: a subset of the prepared data
    :param subset_probs: a vector of inclusion probabilities that correspond to the columns
    in subset_prepared_data
    :param rng: a random number generator
    :returns: a vector of updated inclusion probabilities
    """

    # 2a
    vector_in_kernel = get_vector_in_kernel(subset_prepared_data)
    # 2b
    lambda_1, lambda_2 = get_step_sizes(vector_in_kernel, subset_probs)
    # 2c
    if rng.random() <= (lambda_2 / (lambda_1 + lambda_2)):
        subset_probs = subset_probs + lambda_1 * vector_in_kernel
    else:
        subset_probs = subset_probs - lambda_2 * vector_in_kernel
    # end 2c
    subset_probs = round_off_already_selected(subset_probs)
    assert np.isfinite(subset_probs).all()
    return subset_probs


def get_step_sizes(
    vector_in_kernel: np.ndarray, subset_probs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the largest step sizes in both directions such that we remain in the hypercube
    (Get the max (> 0) lambda_1, lambda_2 s.t.
    0 <= subset_probs + lambda_1 * vector_in_kernel <= 1 and
    0 <= subset_probs - lambda_2 * vector_in_kernel <= 1.)

    Code Reference: https://cran.r-project.org/web/packages/sampling/index.html
    fastflightphase.R (lines 10-28)

    :param vector_in_kernel: a vector in the kernel of subset_prepared_data
    :param subset_probs: a vector of inclusion probabilities that correspond to the columns
    in subset_prepared_data
    :returns: a tuple of the maximum step sizes in both directions
    """

    not_zero = np.where(np.abs(vector_in_kernel) > tol)
    buff1 = (1 - subset_probs[not_zero]) / vector_in_kernel[not_zero]
    buff2 = -subset_probs[not_zero] / vector_in_kernel[not_zero]
    lambda_1 = np.min(np.concatenate((buff1[buff1 > 0], buff2[buff2 > 0])))
    buff1 = subset_probs[not_zero] / vector_in_kernel[not_zero]
    buff2 = (subset_probs[not_zero] - 1) / vector_in_kernel[not_zero]
    lambda_2 = np.min(np.concatenate((buff1[buff1 > 0], buff2[buff2 > 0])))
    return lambda_1, lambda_2


def landing_phase(
    prepared_data: np.ndarray, flight_probs: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Select the final sample by relaxing the balancing constraints via suppression of variables.

    Paper Reference: Efficient balanced sampling: The cube method (Deville and Tille 2004)

    :param prepared_data: a matrix of scaled, active, and transposed data
    :param flight_probs: a vector of inclusion probabilities after all flight phases
    :param rng: a random number generator
    :returns: a vector of 0s and 1s in place of the active units before the first flight phase
    """

    subset_prepared_data = prepared_data
    removed = 0
    while get_num_not_selected(flight_probs) > 0 and subset_prepared_data.shape[0] > 1:
        active_indices = get_active_indices(flight_probs)
        # Remove one constraint and keep only the active units
        subset_prepared_data = prepared_data[: removed - 1, active_indices]
        removed -= 1
        active_flight_probs = flight_probs[active_indices]
        sample = flight_phase(subset_prepared_data, active_flight_probs, rng)
        flight_probs[active_indices] = sample

    return clean_flight_probs(flight_probs, rng)


def get_fake_strata(pop_size: int, num_strata: int, rng: np.random.Generator) -> np.ndarray:
    """
    Create fake strata.

    :param pop_size: the population size
    :param num_strata: the number of fake strata to create
    :param rng: a random number generator
    :returns: randomly generated strata as a vector of strata indices
    between 0 and num_strata - 1 inclusive.
    """

    fake_strata = np.arange(pop_size) % num_strata
    rng.shuffle(fake_strata)
    return fake_strata


def get_vector_in_kernel(subset_prepared_data: np.ndarray) -> np.ndarray:
    """
    Get a vector in the kernel of subset_prepared_data by SVD.

    :param subset_prepared_data: a subset of the prepared data
    :returns: a vector in the kernel of subset_prepared_data
    """

    kernel_ortho_basis = scipy.linalg.null_space(subset_prepared_data)
    vector_in_kernel = kernel_ortho_basis[:, 0]  # select first or only vector in basis
    return vector_in_kernel


def round_off_already_selected(subset_probs: np.ndarray) -> np.ndarray:
    """
    Get rid of small rounding errors in selected elements of subset_probs.

    :param subset_probs: a subset of the inclusion probabilities
    :returns: a subset of the inclusion probabilities without rounding errors
    """
    subset_probs = np.abs(subset_probs)
    for i in range(len(subset_probs)):
        if abs(subset_probs[i] - 0) < tol:
            subset_probs[i] = 0
        if abs(subset_probs[i] - 1) < tol:
            subset_probs[i] = 1
    return subset_probs


def get_num_not_selected(probs: np.ndarray) -> int:
    """
    Get number of elements not selected (i.e., not approximately 0 or 1) in probs.

    :param probs: a vector of inclusion probabilities
    :returns: the number of elements not selected in the inclusion probabilities
    """
    num_not_selected = 0
    for prob in probs:
        if not (prob < tol or prob > (1 - tol)):
            num_not_selected += 1
    return num_not_selected


def get_active_indices(probs: np.ndarray) -> np.ndarray:
    """
    Get indices of units that haven't been selected (i.e., not 0 or 1).

    :param probs: a vector of inclusion probabilities
    :returns: a vector of indices corresponding to probabilities that aren't 0 or 1
    """

    active_indices = np.asarray(
        np.logical_and(np.abs(0 - probs) > tol, np.abs(1 - probs) > tol)
    ).nonzero()[0]
    return active_indices


def get_active_strata(init_probs: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """
    Select strata that such that the corresponding inclusion probabilities aren't 0 or 1.

    :param init_probs: a vector of inclusion probabilities
    :param strata: the strata that the balanced sample should respect
    :returns: a subset of the strata that correspond to inclusion probabilities that aren't 0 or 1
    """

    active_indices = get_active_indices(init_probs)
    active_strata = strata[active_indices]
    return active_strata


def clean_flight_probs(flight_probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Round off probabilities in flight_probs to 0 or 1 with random bias of the current probability

    :param flight_probs: a vector of inclusion probabilities after the landing phase
    :param rng: a random number generator
    :returns: a vector of inclusion probabilities that have been rounded off
    """

    for i in range(len(flight_probs)):
        if flight_probs[i] - 0 > tol and flight_probs[i] < 1 - tol:
            flight_probs[i] = 1 if rng.random() < flight_probs[i] else 0
    return flight_probs


def prepare_output(init_probs: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """
    Prepare output by replacing initial inclusion probabilities
    with final indicator variables for the sample.

    :param init_probs: a vector of inclusion probabilities
    :param sample: a vector of 0s and 1s from the landing phase
    :returns: a vector of 0s and 1s for selected sample
    """

    init_probs = init_probs.copy()
    active_indices = get_active_indices(init_probs)
    init_probs[active_indices] = sample
    return init_probs
