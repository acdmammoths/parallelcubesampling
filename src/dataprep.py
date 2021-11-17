import numpy as np
import math
from typing import Optional, Tuple, Union
from cubeutils import get_active_indices, get_active_strata


def prepare_inputs(
    data: np.ndarray,
    init_probs: np.ndarray,
    is_pop_size_fixed: bool = False,
    is_sample_size_fixed: bool = False,
    strata: Optional[np.ndarray] = None,
    is_sample_size_in_strata_fixed: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Prepares inputs by scaling the data, selecting active units,
    and prepending the required constraints.

    :param data: a matrix with dimensions population size x number of auxiliary variables
    :param init_probs: a vector of inclusion probabilities
    :param is_pop_size_fixed: whether the algorithm should fix the population size
    :param is_sample_size_fixed: whether the algorithm should fix the sample size
    :param strata: the strata that the balanced sample should respect
    :param is_sample_size_in_strata_fixed: whether the algorithm should fix sample size
    within each strata
    :returns: 3-tuple of (matrix of prepared data, vector of prepared inclusion probabilities,
    vector of prepared strata) if strata is not none. If strata is none, returns tuple of
    a matrix of prepared data and a vector of prepared inclusion probabilities
    """

    prepared_data = data.copy()

    # fixed pop size constraint needs to be added first
    # so that it is included as an aux var in stratified sample cube methods
    if is_pop_size_fixed:
        prepared_data = prepend_pop_size_constraint(prepared_data)

    if is_sample_size_fixed:
        prepared_data = prepend_sample_size_constraint(prepared_data, init_probs)

    prepared_data, prepared_init_probs = drop_inactive_units_and_scale(prepared_data, init_probs)

    if strata is not None:
        # Use natural numbers as strata labels.
        unique_strata = np.unique(strata)
        labels = np.arange(len(unique_strata))
        unique_strata_labels_dict = dict(zip(unique_strata, labels))
        strata = np.array([unique_strata_labels_dict[s] for s in strata])
        active_strata = get_active_strata(init_probs, strata)
        if is_sample_size_in_strata_fixed:
            prepared_data = prepend_strata_constraints(prepared_data, active_strata)
        return prepared_data.T, prepared_init_probs, active_strata
    else:
        return prepared_data.T, prepared_init_probs


def prepare_inputs_for_strat_fp_2(
    prepared_data: np.ndarray,
    flight_probs: np.ndarray,
    strata: np.ndarray,
    num_aux_vars: int,
    is_sample_size_in_strata_fixed: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for second flight phase in sample cubes with strata.

    :param prepared_data: a matrix of scaled, active, and transposed data
    with desired constraints prepended
    :param flight_probs: a vector of inclusion probabilities after the first flight phase
    :param strata: the strata that the balanced sample should respect
    :param num_aux_vars: the number of auxiliary variables including
    fixed population size constraints
    :param is_sample_size_in_strata_fixed: whether the algorithm should fix the sample size
    within each strata
    :returns: a tuple with a matrix of the active prepared data and a vector of
    the active inclusion probabilities from the first flight phase
    """

    # Remove column with inclusion probabilities and
    # keep original auxiliary variables and pop size contraint
    num_constraints = prepared_data.shape[0] - num_aux_vars
    prepared_data_new = prepared_data[num_constraints:, :]

    active_indices = get_active_indices(flight_probs)
    prepared_data_new = prepared_data_new[:, active_indices]
    active_flight_probs = flight_probs[active_indices]
    strata = strata[active_indices]

    if is_sample_size_in_strata_fixed:
        prepared_data_new = np.append(get_strata_constraints(strata).T, prepared_data_new, axis=0)

    return prepared_data_new, active_flight_probs


def drop_inactive_units_and_scale(
    data: np.ndarray, init_probs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selects active units and scales the data.

    :param data: a matrix with dimensions population size x number of auxiliary variables
    :param init_probs: a vector of inclusion probabilities
    :returns: a tuple with a matrix of active and scaled data and a vector of
    active inclusion probabilities
    """

    init_probs = init_probs.copy()
    active_indices = get_active_indices(init_probs)
    prepared_init_probs = init_probs[active_indices]
    prepared_data = data[active_indices, :] / prepared_init_probs.reshape(-1, 1)

    return (prepared_data, prepared_init_probs)


def prepend_pop_size_constraint(prepared_data: np.ndarray) -> np.ndarray:
    """
    Prepend constraint to prepared_data to fix the population size.

    :param prepared_data: a matrix of scaled, active, and transposed data with
    desired constraints prepended
    :returns: a matrix of the prepared data with columns to fix the population size prepended
    """

    prepared_data = np.append(
        np.repeat(1, prepared_data.shape[0]).reshape(-1, 1), prepared_data, axis=1
    )
    return prepared_data


def prepend_sample_size_constraint(
    prepared_data: np.ndarray, prepared_init_probs: np.ndarray
) -> np.ndarray:
    """
    Prepend constraint to prepared_data to fix the sample size

    :param prepared_data: a matrix of scaled, active, and transposed data with
    desired constraints prepended
    :param prepared_init_probs: a vector of active inclusion probabilities
    :returns: a matrix of the prepared data with a column to fix the sample size prepended
    """
    prepared_data = np.append(prepared_init_probs.reshape(-1, 1), prepared_data, axis=1)
    return prepared_data


def prepend_strata_constraints(prepared_data: np.ndarray, active_strata: np.ndarray) -> np.ndarray:
    """
    Prepend constraints to prepared_data to fix the sample size within each strata

    :param prepared_data: a matrix of scaled, active, and transposed data with
    desired constraints prepended
    :param active_strata: a vector of strata indices that have active inclusion probabilities
    :returns: a matrix of the prepared data with columns to fix the sample size within each strata
    prepended
    """
    strata_constraints = get_strata_constraints(active_strata)
    prepared_data = np.append(strata_constraints, prepared_data, axis=1)
    return prepared_data


def get_strata_constraints(strata: np.ndarray) -> np.ndarray:
    """
    Create constraints to generate a sample that respects each strata.

    :param strata: the strata that the balanced sample should respect
    :returns: a matrix of strata constraints where each row is a member of the population and
    the ith column is a 1 if the belongs to ith strata and 0 in every other column.
    """

    num_strata = len(np.unique(strata))
    pop_size = len(strata)
    strata_constraints = np.zeros((pop_size, num_strata))
    for i in range(pop_size):
        strata_constraints[i, strata[i]] = 1
    return strata_constraints


def create_inc_probs_for_strata(pop_size: int, strata: np.ndarray) -> np.ndarray:
    """
    Given strata indices for every member of the population,
    create init_probs such that the sum of the
    inclusion probabilities in each strata is an integer.

    :param pop_size: the population size
    :param strata: the strata that the balanced sample should respect
    :returns: a vector of inclusion probabilities
    """

    rng = np.random.default_rng()
    init_probs = rng.random(pop_size)
    num_strata = max(strata) + 1

    for i in range(num_strata):
        stratum = init_probs[strata == i]
        diff = math.ceil(sum(stratum)) - sum(stratum)

        if stratum[-1] + diff < 1:
            stratum[-1] = stratum[-1] + diff
        else:
            stratum[-1] = stratum[-1] - (1 - diff)

        init_probs[strata == i] = stratum
    return init_probs
