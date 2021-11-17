#! /usr/bin/env python3
import unittest

import numpy as np

from cubeutils import (
    get_fake_strata,
    get_vector_in_kernel,
    round_off_already_selected,
    get_num_not_selected,
    flight_phase,
)
from cube import (
    sample_cube,
    stratified_sample_cube,
    parallel_stratified_sample_cube,
    sample_cube_parallel,
)
from dataprep import (
    get_strata_constraints,
    prepare_inputs,
    create_inc_probs_for_strata,
)


class TestUtilities(unittest.TestCase):
    def test_get_strata_constraints(self):
        strata = [1, 0, 2, 0, 1]
        np.testing.assert_array_equal(
            get_strata_constraints(strata),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        )

    def test_prepare_inputs(self):
        data = np.array([[0, 3], [1, 2], [3, 4], [-1, 3]])
        init_probs = np.array([0.4, 0.5, 0.2, 0.5])
        prepared_data, _ = prepare_inputs(data, init_probs)
        np.testing.assert_array_equal(
            prepared_data,
            np.array([[0, 2, 15, -2], [7.5, 4, 20, 6]]),
            err_msg="prepared-data is not data/init_probs transpose.",
        )
        prepared_data, _, _ = prepare_inputs(
            data,
            init_probs,
            is_pop_size_fixed=True,
            is_sample_size_fixed=True,
            strata=np.array([4, 3, 3, 4]),
            is_sample_size_in_strata_fixed=True,
        )
        np.testing.assert_array_equal(
            prepared_data,
            np.array(
                [
                    [0, 1, 1, 0],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1],
                    [2.5, 2, 5, 2],
                    [0, 2, 15, -2],
                    [7.5, 4, 20, 6],
                ]
            ),
        )

    def test_get_vector_in_kernel(self):
        prepared_data = np.array(
            [
                [0.576, 0.712, 0.013, 0.752, 0.335, 0.656],
                [0.975, 0.842, 0.311, 0.53, 0.354, 0.769],
                [0.762, 0.91, 0.953, 0.939, 0.488, 0.489],
                [0.395, 0.825, 0.485, 0.803, 0.443, 0.961],
                [0.579, 0.707, 0.54, 0.471, 0.705, 0.166],
            ]
        )
        vector_in_kernel = get_vector_in_kernel(prepared_data)
        np.testing.assert_allclose(
            np.dot(prepared_data, vector_in_kernel),
            np.zeros(prepared_data.shape[0]),
            atol=1e-12,
            err_msg="Not in kernel.",
        )

    def test_round_off_already_selected(self):
        with_roundoff_errors = np.array([0, 1e-14, -1e-17, 0.56, 0.999999999999999, 1])
        rounded_off = round_off_already_selected(with_roundoff_errors)
        np.testing.assert_array_equal(
            rounded_off,
            np.array([0, 0, 0, 0.56, 1, 1]),
            err_msg="Didn't deal with roundoff.",
        )

    def test_get_num_not_selected(self):
        arr = np.array([0.5, 0.3, 0.2, 1, 0, 0.5, 0])
        self.assertEqual(get_num_not_selected(arr), 4, msg="Not counting work left properly.")

    def test_inc_probs_in_strata(self):
        """
        Test to ensure the sum of inclusion probabilities in each strata in an integer.
        """
        pop_size = 1000
        num_strata = 5
        strata = get_fake_strata(pop_size, num_strata, np.random.default_rng())
        init_probs = create_inc_probs_for_strata(pop_size, strata)
        sum_by_strata = [sum(init_probs[strata == i]) for i in range(num_strata)]
        self.assertTrue(all([x.is_integer() for x in sum_by_strata]))

    @unittest.skip("not implemented")
    def test_get_step_sizes(self):
        pass

    @unittest.skip("not implemented")
    def test_get_active_indices(self):
        pass


class TestCubeSampling(unittest.TestCase):
    def setUp(self):
        pop_size, num_aux_vars = 103, 29
        rng = np.random.default_rng()
        self.data = rng.random((pop_size, num_aux_vars))
        self.init_probs = rng.random(pop_size)

    def test_sample_with_fixed_sample_size(self):
        sample_size = np.round(np.sum(self.init_probs))
        sample = sample_cube(self.data, self.init_probs, is_sample_size_fixed=True)
        self.assertAlmostEqual(
            np.sum(sample),
            sample_size,
            delta=1,
            msg="Balancing with probs->fixed sample size.",
        )

    def test_sample_proper_sample_output(self):
        sample = sample_cube(self.data, self.init_probs)
        num_non_integers = get_num_not_selected(sample)
        self.assertEqual(num_non_integers, 0, msg="Sample selected must be either 0s or 1s.")

    def test_flight_results(self):
        rng = np.random.default_rng()
        prepared_data, prepared_init_probs = prepare_inputs(self.data, self.init_probs)
        flight_probs = flight_phase(prepared_data, prepared_init_probs, rng)
        self.assertLessEqual(
            get_num_not_selected(flight_probs),
            self.data.shape[1],
            msg="Expected <=p unrounded units post-flight.",
        )
        is_betweenish_zero_or_one = np.where(
            np.logical_or(flight_probs > -1e-12, flight_probs < 1 + 1e-12), True, False
        )
        self.assertTrue(
            is_betweenish_zero_or_one.all(),
            msg="Probs post-flight should be betweenish 0 and 1.",
        )

    @unittest.skip("not implemented")
    def test_total_num_iterations(self):
        pass

    @unittest.skip("not implemented")
    def test_malicious_input(self):
        pass

    def test_handles_already_chosen_units(self):
        indices = [0, 2, 5, 7, 13, 21]
        already_chosen = [0, 1, 1, 0, 0, 1]
        self.init_probs[indices] = already_chosen
        sample = sample_cube(self.data, self.init_probs)
        np.testing.assert_array_equal(
            sample[indices], already_chosen, err_msg="Chosen units changed."
        )

    def test_respect_inclusion_probs(self):
        pop_size = self.data.shape[0]
        empirical = np.zeros(pop_size)
        num_sims = 1000
        for _ in range(num_sims):
            sample = sample_cube(self.data, self.init_probs)
            empirical += sample / num_sims
        np.testing.assert_array_almost_equal(
            empirical, self.init_probs, decimal=1, err_msg="Not respecting probs."
        )

    def test_replication_with_rand_seed(self):
        seed_sequence = np.random.SeedSequence()
        child_seeds = seed_sequence.spawn(5)
        for i in range(len(child_seeds)):
            base_sample = sample_cube(self.data, self.init_probs, seed=child_seeds[i].entropy)
            for _ in range(5):
                sample = sample_cube(self.data, self.init_probs, seed=child_seeds[i].entropy)
                np.testing.assert_array_equal(
                    base_sample, sample, err_msg="Samples are not replicated with random seed."
                )

    def test_with_inactive_init_probs(self):
        init_probs = self.init_probs.copy()
        init_probs[0] = 1
        failed = False
        try:
            sample = sample_cube(self.data, init_probs)
        except Exception:
            failed = True
        self.assertFalse(failed, "sample_cube() failed to complete")
        self.assertTrue(len(sample) == len(init_probs))


class TestStratifiedCubeSampling(unittest.TestCase):
    def setUp(self):
        pop_size, self.num_aux_vars = 103, 29
        rng = np.random.default_rng()
        self.data = rng.random((pop_size, self.num_aux_vars))
        self.strata = get_fake_strata(pop_size, 7, rng)
        self.init_probs = create_inc_probs_for_strata(pop_size, self.strata)

    def test_fixed_sample_size_each_strata(self):
        sample = stratified_sample_cube(self.data, self.init_probs, self.strata)
        expected_sizes = []
        actual_sizes = []
        for h in np.arange(max(self.strata) + 1):
            expected_size = np.round(np.sum(self.init_probs[self.strata == h]))
            expected_sizes.append(expected_size)
            actual_size = np.sum(sample[self.strata == h])
            actual_sizes.append(actual_size)
        np.testing.assert_allclose(
            expected_sizes, actual_sizes, atol=1, err_msg="Not fixed size in strata."
        )

    def test_respect_inclusion_probs(self):
        pop_size = self.data.shape[0]
        empirical = np.zeros(pop_size)
        num_sims = 1000
        for _ in range(num_sims):
            sample = stratified_sample_cube(self.data, self.init_probs, self.strata)
            empirical += sample / num_sims
        np.testing.assert_array_almost_equal(
            empirical, self.init_probs, decimal=1, err_msg="Not respecting probs."
        )

    def test_replication_with_rand_seed(self):
        seed_sequence = np.random.SeedSequence()
        child_seeds = seed_sequence.spawn(5)
        for i in range(len(child_seeds)):
            base_sample = stratified_sample_cube(
                self.data, self.init_probs, self.strata, seed=child_seeds[i].entropy
            )
            for _ in range(5):
                sample = stratified_sample_cube(
                    self.data, self.init_probs, self.strata, seed=child_seeds[i].entropy
                )
                np.testing.assert_array_equal(
                    base_sample, sample, err_msg="Samples are not replicated with random seed."
                )

    def test_fixed_sample_size_each_strata_parallel(self):
        sample = parallel_stratified_sample_cube(
            self.data,
            self.init_probs,
            self.strata,
            num_proc=4,
        )
        expected_sizes = []
        actual_sizes = []
        for h in np.arange(max(self.strata) + 1):
            expected_size = np.round(np.sum(self.init_probs[self.strata == h]))
            expected_sizes.append(expected_size)
            actual_size = np.sum(sample[self.strata == h])
            actual_sizes.append(actual_size)
        np.testing.assert_allclose(
            expected_sizes, actual_sizes, atol=1, err_msg="Not fixed size in strata."
        )

    def test_respect_inclusion_probs_parallel(self):
        pop_size = self.data.shape[0]
        empirical = np.zeros(pop_size)
        num_sims = 1000
        for _ in range(num_sims):
            sample = parallel_stratified_sample_cube(
                self.data,
                self.init_probs,
                self.strata,
                num_proc=4,
            )
            empirical += sample / num_sims
        np.testing.assert_array_almost_equal(
            empirical, self.init_probs, decimal=1, err_msg="Not respecting probs."
        )

    def test_replication_with_rand_seed_parallel(self):
        seed_sequence = np.random.SeedSequence()
        child_seeds = seed_sequence.spawn(5)
        for i in range(len(child_seeds)):
            base_sample = parallel_stratified_sample_cube(
                self.data, self.init_probs, self.strata, num_proc=4, seed=child_seeds[i].entropy
            )
            for _ in range(5):
                sample = parallel_stratified_sample_cube(
                    self.data,
                    self.init_probs,
                    self.strata,
                    num_proc=4,
                    seed=child_seeds[i].entropy,
                )
                np.testing.assert_array_equal(
                    base_sample, sample, err_msg="Samples are not replicated with random seed."
                )

    def test_with_inactive_init_probs(self):
        init_probs = self.init_probs.copy()
        init_probs[0] = 0
        failed = False
        try:
            sample = stratified_sample_cube(self.data, init_probs, self.strata)
        except Exception:
            failed = True
        self.assertFalse(failed, "sample_cube() failed to complete")
        self.assertTrue(len(sample) == len(init_probs))

    def test_with_inactive_init_probs_parallel(self):
        init_probs = self.init_probs.copy()
        init_probs[0] = 1
        failed = False
        try:
            sample = parallel_stratified_sample_cube(self.data, init_probs, self.strata, num_proc=4)
        except Exception:
            failed = True
        self.assertFalse(failed, "sample_cube() failed to complete")
        self.assertTrue(len(sample) == len(init_probs))


class TestParallelCubeSampling(unittest.TestCase):
    def setUp(self):
        pop_size, self.num_aux_vars = 103, 29
        rng = np.random.default_rng()
        self.data = rng.random((pop_size, self.num_aux_vars))
        self.init_probs = rng.random(pop_size)

    def test_sample_with_fixed_sample_size(self):
        sample_size = np.round(np.sum(self.init_probs))
        sample = sample_cube_parallel(
            self.data, self.init_probs, num_proc=4, is_sample_size_fixed=True
        )
        self.assertAlmostEqual(
            np.sum(sample),
            sample_size,
            delta=1,
            msg="Balancing with probs->fixed sample size.",
        )

    def test_respect_inclusion_probs_parallel(self):
        pop_size = self.data.shape[0]
        empirical = np.zeros(pop_size)
        num_sims = 1000
        for _ in range(num_sims):
            sample = sample_cube_parallel(
                self.data,
                self.init_probs,
                num_proc=4,
            )
            empirical += sample / num_sims
        np.testing.assert_array_almost_equal(
            empirical, self.init_probs, decimal=1, err_msg="Not respecting probs."
        )

    def test_replication_with_rand_seed(self):
        seed_sequence = np.random.SeedSequence()
        child_seeds = seed_sequence.spawn(5)
        for i in range(len(child_seeds)):
            base_sample = sample_cube_parallel(
                self.data, self.init_probs, num_proc=4, seed=child_seeds[i].entropy
            )
            for _ in range(5):
                sample = sample_cube_parallel(
                    self.data, self.init_probs, num_proc=4, seed=child_seeds[i].entropy
                )
                np.testing.assert_array_equal(
                    base_sample, sample, err_msg="Samples are not replicated with random seed."
                )

    def test_with_inactive_init_probs(self):
        init_probs = self.init_probs.copy()
        init_probs[0] = 1
        failed = False
        try:
            sample = sample_cube_parallel(self.data, init_probs, num_proc=4)
        except Exception:
            failed = True
        self.assertFalse(failed, "sample_cube() failed to complete")
        self.assertTrue(len(sample) == len(init_probs))


if __name__ == "__main__":
    unittest.main()
