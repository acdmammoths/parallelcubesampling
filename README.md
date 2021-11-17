# parallelcubesampling

To use our various algorithms, do the following:

## Creating the virtual environment

In order to create the virtual environment, install the dependencies, and activate it, use the following commands:

```
python3 -m venv venv 
python3 -m pip install -r cube/requirements.txt
source venv/bin/activate
```

The virtual environment must be activated in order to run any of the algorithms.


## 1. Example for running `sample_cube()`:

```
import numpy as np
import cube

if __name__ == "__main__":
  pop_size, num_aux_var = 100, 10
  data = np.random.random((pop_size, num_aux_var))
  init_probs = np.random.random(pop_size)

  sample = cube.sample_cube(data, init_probs)
```

`sample_cube()` also has the following optional arguments:
- `is_pop_size_fixed`: `True` to include an auxiliary variable that fixes the population size when sampling
- `is_sample_size_fixed`: `True` to include an auxiliary variable that fixes the sample size
- `seed`: any integer for replicating samples
- `use_internal_timing`: `True` to return timings of various stages of the algorithm in addition to the sample

## 2. Example for running `sample_cube_parallel()`:

```
import numpy as np
import cube

if __name__ == "__main__":
  pop_size, num_aux_var = 100, 10
  num_proc = 4
  data = np.random.random((pop_size, num_aux_var))
  init_probs = np.random.random(pop_size)

  sample = cube.sample_cube_parallel(data, init_probs, num_proc)
```

`sample_cube_parallel()` also has the following optional arguments:
- `num_strata`: Any integer to specify the number of strata - has a default value of `num_proc`
- `is_pop_size_fixed`: `True` to include an auxiliary variable that fixes the population size when sampling
- `is_sample_size_fixed`: `True` to include an auxiliary variable that fixes the sample size
- `seed`: any integer for replicating samples
- `use_internal_timing`: `True` to return timings of various stages of the algorithm in addition to the sample

## 3. Example for running `stratified_sample_cube()`:

```
import numpy as np
import cube

if __name__ == "__main__":
  pop_size, num_aux_var = 100, 10
  num_strata = 7
  data = np.random.random((pop_size, num_aux_var))
  init_probs = np.random.random(pop_size)
  strata = np.array([i % num_strata for i in range(pop_size)])

  sample = cube.stratified_sample_cube(data, init_probs, strata)
```

`stratified_sample_cube()` also has the following optional arguments:
- `is_pop_size_fixed`: `True` to include an auxiliary variable that fixes the population size when sampling
- `seed`: any integer for replicating samples

## 4. Example for running `parallel_stratified_sample_cube()`:

```
import numpy as np
import cube

if __name__ == "__main__":
  pop_size, num_aux_var = 100, 10
  num_strata = 7
  num_proc = 4
  data = np.random.random((pop_size, num_aux_var))
  init_probs = np.random.random(pop_size)
  strata = np.array([i % num_strata for i in range(pop_size)])

  sample = cube.parallel_stratified_sample_cube(data, init_probs, strata, num_proc)
```

`parallel_stratified_sample_cube()` also has the following optional arguments:
- `is_pop_size_fixed`: `True` to include an auxiliary variable that fixes the population size when sampling
- `seed`: any integer for replicating samples

## 5. Run Tests

Our test suite can run using the following command (the virtual environment must be activated):

```
python3 src/test_cube.py
```
