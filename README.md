# Easy benchmark framework for cupy
## Features
+ Reports both, cpu and gpu time.
+ Produces plots of the execution time, speedup or custom metrics.
+ Saves the results in csv files.
+ Allows automatic performance comparison with numpy or numpy API compat
libraries.
+ TODO: CPU routines profiling.
+ TODO: Profile kernels using nvprof.
+ TODO: Performance regression detection.

## How to use
### Benchmark creation
Similarly to pytest, benchmarks are stored in `bench_*.py` files.
Inside a bench file `class` definitions with `Benchmark` on their name are
considered as such.

There are 3 special methods in the benchmark
`setup` used to generate the actual inputs to the routines and cupy memory allocations.
`teardown` used to clean up the state and free the allocated memory.
`args_key` which returns a string to name the current benchmark case (parameters combination)

`setup` and `teardown` are called before and after every function in the class starting with
`time_`. These `time_` functions are in charge of doing the actual benchmark and they will be
executed multiple times to get statistically significant results. Therefore object state modification
should be avoided.

Benchmars are parametrized by using the `params` class attribute. This attribute is a dictionary
whose entries are each a list of possible values for that parameter name. The  cross-product of all the
entries is calculated before running the benchmark, and every benchmark is run for all the possible
combinations of the parameters. The parameters are set before calling the `setup` functions and the
current value is accessed with `self.parameter_name`.

### Types of benchmarks
Benchmarks should inherit from one of these base classes.
+ `cupy_prof.benchmark.CupyBenchmark` Basic benchmark class for testing cupy only that plots
a logarithmic time graph of both the cpu and gpu time.
+ `NumpyCompareBenchmark` performs a comparison within numpy and cupy for the specified routine.
numpy or cupy namespaces are accessed in the benchmark with `self.xp` as in chainer tests.
It aditionally calculates the speedup and plot it too.

### Plot parameters
Benchmarks are plotted according the `_plots` class variable in the benchmarks.
```python
    _plots = [{'facet': {'col': 'name', 'hue': 'backend'},
               'plot': 'line',
               'x': 'key',
               'y': 'time',
               'yscale': 'log'},
              {'facet': {'col': 'name', 'hue': 'xp'},
               'plot': 'bar',
               'x': 'key',
               'y': 'speedup'}]
```
`_plots` is a list with all the graphs that will be generated for the class.
the `facet` entry is in charge of subplotting all the `def bench_` results in 
different sub-graphs. The first example says that for this facet, the column is the
benchmark name (method name in the class definition) and the `hue`, or the legend is the
backend, with is either `numpy`, `cupy-gpu`, `cupy-cpu`. For each of the graph, the x 
is the `key` values obtained by calling the benchmark `args_key` method. and the `y` is the execution time.

This values can be altered in the setup to generate different kinds of graphs.

### Running benchmarks

```
$ python prof.py benchmarks
```
Alternatively, if a directory is specified, it will collect all the benchmarks with the file name
starting with `bench_` as in pytest.

### Comparing different commits or branches

```
python prof.py --repo /home/ecastill/em-cupy --commits master v7 --plot -- benchmarks/bench_ufunc_cupy.py
```

Is it possible to compare different commits or branches of a repository.
This script will automatically checkout the branch and compile cupy, but it will install it in a virtual environment
`virtualenv` is required for this functionaility.

`--repo` can specify a common repo for all the commits, or a list of repositories, one per-commit, so that build time
can be reduced.

