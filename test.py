import cupy
import numpy

import cupy_prof


def run_bench(bench_class):
    bench = benchmark()
    cupy_prof.Measure(bench).measure()


if __name__ == '__main__':
    benchmarks = [# cupy_prof.benchmarks.core.CreationBenchmark,
                  # cupy_prof.benchmarks.core.CreationBenchmarkSquares,
                  # cupy_prof.benchmarks.core.StackingBenchmark,
                  # cupy_prof.benchmarks.core.ArrayBenchmark,
                  # cupy_prof.benchmarks.core.FromArrayBenchmark,
                  # cupy_prof.benchmarks.core.TemporariesBenchmark,
                  cupy_prof.benchmarks.ufunc.UfuncBenchmark]
                  # cupy_prof.benchmarks.noncontiguous.ReductionBenchmark]

    for benchmark in benchmarks:
        print('Running', benchmark.__name__)
        run_bench(benchmark)
