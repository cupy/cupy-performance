import numpy

from cupy_prof import benchmark


class CreationBenchmark(benchmark.NumpyCompareBenchmark):

    params = {'size': range(0, 500000, 100000)}

    def time_empty(self):
        self.xp.array([])

    def time_zeros(self):
        self.xp.zeros(self.size)

    def time_ones(self):
        self.xp.ones(self.size)

    def time_arange(self):
        self.xp.arange(self.size)

    def args_key(self):
        return self.size

class CreationBenchmarkSquares(benchmark.NumpyCompareBenchmark):

    params = {'size': range(0, 50000, 10000)}

    def time_eye(self):
        self.xp.eye(self.size)

    def time_identity(self):
        self.xp.identity(self.size)

    def args_key(self):
        return self.size

class StackingBenchmark(benchmark.NumpyCompareBenchmark):

    params = {'size': range(0, 500000, 100000),
              'narrays': [2]}

    def setup(self, bench_name):
        self.inputs = [self.xp.arange(self.size)
                       for i in range(self.narrays)]

    def args_key(self, **kwargs):
        # Need to use kwargs as some of the
        # elements may not be used in all benchmarks
        return self.size

    def teardown(self):
        self.inputs = None

    def time_vstack(self):
        self.xp.vstack(self.inputs)

    def time_hstack(self):
        self.xp.hstack(self.inputs)

    def time_dstack(self):
        self.xp.dstack(self.inputs)

    def args_key(self):
        return self.size

class ArrayBenchmark(benchmark.CupyBenchmark):

    params = {'shape': [(0,), (1, 1), (100, 100),
              (100, 100, 100), (100, 100, 100, 100)]}

    def setup(self, bench_name):
        self.array = numpy.zeros(self.shape)

    def time_from_numpy(self):
        self.xp.array(self.array)

    def args_key(self):
        return '{}_x'.format(self.shape, len(self.shape))

class FromArrayBenchmark(benchmark.NumpyCompareBenchmark):

    params = {'shape': [(0,), (1, 1), (100, 100), (200, 200)]}

    def setup(self, bench_name):
        self.array = self.xp.zeros(self.shape)

    def teardown(self):
        self.array = None

    def args_key(self, **kwargs):
        # Need to use kwargs as some of the
        # elements may not be used in all benchmarks
        return self.shape[0]

    def time_diag(self):
        self.xp.diag(self.array)

    def time_triu(self):
        self.xp.triu(self.array)

    def time_tril(self):
        self.xp.tril(self.array)


class TemporariesBenchmark(benchmark.NumpyCompareBenchmark):

    params = {'size': (10000, 20000, 50000, 100000, 500000)}

    def setup(self, bench_name):
        self.input_arrays = (self.xp.ones(self.size),
                             self.xp.ones(self.size))

    def teardown(self):
        self.input_arrays = None

    def args_key(self, **kwargs):
        # Need to use kwargs as some of the
        # elements may not be used in all benchmarks
        return self.size

    def time_temporary(self):
        a, b = self.input_arrays
        (a * 2) + b

    def time_temporary2(self):
        a, b = self.input_arrays
        (a + b) - 2
