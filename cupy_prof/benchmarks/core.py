import numpy

from cupy_prof.benchmarks import benchmark


class CreationBenchmark(benchmark.Benchmark):

    nelems = range(0, 500000, 100000)

    def setup(self, bench_name, xp, nelems):
        self.xp = xp
        self.size = nelems

    def time_empty(self):
        self.xp.array([])

    def time_zeros(self):
        self.xp.zeros(self.size)

    def time_ones(self):
        self.xp.ones(self.size)

    def time_arange(self):
        self.xp.arange(self.size)


class CreationBenchmarkSquares(benchmark.Benchmark):

    nelems = range(0, 50000, 10000)

    def setup(self, bench_name, xp, nelems):
        self.xp = xp
        self.size = nelems

    def time_eye(self):
        self.xp.eye(self.size)

    def time_identity(self):
        self.xp.identity(self.size)


class StackingBenchmark(benchmark.Benchmark):

    nelems = range(0, 1000000, 200000)

    narrays = [2]

    def setup(self, bench_name, xp, nelems, narrays):
        # Use a generator expresion to save memory
        self.xp = xp
        self.inputs = [xp.arange(nelems) for i in range(narrays)]

    def args_key(self, **kwargs):
        # Need to use kwargs as some of the
        # elements may not be used in all benchmarks
        return kwargs['nelems']

    def teardown(self):
        self.inputs = None

    def time_vstack(self):
        self.xp.vstack(self.inputs)

    def time_hstack(self):
        self.xp.hstack(self.inputs)

    def time_dstack(self):
        self.xp.dstack(self.inputs)


class ArrayBenchmark(benchmark.Benchmark):

    shape = [(0,), (1, 1), (100, 100),
             (100, 100, 100), (100, 100, 100, 100)]

    def setup(self, bench_name, xp, shape):
        self.xp = xp
        self.array = numpy.zeros(shape)

    def time_from_numpy(self):
        self.xp.array(self.array)


class FromArrayBenchmark(benchmark.Benchmark):

    shape = [(0,), (1, 1), (100, 100), (200, 200)]

    def setup(self, bench_name, xp, shape):
        self.xp = xp
        self.array = xp.zeros(shape)

    def teardown(self):
        self.array = None

    def args_key(self, **kwargs):
        # Need to use kwargs as some of the
        # elements may not be used in all benchmarks
        return kwargs['shape'][0]

    def time_diag(self):
        self.xp.diag(self.array)

    def time_diagflat(self):
        self.xp.diagflat(self.array)

    def time_triu(self):
        self.xp.triu(self.array)

    def time_tril(self):
        self.xp.tril(self.array)


class TemporariesBenchmark(benchmark.Benchmark):

    size = (10000, 20000, 50000, 100000, 500000)

    def setup(self, bench_name, xp, size):
        # Use a generator expresion to save memory
        self.xp = xp
        self.input_arrays = (xp.ones(size), xp.ones(size))

    def teardown(self):
        self.input_arrays = None

    def args_key(self, **kwargs):
        # Need to use kwargs as some of the
        # elements may not be used in all benchmarks
        return kwargs['size']

    def time_temporary(self):
        a, b = self.input_arrays
        (a * 2) + b

    def time_temporary2(self):
        a, b = self.input_arrays
        (a + b) - 2
