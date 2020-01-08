import cupy

from cupy_prof import benchmark


def _random_matrix(shape, xp):
    return cupy.testing.shaped_random(shape, xp=xp, dtype=cupy.float64)


ufuncs = ['abs', 'absolute', 'add', 'arccos', 'arccosh', 'arcsin', 'arcsinh',
          'arctan', 'arctan2', 'arctanh', 'cbrt', 'ceil', 'conj',
          'copysign', 'cos', 'cosh', 'deg2rad', 'degrees', 'divide', 'divmod',
          'equal', 'exp', 'exp2', 'expm1', 'floor',
          'floor_divide', 'fmax', 'fmin', 'fmod', 'frexp', 'greater',
          'greater_equal', 'hypot', 'isfinite', 'isinf',
          'isnan', 'less', 'less_equal', 'log',
          'log10', 'log1p', 'log2', 'logaddexp', 'logaddexp2', 'logical_and',
          'logical_not', 'logical_or', 'logical_xor', 'maximum', 'minimum',
          'mod', 'modf', 'multiply', 'negative', 'nextafter', 'not_equal',
          'power', 'rad2deg', 'radians', 'reciprocal', 'remainder',
          'rint', 'sign', 'signbit', 'sin', 'sinh',
          'sqrt', 'square', 'subtract', 'tan', 'tanh', 'true_divide', 'trunc']


class UfuncBenchmark(benchmark.NumpyCompareBenchmark):

    params = {'shape': ((100, 100), (200, 200),
              (300, 300), (400, 400), (500, 500))}

    def __init__(self):
        # Hack to create one method per ufunc
        for ufunc in ufuncs:
            exec('self.time_{} = self.run_ufunc'.format(ufunc))

    def setup(self, bench_name):
        self.f = getattr(self.xp, bench_name.split('time_')[1])
        self.args = (_random_matrix(self.shape, self.xp),) * self.f.nin

    def args_key(self, **kwargs):
        # Need to use kwargs as some of the
        # elements may not be used in all benchmarks
        return self.shape[0]

    def run_ufunc(self):
        self.f(*self.args)


"""
class BroadcastBenchmark(benchmark.NumpyCompareBenchmark):
    def setup(self):
        shapes_a = ()
        shapes_b = ()
        self.input_arrays = ((cupy.ones(sa, dtype=cupy.float64),
                             cupy.ones(sb, dtype=cupy.float64))
                             for sa, sb in zip(shapes_a, shapes_b))

    def time_broadcast(self, input_arrays):
        a, b = input_arrays
        a-b
"""
