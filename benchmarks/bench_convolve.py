from unittest import mock

from cupy import testing

from cupy_prof import benchmark


# TODO Get these from CuPy benchmarks?
class ConvolveBenchmark(benchmark.CupyBenchmark):

    params = {'shape': ((100,), (200,),
              (5000,), (10000,), (100000,))}

    def setup(self, bench_name):
        self.a = testing.shaped_random(self.shape, self.xp, self.xp.float32)
        self.b = testing.shaped_random(self.shape, self.xp, self.xp.float32)

    def time_convolve_dot(self):
        with mock.patch('cupyx.scipy.signal.choose_conv_method') as method:
            method.return_value = 'direct'
            self.xp.convolve(self.a, self.b)

    def time_convolve_fft(self):
        with mock.patch('cupyx.scipy.signal.choose_conv_method') as method:
            method.return_value = 'fft'
            self.xp.convolve(self.a, self.b)

    def teardown(self):
        self.a = None
        self.b = None

    def args_key(self, **kwargs):
        # Need to use kwargs as some of the
        # elements may not be used in all benchmarks
        return self.shape[0]
