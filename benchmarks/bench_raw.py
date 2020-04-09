import numpy
import cupy

from cupy import testing

from cupy_prof import benchmark


_test_source1 = r'''
extern "C" __global__
void test_sum(const float* x1, const float* x2, float* y, unsigned int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
        y[tid] = x1[tid] + x2[tid];
}
'''


class RawKernelBenchmark(benchmark.CupyBenchmark):

    params = {'case': [
                       {'shape': (10,)},
                       {'shape': (100,)},
                       {'shape': (1000,)},
                       {'shape': (10000,)},
                       {'shape': (20000,)}],
              'datatype': [numpy.float32]}

    def setup(self, bench_name):
        N = self.case['shape'][0]
        x1 = testing.shaped_random((N,), self.xp, self.datatype)
        x2 = testing.shaped_random((N,), self.xp, self.datatype)
        y = cupy.zeros((N,), self.datatype)
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.N = N
        self.kern = cupy.RawKernel(
            _test_source1, 'test_sum')

    def teardown(self):
        self.x1 = None
        self.x2 = None
        self.y = None

    def time_raw(self):
        N = self.N
        self.kern((N,), (1,), (self.x1, self.x2, self.y, N))

    def args_key(self, **kwargs):
        return '{}'.format(self.N)
