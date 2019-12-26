import numpy

from cupy import testing

from cupy_prof.benchmarks import benchmark


# TODO Get these from CuPy benchmarks?
class ReductionBenchmark(benchmark.CupyBenchmark):

    datatype = [numpy.float64, numpy.complex128]

    params = [
        # reduce at head axes
        {'shape': (200, 400, 300), 'trans': (2, 1, 0), 'axis': (0, 1), 'name': 'head'},
        # reduce at middle axes
        {'shape': (200, 400, 300, 10), 'trans': (3, 2, 1, 0), 'axis': (1, 2), 'name': 'middle'},
        # reduce at tail axes
        {'shape': (200, 400, 300), 'trans': (2, 1, 0), 'axis': (1, 2), 'name': 'tail'},
        # out_axis = ()
        {'shape': (200, 400, 300), 'trans': (2, 1, 0), 'axis': (0, 1, 2), 'name': 'out'},
    ]

    def setup(self, bench_name, xp, datatype, params):
        a = testing.shaped_random(params['shape'], xp, datatype)
        self.axis = params['axis']
        self.keepdims = False
        trans = params['trans']
        if trans:
            a = a.transpose(*trans)

        self.xp = xp
        self.array = a
        # sum_func = self.get_sum_func()
        # if xp == cupy:
        #     return sum_func(
        #         a, axis=axis, keepdims=keepdims)
        # else:
        #     return a.sum(axis=axis, keepdims=keepdims, dtype='b')

    def time_reduction(self):
        self.array.sum(axis=self.axis, keepdims=self.keepdims)

    def teardown(self):
        self.array = None

    def args_key(self, **kwargs):
        return kwargs['params']['name']
