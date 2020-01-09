import numpy

from cupy import testing

from cupy_prof import benchmark


# TODO Get these from CuPy benchmarks?
class ReductionBenchmark(benchmark.CupyBenchmark):

    params = {'case': [
                       # reduce at head axes
                       {'shape': (200, 400, 300), 'trans': (2, 1, 0),
                        'axis': (0, 1), 'name': 'head'},
                       # reduce at middle axes
                       {'shape': (200, 400, 300, 10), 'trans': (3, 2, 1, 0),
                        'axis': (1, 2), 'name': 'middle'},
                       # reduce at tail axes
                       {'shape': (200, 400, 300), 'trans': (2, 1, 0),
                        'axis': (1, 2), 'name': 'tail'},
                       # out_axis = ()
                       {'shape': (200, 400, 300), 'trans': (2, 1, 0),
                        'axis': (0, 1, 2), 'name': 'out'}],
              'datatype': [numpy.float64, numpy.complex128]}

    def setup(self, bench_name):
        a = testing.shaped_random(self.case['shape'], self.xp, self.datatype)
        self.axis = self.case['axis']
        self.keepdims = False
        trans = self.case['trans']
        if trans:
            a = a.transpose(*trans)
        self.array = a

    def time_reduction(self):
        self.array.sum(axis=self.axis, keepdims=self.keepdims)

    def teardown(self):
        self.array = None

    def args_key(self):
        return self.case['name']
