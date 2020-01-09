import cupy
from cupy import testing
from cupy import cutensor

from cupy_prof import benchmark


# TODO Get these from CuPy benchmarks?
class CubCuTensorReductionBenchmark(benchmark.CupyBenchmark):

    params = {'case': [
                       # reduce at head axes
                       {'shape': (200, 300, 400),
                        'axis': (0, 1, 2), 'name': 'all'},
                       # reduce at middle axes
                       {'shape': (200, 300, 400),
                        'axis': (1, 2), 'name': 'first'},
                       # reduce at tail axes
                       {'shape': (200, 300, 400),
                        'axis': (1, 2), 'name': 'mid'},
                       # out_axis = ()
                       {'shape': (200, 300, 400),
                        'axis': (0, 1, 2), 'name': 'batch'}],
              'datatype': ['float64', 'float32'],
              'mode': ['naive', 'cub', 'cutensor']}

    def setup(self, bench_name):
        a = testing.shaped_random(self.case['shape'], self.xp, self.datatype)
        self.axis = self.case['axis']
        self.array = a
        cupy.cuda.cub_enabled = self.mode == 'cub'
        out_shape = [dim for i, dim in enumerate(a.shape) if (i not in self.axis)]
        self.out = cupy.zeros(out_shape, dtype=self.datatype)
        if self.mode == 'cutensor':
            self.desc_x = cutensor.create_tensor_descriptor(self.array)
            self.desc_out = cutensor.create_tensor_descriptor(self.out)
            self.mode_x = (0, 1, 2)
            self.mode_out = [i for i in self.mode_x if (i not in self.axis)]

    def time_reduction(self):
        if self.mode == 'cutensor':
            cutensor.reduction(1, self.array, self.desc_x, self.mode_x, 0, self.out, self.desc_out, self.mode_out)
        else:
            cupy.sum(self.array, self.axis, None, self.out)

    def teardown(self):
        self.array = None
        self.out = None

    def args_key(self):
        return '{} {} {}'.format(self.case['name'], self.mode, self.datatype)
