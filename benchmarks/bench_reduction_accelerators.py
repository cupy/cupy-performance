import cupy
from cupy import testing
from cupy import cutensor

from cupy_prof import benchmark


def _get_accelerators():
    accelerators = ['naive']
    if cupy.cuda.cub.available:
        accelerators += ['cub']
    if cupy.cuda.cutensor.available:
        accelerators += ['cutensor']
    return accelerators


# TODO Get these from CuPy benchmarks?
class CubCuTensorReductionBenchmark(benchmark.CupyBenchmark):

    params = {
        'case': [
            {'shape': (400, 400, 400), 'axis': (0, 1, 2)},
            {'shape': (400, 400, 400), 'axis': (0,)},
            {'shape': (400, 400, 400), 'axis': (1,)},
            {'shape': (400, 400, 400), 'axis': (2,)},
            {'shape': (400, 400, 400), 'axis': (0, 1)},
            {'shape': (400, 400, 400), 'axis': (0, 2)},
            {'shape': (400, 400, 400), 'axis': (1, 2)},
        ],
        'dtype': ['float32'],
        'accelerator': _get_accelerators(),
    }

    def __init__(self):
        self._plots[0]['plot'] = 'bar'
        # del self._plots[0]['yscale']

    def setup(self, bench_name):
        a = testing.shaped_random(self.case['shape'], self.xp, self.dtype)
        self.axis = self.case['axis']

        self.old_routine_accelerators = cupy.core.get_routine_accelerators()
        self.old_reduction_accelerators = cupy.core.get_reduction_accelerators()

        if self.accelerator == 'naive':
            accelerators = []
        elif self.accelerator == 'cub':
            accelerators = ['cub']
        elif self.accelerator == 'cutensor':
            accelerators = ['cutensor']
        cupy.core.set_routine_accelerators(accelerators)
        cupy.core.set_reduction_accelerators(accelerators)

        self.array = a
        out_shape = [dim for i, dim in enumerate(a.shape) if (i not in self.axis)]
        self.out = cupy.empty(out_shape, dtype=self.dtype)

    def time_reduction(self):
        cupy.sum(self.array, self.axis, None, self.out)

    def teardown(self):
        self.array = None
        self.out = None
        cupy.core.set_routine_accelerators(self.old_routine_accelerators)
        cupy.core.set_reduction_accelerators(self.old_reduction_accelerators)

    def args_key(self):
        return 'axis = %9s, %8s' % (repr(self.axis), self.accelerator)
