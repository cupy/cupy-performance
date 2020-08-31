import itertools

import gc


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


class Runner(object):

    def __init__(self, benchmark):
        self.benchmark = benchmark

    def _create_key_from_args(self, case):
        benchmark = self.benchmark
        if hasattr(benchmark, 'args_key'):
            return benchmark.args_key()
        if type(case) in (list, tuple) and len(case) == 1:
            return self._create_key_from_args(case[0])
        return str(case)

    def run(self, report):
        import cupy
        import cupyx

        print(cupy.__version__)
        benchmark = self.benchmark
        object_methods = [method_name
                          for method_name in dir(benchmark)
                          if 'time_' in method_name]
        for method_name in object_methods:
            method = getattr(benchmark, method_name)
            args = benchmark.params

            # The benchmark might be tested on different modules
            # we explicitly track the model to easily operate with
            # the dataframes
            args['xp'] = getattr(benchmark, '_xp', cupy)
            for case in product_dict(**args):
                for arg in case:
                    setattr(benchmark, arg, case[arg])
                del case['xp']
                if hasattr(benchmark, 'setup'):
                    benchmark.setup(method_name)
                key = self._create_key_from_args(case)
                name = '{:20} - case {:10}'.format(method_name, key)
                times = cupyx.time.repeat(
                    method, n_repeat=10, n_warmup=10, name=name)
                bench_times = {'cpu': times.cpu_times,
                               'gpu': times.gpu_times}
                print(times)
                report(method_name, key, bench_times, benchmark.xp.__name__)
        if hasattr(benchmark, 'teardown'):
            benchmark.teardown()
        gc.collect()
