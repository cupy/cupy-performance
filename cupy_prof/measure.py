import pandas as pd

import cupy_prof  # NOQA
from collections import defaultdict  # NOQA


class Measure(object):

    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.df = defaultdict(list)

    def capture(self, name, key, times, xp_name):
        for dev in times:
            for i, time in enumerate(times[dev]):
                self.df['xp'].append(xp_name)
                self.df['backend'].append('{}-{}'.format(
                                          xp_name, dev))
                self.df['name'].append(name)
                self.df['key'].append(key)
                self.df['time'].append(time)
                self.df['dev'].append(dev)
                self.df['run'].append(i)

    def measure(self, csv=True, plot=True):
        runner = cupy_prof.Runner(self.benchmark)
        runner.run(self.capture)
        df = pd.DataFrame(self.df)
        # Process the dataframe according to the benchmark
        df = self.benchmark.process_dataframe(df)
        # Save csv
        if csv:
            bench_name = self.benchmark.__class__.__name__
            df.to_csv('{}.csv'.format(bench_name))
        if plot:
            self.benchmark.plot(df)
        return df
