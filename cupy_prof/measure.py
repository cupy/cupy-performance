import numpy
import pandas as pd

import cupy_prof  # NOQA
from collections import defaultdict  # NOQA

import seaborn as sns
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # NOQA


class Measure(object):

    def __init__(self, benchmark, modules):
        self.benchmark = benchmark
        self.modules = modules
        self.df = defaultdict(list)

    def capture(self, name, times):
        for key in times:
            for dev in times[key]:
                for i, time in enumerate(times[key][dev]):
                    self.df['xp'].append(self.module_name)
                    self.df['backend'].append('{}-{}'.format(
                                              self.module_name, dev))
                    self.df['name'].append(name)
                    self.df['key'].append(key)
                    self.df['time'].append(time)
                    self.df['dev'].append(dev)
                    self.df['run'].append(i)

    def measure(self, csv=True, plot=True):
        for module in self.modules:
            runner = cupy_prof.Runner(self.benchmark)
            self.module_name = module.__name__
            runner.run(self.capture, module)
        self.df = pd.DataFrame(self.df)
        # Calculate speedup over numpy
        others = self.df
        baseline = others.merge(self.df.loc[self.df['xp'] == 'numpy'],
                                on=['name', 'key', 'dev', 'run'], how='left')
        self.df['speedup'] = (baseline['time_y']/baseline['time_x'])
        # Save csv
        if csv:
            bench_name = self.benchmark.__class__.__name__
            self.df.to_csv('{}.csv'.format(bench_name))
        if plot:
            self.plot()

    def _get_lines_and_errors(self, series):
        lines = []
        # TODO(ecastill) cleaner way to get the keys
        keys = None
        for module in series:
            times = series[module]
            keys = sorted(times.keys())
            # CPU & GPU Times
            labels = ['cpu', 'gpu']
            for l in labels:
                means = numpy.array([times[key][l].mean() for key in keys])
                std = numpy.array([times[key][l].std() for key in keys])
                mins = numpy.array([times[key][l].min() for key in keys])
                maxes = numpy.array([times[key][l].max() for key in keys])
                lines.append(('{}-{}'.format(l, module),
                              means, std, mins, maxes))
        # Lets do this using the dataframe
        return keys, lines

    def plot(self):
        # Plot Exec times
        bench_name = self.benchmark.__class__.__name__
        sns.set()
        g = sns.FacetGrid(self.df, col='name', hue="backend", col_wrap=4,
                          sharey=False, sharex=False)
        g.map(sns.lineplot, "key", "time").set(yscale='log')
        g.add_legend()
        for ax in g.axes.flatten():
            ax.set_title(ax.get_title().split('_')[1])
        g.fig.savefig("{}_exec_time.png".format(bench_name))

        # Plot speedup
        sns.set()
        df = self.df.loc[(self.df['dev'] == 'gpu')
                         & (self.df['xp'] != 'numpy')]
        g = sns.FacetGrid(df, col='name', hue="xp", col_wrap=4,
                          sharey=False, sharex=False)
        g.map(sns.barplot, "key", "speedup")
        for ax in g.axes.flatten():
            ax.set_title(ax.get_title().split('_')[1])
        g.add_legend()
        g.fig.savefig("{}_speedup.png".format(bench_name))
