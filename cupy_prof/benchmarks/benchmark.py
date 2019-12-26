import numpy
import cupy

import cupy_prof


class CupyBenchmark(object):
    _xp = [cupy]
    _plots = {'facet': {'col': 'name', 'hue': 'backend'},
              'plot': 'line',
              'x': 'key',
              'y': 'time',
              'yscale': 'log'}

    def process_dataframe(self, df):
        # We dont need a numpy-gpu time, clean it.
        df = df.drop(df[df['backend'] == 'numpy-gpu'].index)
        return df

    def plot(self, df):
        for plot in self._plots:
            plotter = cupy_prof.plot.SnsPlotter(plot)
            plotter.plot(self.__class__.__name__, df)


class NumpyCompareBenchmark(CupyBenchmark):
    """Runs a comparison with numpy and
    plots both, execution time and speedup
    """
    _xp = [numpy, cupy]
    _plots = [{'facet': {'col': 'name', 'hue': 'backend'},
               'plot': 'line',
               'x': 'key',
               'y': 'time',
               'yscale': 'log'},
              {'facet': {'col': 'name', 'hue': 'xp'},
               'plot': 'bar',
               'x': 'key',
               'y': 'speedup'}]

    def process_dataframe(self, df):
        others = df
        baseline = others.merge(df.loc[df['xp'] == 'numpy'],
                                on=['name', 'key', 'dev', 'run'], how='left')
        df['speedup'] = (baseline['time_y']/baseline['time_x'])
        df = df.drop(df[df['backend'] == 'numpy-gpu'].index)
        return df
