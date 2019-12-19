import numpy
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # NOQA


class Plotter(object):
    def __init__(self, name):
        self.name = name
        self.colors = ['red', 'blue', 'black', 'green']

    def plot(self, keys, lines):
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(1, 1, 1)
        x = numpy.arange(len(keys))
        for i, line in enumerate(lines):
            ax.plot(x, line[1], '-', linewidth=1, color=self.colors[i],
                    label=line[0])
            if len(line) > 2:
                ax.errorbar(x, line[1],
                            line[2], fmt=',k', lw=3, ecolor=self.colors[i])
                ax.errorbar(x, line[1],
                            [line[1] - line[3], line[4] - line[1]],
                            fmt=',k', ecolor='indianred', lw=1)

        ax.legend(loc='upper left', numpoints=1)
        ax.set_ylabel('Exec. Time', fontsize=18)
        ax.set_xlabel('Array Size', fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels([str(key) for key in keys], rotation=45)
        ax.set_title(self.name)
        matplotlib.rc('xtick', labelsize=12)
        fig.tight_layout()
        fig.savefig(self.name+'.png', bbox_inches='tight', colormap='viridis')
        plt.close(fig)
