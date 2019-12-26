import seaborn as sns
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # NOQA


plotfuncs = {'line': sns.lineplot,
             'bar': sns.barplot}


class Plotter(object):
    def __init__(self, properties):
        self.properties = properties

    def plot(self, name, df):
        raise NotImplementedError()


class SnsPlotter(Plotter):
    def plot(self, name, df):
        properties = self.properties
        plot_fn = plotfuncs[properties['plot']]
        sns.set()
        if 'facet' in properties:
            facet = properties['facet']
            g = sns.FacetGrid(df, col=facet['col'], hue=facet['hue'],
                              col_wrap=4, sharey=False, sharex=False)
            ax = g.map(plot_fn, properties['x'], properties['y'])
            if 'yscale' in properties:
                ax.set(yscale=properties['yscale'])
        else:
            plot_fn(x=properties['x'], y=properties['y'], data=df)

        g.add_legend()
        for ax in g.axes.flatten():
            ax.set_title(ax.get_title().split('_')[1])
        g.fig.savefig("{}_{}.png".format(name, properties['y']))
