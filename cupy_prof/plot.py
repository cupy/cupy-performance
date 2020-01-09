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
        num_plots = df[properties['facet']['col']].nunique()
        if 'facet' in properties:
            facet = properties['facet']
            if properties['plot'] == 'bar':
                g = sns.catplot(data=df, x=properties['x'],
                                y=properties['y'], hue=facet['hue'],
                                col=facet['col'],
                                kind=properties['plot'],
                                col_wrap=min(num_plots, 4),
                                sharey=False, sharex=False)
            else:
                g = sns.FacetGrid(df, col=facet['col'], hue=facet['hue'],
                                  col_wrap=min(num_plots, 4),
                                  sharey=False, sharex=False,
                                  height=5)
                ax = g.map(plot_fn, properties['x'], properties['y'])
                if 'yscale' in properties:
                    ax.set(yscale=properties['yscale'])
                g.add_legend()
        else:
            plot_fn(x=properties['x'], y=properties['y'], data=df)

        for ax in g.axes.flatten():
            ax.set_title(ax.get_title().split('_')[1])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                               horizontalalignment='right')
        g.fig.savefig("{}_{}.png".format(name, properties['y']))
