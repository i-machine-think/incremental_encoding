from __future__ import print_function

import os
import matplotlib
#if os.environ.get('DISPLAY','') == '':
#        print('no display found. Using non-interactive Agg backend')
#        matplotlib.use('Agg')

import matplotlib.pyplot as plt

from machine.util.log import LogCollection


class CustomLogCollection(LogCollection):

    def plot_metric(self, metric_name, restrict_model=lambda x: True,
                          restrict_data=lambda x: True,
                          data_name_parser=None,
                          color_group=False,
                          title='', eor=-1, **line_kwargs):

        """
        Plot all values for a specific metrics. A function restrict can be
        inputted to restrict the set of models being plotted. A function group
        can be used to group the results colour-wise.
        Args
            restrict (func):
            group (func):
        """

        # colormap = plt.get_cmap('plasma')(np.linspace(0,1, 25))
        fig, ax = plt.subplots(figsize=(13,11))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.set_color_cycle(colormap)

        for i, name in enumerate(self.log_names):
            if restrict_model(name):
                label = name+' '
                log = self.logs[i]
                for dataset in log.data.keys():
                    if restrict_data(dataset):
                        label_name = data_name_parser(dataset, name) if data_name_parser else dataset
                        steps = [step/float(232) for step in log.steps[:eor]]
                        if color_group:
                            steps, data = self.prune_data(steps, log.data[dataset][metric_name][:eor])
                            ax.plot(steps, data,
                                     color_group(name, dataset),
                                     label=label+label_name, linewidth=3.0, **line_kwargs)
                        else:
                            ax.plot(steps,
                                     log.data[dataset][metric_name][:eor],
                                     label=label+label_name, **line_kwargs)
                        ax.tick_params(axis='both', which='major', labelsize=20)
                        plt.xlabel("Epochs", fontsize=24)
                        plt.ylabel("Loss", fontsize=24)
                        plt.title(title)

        plt.legend()
        #plt.show()
        return fig
