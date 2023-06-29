import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(metrics, titles, x_labels_list, y_labels_list, ncols=3):
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))

    for idx, (ax, metric, title, x_labels, y_labels) in enumerate(zip(axes.flat, metrics, titles, x_labels_list, y_labels_list)):
        heatmap = ax.imshow(metric, cmap='viridis')
        cbar = fig.colorbar(heatmap, ax=ax)

        ax.set_xticks(np.arange(metric.shape[1]))
        ax.set_yticks(np.arange(metric.shape[0]))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(metric.shape[0]):
            for j in range(metric.shape[1]):
                ax.text(j, i, metric[i, j], ha="center", va="center", color="w")

        ax.set_title(title)

    # Remove empty subplots
    for idx in range(len(metrics), nrows * ncols):
        fig.delaxes(axes.flat[idx])

    plt.tight_layout()
    plt.show()

# Example usage
metric1 = np.array([[1667.,    0.,    0. ,   0. ,   0.],
 [ 455., 1670.,    0. ,   0.,    0.],
 [ 417.,  412., 1384.,    0.,    0.],
 [ 324.,  310.,  339., 1112.,    0.],
 [ 187.,  169.,  199., 160.,  526.]])

metric2 = np.array([[586.,   0.,   0.,   0.,   0.],
                    [117., 358.,   0,   0.,   0.],
                    [ 88.,  71., 388.,   0.,   0.],
                    [160., 122., 105., 692.,   0.],
                    [145., 118., 123., 153., 540.]])
metric3 = np.array([[453. ,  0. ,  0. ,  0.],
 [148., 569.,   0.,   0.],
 [106., 124., 609.,   0.],
 [ 94., 104., 118., 444.]])

x_labels1 = ['Macro', 'Mega', 'Micro', 'Nano', 'No Influencer']
y_labels1 = ['Macro', 'Mega', 'Micro', 'Nano', 'No Influencer']

x_labels2 =['Gaming', 'Other', 'Price Update', 'Technical Information', 'Trading Matters']
y_labels2 = ['Gaming', 'Other', 'Price Update', 'Technical Information', 'Trading Matters']

x_labels3 = ['Advertising', 'Announcement', 'Financial Information', 'Subjective Opinion']
y_labels3 = ['Advertising', 'Announcement', 'Financial Information', 'Subjective Opinion']

plot_metrics([metric1, metric2, metric3], ['Subtask 1', 'Subtask 2', 'Subtask 3'], [x_labels1, x_labels2, x_labels3], [y_labels1, y_labels2, y_labels3])
