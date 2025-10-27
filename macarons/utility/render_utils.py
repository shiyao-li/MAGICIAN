import torch
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import iplot

from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene


def plot_point_cloud(points, features, name="", point_size=2, max_points=200000,
                     width=None, height=None, cmap='rgb'):
    device = points.device

    if cmap == 'rgb':
        features_to_plot = features
    elif cmap == 'gray':
        features_to_plot = features.view(-1, 1).expand(-1, 3)
    elif cmap == 'rainbow':
        features_to_plot = (points - points.min()) / (points.max() - points.min())
    else:
        magma_colors = torch.Tensor(plt.get_cmap(cmap)(np.linspace(0, 1, 256))[..., :3]).to(device)
        features_to_plot = magma_colors[(255 * features.clamp(0., 1.).view(-1)).long()]

    pt_cloud = Pointclouds(points=[points],
                           features=[features_to_plot]
                           )
    fig = plot_scene(plots={name: {"cloud": pt_cloud}},
                     # camera_scale=1.,  # 0.3
                     pointcloud_max_points=max_points,  # 20000
                     pointcloud_marker_size=point_size)  # 1

    if (width is not None) or (height is not None):
        fig.update_layout(autosize=False, width=width, height=height)

    return fig


def plot_graph(x, y, x_label='X-axis', y_label='Y-axis', title=None,
               width=None, height=None, mode='lines+markers', names=None):
    if type(y) == list:
        y_list = y
    else:
        y_list = [y]

    data = []
    for i in range(len(y_list)):
        if names is None:
            trace_i = go.Scatter(x=x, y=y_list[i], mode=mode)
        else:
            trace_i = go.Scatter(x=x, y=y_list[i], mode=mode, name=names[i])
        data.append(trace_i)

    layout = go.Layout(title=title, xaxis=dict(title=x_label), yaxis=dict(title=y_label))
    fig = go.Figure(data=data, layout=layout)

    if (width is not None) or (height is not None):
        fig.update_layout(autosize=False, width=width, height=height)

    return fig
