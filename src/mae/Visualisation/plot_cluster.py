import plotly.express as px
import matplotlib.pyplot as plt 


def plot_cluters(pca_data,color):

    fig = px.scatter_matrix(
        pca_data,
        color=color,
        dimensions=range(pca_data[0].shape[0]),
        color_continuous_scale='jet' , #[(0,'#030F4F'), (0.5, '#DADADA'), (1,'#990000')]
        title=f'somehting',
        
    )
    fig.update_traces(diagonal_visible=True,showupperhalf = True)

    fig.update_layout(
        title='PC components',
        width=1600,
        height=1600,
    )

    fig.show()