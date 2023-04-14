
#import pywt
#import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

#from collections import defaultdict

from matplotlib import rc
import matplotlib


def initi(font_size:int=32):
    matplotlib.rcParams.update({'font.size': font_size})
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    
    
def add_colorbar_sar(mappable, title:str='',ticks:bool=False):
    
    last_axes = plt.gca()
    if ticks==False:
        plt.tick_params(left = False, right = True , labelleft = False ,
                labelbottom = False, bottom = False)
    #else:
    #    
    #    
    last_axes.xaxis.set_tick_params(labelbottom=False)
    last_axes.xaxis.set_tick_params(labeltop=True)
    
    last_axes.xaxis.set_tick_params(labelleft=False)
    last_axes.xaxis.set_tick_params(labelright=True)
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position='bottom', size="10%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax,orientation='horizontal')
    cax.set_xlabel(f'{title}')
    plt.sca(last_axes)
    return cbar