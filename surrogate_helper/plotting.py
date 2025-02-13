import numpy as np
import keras
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from PIL import Image
import io

def plotData(y_test, y_pred, folder: str, label: str, ranges=("full")):
    """Plot comparisons between true and predicted values.
    All plots are also saved as pdf in the same folder according to the label and range of test data printed.

    Args: 
        y_test: Array of true values to be plotted
        y_pred: Array of predicted values to be plotted
        folder: String of the foldername without /
        label: String of the describing label, this gives the first part of the filename
        ranges: Tuple of ranges to be plotted. 
            Valid entries are "full", "day", "week".
    
    """
    for x in ranges:
        if x is None:
            print("Empty Range, cannot create Figure!")
        elif "full" in x:
            plt.figure(figsize=(12,4))
            plt.plot(y_test, label="True")
            plt.plot(y_pred, alpha=0.8, label="Pred")

        else:
            if x == "day": index = 24
            elif x == "week": index = 7*24
            plt.figure(figsize=(6,4))
            plt.plot(y_test[:index], label="True")
            plt.plot(y_pred[:index], alpha=0.8, label="Pred")
        plt.legend()
        Path(f"./{folder}/logs/diagrams").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"./{folder}/logs/diagrams/{label}-{x}.pdf", bbox_inches='tight')
        plt.show()

def plotActivations(model, folder: str, label: str, data):
    """Plot Activations of each layer given an input.
    This includes a visualization of the structure of the model
    
    Args:
        model: Model object to be analyzed
        data: Input data to feed the model, generally as a batched dataset given by processData()
        folder: String of the foldername without /
        label: String of the describing label, this gives the first part of the filename
        
    """
    # Create Model with outputs for each layer
    layer_names = [layer.name for layer in model.layers]
    activation_model = keras.models.Model(inputs=model.input, outputs=[model.get_layer(name).output for name in layer_names])
    activations = activation_model.predict(data)

    # Add missing add layer if necessary
    if "add" not in layer_names:
        layer_names.insert(4, "add")
        activations.insert(4, None)

    # Create plot containing the model plot and the activation of each layer
    fig = plt.figure(figsize=(10,14), layout="constrained")
    gs = GridSpec(12,2,figure=fig)

    # left column
    model_graph = fig.add_subplot(gs[:,0])
    model_graph.imshow(Image.open(io.BytesIO(keras.utils.plot_model(model).data)))
    model_graph.axis("off")

    # right column iterate over layers
    ax = dict()
    for i, activation in enumerate(activations):
        if "add" in layer_names[i]:
            continue
        activation = np.expand_dims(activation,-1)
        activation = np.mean(activation[::24,:,:],0)

        ax[i] = fig.add_subplot(gs[i,1])
        if activation.size == 1:
            ax[i].axis("off")
            aspect = None
        elif len(activation.shape) == 2:
            activation = activation.T
            ax[i].get_yaxis().set_visible(False)
            aspect = 'auto'
        else:
            aspect = 'auto'
        ax[i].imshow(activation, cmap='hot', interpolation='nearest', aspect=aspect)
    
    plt.savefig(f"./{folder}/logs/diagrams/{label}-activation.pdf", bbox_inches='tight')
    plt.show()
    
    
def plotScatter(y_test, y_pred, folder: str, label: str):
    x = y_test
    y = y_pred
    minimum = min(min(x),min(y))-0.5
    maximum = max(max(x),max(y))+0.5
    plt.figure(figsize=(4,4))
    plt.gca().set_aspect('equal')
    plt.axhline(y=0, color="k",lw=0.75)
    plt.axvline(x=0, color='k',lw=0.75)
    plt.plot(range(-10,10),range(-10,10), color="k",lw=0.75)


    plt.xlim(minimum,maximum)
    plt.ylim(minimum,maximum)

    plt.scatter(x,y,marker=".",s=1)


    b, a = np.polyfit(x, y, deg=1)
    xseq = np.linspace(-10, 10, num=100)

    #plt.plot(xseq, a + b * xseq, color="tab:red", lw=1.5)
    plt.xlabel("True")
    plt.ylabel("Pred")

    plt.savefig(f"./{folder}/logs/diagrams/{label}-scatter.pdf", bbox_inches='tight')
    