import matplotlib.pyplot as plt
import numpy as np

def plot_vector_heatmap(vector, graph_title):

    # Create figure and axis object 
    figure, axis = plt.subplots(figsize=(8, 1.2))

    # Convert 1D vector to 2D array for heatmap
    heatmap_data = vector[np.newaxis, :]

    # Draw heatmap
    heatmap_image = axis.imshow(
        heatmap_data,
        aspect="auto",
        cmap="viridis",
    )

    # Title for the heatmap
    axis.set_title(graph_title)

    # Hide y-axis labels (not meaningful for 1D data)
    axis.set_yticks([])

    # Add color bar for visual reference
    figure.colorbar(
        heatmap_image,
        ax=axis,
        fraction=0.04,
        pad=0.01
    )

    # Adjust layout to prevent clipping
    figure.tight_layout()

    return figure