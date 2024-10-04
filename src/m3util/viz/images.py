import matplotlib.image as mpimg


def display_image(ax, image):
    """
    Display an image on the given matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to display the image.
        image (str): The file path to the image to be displayed.

    Returns:
        None
    """
    img = mpimg.imread(image)
    ax.imshow(img)
    ax.axis("off")  # Turn off axis for image subplot
