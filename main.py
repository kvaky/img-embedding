import base64
import io
import os
from matplotlib import pyplot as plt

import numpy as np
import timm
from PIL import Image
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# Load images from a folder
def load_images(folder, n=-1):
    images = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if not file.endswith(".jpeg"):
                continue
            with Image.open(os.path.join(root, file)) as img:
                images.append(img.copy())
    if n > 0:
        images = np.random.choice(images, n, replace=False)
    return images


# Extract features from images
def extract_features(images, model, transform):
    features = []
    for image in images:
        img_tensor = transform(image).unsqueeze(0)
        feature = model(img_tensor).squeeze().detach().numpy()
        features.append(feature)
    return np.array(features)


# TSNE for feature embedding
def tsne_embedding(features, dimensions):
    tsne = TSNE(n_components=dimensions)
    embedded_features = tsne.fit_transform(features)
    return embedded_features


# Create model
model = timm.create_model("vgg11_bn", pretrained=True, num_classes=0)
data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_cfg)

# Load images
folder = "imgs"
images = load_images(folder, 1000)

# Embed features in 2D
features = extract_features(images, model, transform)
embedded_features = tsne_embedding(features, 2)


def plot_images_on_scatter(x, y, images):
    fig, ax = plt.subplots()

    # Define a function to update the zoom level on scroll
    def update_zoom(event):
        if event.inaxes is ax:
            for im in images:
                current_zoom = im.get_zoom()
                zoom_factor = 1.2 if event.button == 'up' else 0.8
                im.set_zoom(current_zoom * zoom_factor)
                plt.draw()

    for i, image in enumerate(images):
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="png")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        img = Image.open(io.BytesIO(base64.b64decode(img_base64)))

        im = OffsetImage(img, zoom=0.18)
        ab = AnnotationBbox(im, (x[i], y[i]), frameon=False)
        images[i] = im
        ax.add_artist(ab)

    # Register the update_zoom function with the on_scroll event
    fig.canvas.mpl_connect('scroll_event', update_zoom)
    ax.scatter(x, y, s=1, alpha=0)
    plt.show()


# Plot the embedded images
x = embedded_features[:, 0]
y = embedded_features[:, 1]

plot_images_on_scatter(x, y, images)
