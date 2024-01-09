from pathlib import Path
import random

import numpy as np
import timm
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.manifold import TSNE

N_IMAGES = 1000
FOLDER_NAME = "imgs"


# Load random images from a folder
def load_images(folder, n) -> list[Image.Image]:
    
    images = []
    file_paths = []
    accepted_extensions = [".jpg", ".jpeg", ".png"]

    for path in Path(folder).rglob("*"):
        if path.suffix.lower() in accepted_extensions:
            file_paths.append(path)
    
    if n > 0:
        selected_paths = random.sample(file_paths, min(n, len(file_paths)))
    else:
        selected_paths = file_paths
    
    for path in selected_paths:
        with Image.open(path) as img:
            images.append(img.copy())
    
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


# Update the zoom level on scroll
def update_zoom(event):
    if event.inaxes is ax:
        for im in images:
            current_zoom = im.get_zoom()
            zoom_factor = 1.2 if event.button == "up" else 0.8
            im.set_zoom(current_zoom * zoom_factor)
            plt.draw()


# Create model
model = timm.create_model("vgg11_bn", pretrained=True, num_classes=0)
data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_cfg)

# Load images
images = load_images(FOLDER_NAME, N_IMAGES)

# Get features and embed them in 2D
features = extract_features(images, model, transform)
embedded_features = tsne_embedding(features, 2)

# Plot the embedded images
x = embedded_features[:, 0]
y = embedded_features[:, 1]

fig, ax = plt.subplots()

for i, image in enumerate(images):
    image.thumbnail((300, 300))
    im = OffsetImage(image, zoom=0.18)
    ab = AnnotationBbox(im, (x[i], y[i]), frameon=False)
    images[i] = im
    ax.add_artist(ab)

# Register the update_zoom function with the on_scroll event
fig.canvas.mpl_connect("scroll_event", update_zoom)
ax.scatter(x, y, s=1, alpha=0)
plt.show()