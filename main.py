import random
from functools import partial
from pathlib import Path

import timm
import timm.data
import torch
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, UnidentifiedImageError
from sklearn.manifold import TSNE

N_IMAGES = 1000
FOLDER_NAME = "imgs"
BATCH_SIZE = 32


# Determine the device to use
device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {device}")


# Load random images from a folder
def load_images(folder, n) -> list[Image.Image]:
    file_paths = [
        path
        for path in Path(folder).rglob("*")
        if path.suffix.lower() in Image.registered_extensions()
    ]

    selected_paths = random.sample(file_paths, min(n, len(file_paths)))

    images = []
    for path in selected_paths:
        try:
            with Image.open(path) as img:
                images.append(img.copy())
        except UnidentifiedImageError:
            print(f"Could not load image {path}")

    return images


# Extract features from images
def extract_features(images, model, transform, batch_size):
    n_images = len(images)
    n_features = model.num_features
    features = torch.zeros((n_images, n_features))

    # Process images in batches
    for i in range(0, n_images, batch_size):
        batch_images = images[i : i + batch_size]
        batch_tensors = torch.stack([transform(image) for image in batch_images]).to(device)

        with torch.no_grad():
            batch_features = model(batch_tensors).cpu()

        features[i : i + batch_size] = batch_features

    return features


# TSNE for feature embedding
def tsne_embedding(features, dimensions):
    tsne = TSNE(n_components=dimensions, perplexity=min(30, len(features) - 1))
    embedded_features = tsne.fit_transform(features)
    return embedded_features


# Update the zoom level on scroll
def update_zoom(images, ax, event):
    zoom_factor = 1.2 if event.button == "up" else 0.8
    if event.inaxes is ax:
        for im in images:
            current_zoom = im.get_zoom()
            im.set_zoom(current_zoom * zoom_factor)
    plt.draw()


print("Loading model...")
model = timm.create_model("vgg11_bn", pretrained=True, num_classes=0).to(device)
model.eval()

# Get model transform
data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_cfg)

print("Loading images...")
images = load_images(FOLDER_NAME, N_IMAGES)
if not images:
    print("No images found")
    exit()

print(f"Extracting features...")
features = extract_features(images, model, transform, BATCH_SIZE)
embedded_features = tsne_embedding(features, 2)

print("Plotting...")
x = embedded_features[:, 0]
y = embedded_features[:, 1]

fig, ax = plt.subplots()

for i, image in enumerate(images):
    image.thumbnail((200, 200))
    im = OffsetImage(image, zoom=0.18)
    ab = AnnotationBbox(im, (x[i], y[i]), frameon=False)
    images[i] = im
    ax.add_artist(ab)

# Connect the scroll event to the zoom function
update_zoom_partial = partial(update_zoom, images, ax)
fig.canvas.mpl_connect("scroll_event", update_zoom_partial)
ax.scatter(x, y, s=1, alpha=0)
plt.show()
