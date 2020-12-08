import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

import matplotlib.pyplot as plt

Im_W = 28
Im_H = 28
N_train = 60000
N_test = 10000

# Font data is based on the "Droid Sans Mono" font.
# Each digit is rendered to 28x28.
img = Image.open("Font.png")
img.load()
img = np.asarray(img, dtype=np.float32)

bw = np.sum(img, axis=2)/3
img = np.empty((10, Im_H, Im_W), dtype=np.float32)
for i in range(10):
  img[i] = bw[:, Im_W*i:Im_W*(i+1)]

def noise(im, blur=0.5, gauss=30, sp=0.03):
  # Blur image
  im = gaussian_filter(im, abs(np.random.normal(0, blur)))
  
  # Add gaussian noise
  im += np.random.normal(0, gauss, im.shape)
  
  # Add salt and pepper noise
  im += 255*(np.random.sample(im.shape) < sp)
  im -= 255*(np.random.sample(im.shape) < sp)
  return np.clip(im, 0, 255)
  
train = np.empty((N_train, Im_H, Im_W), dtype=np.float32)
test = np.empty((N_train, Im_H, Im_W), dtype=np.float32)
for i in range(N_train):
  train[i] = noise(img[np.random.randint(10)])
for i in range(N_test):
  test[i] = noise(img[np.random.randint(10)])

train = np.expand_dims(train, axis=3)
test = np.expand_dims(test, axis=3)
np.savez("font_data.npz", train=train, test=test)