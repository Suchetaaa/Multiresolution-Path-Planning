import numpy as np 
import pywt
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("lenna.png")
img = np.asarray(img)

coeffs = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs

cA = np.sqrt(np.square(cA[:, :, 0]) + np.square(cA[:, :, 1]))
cH = np.sqrt(np.square(cH[:, :, 0]) + np.square(cH[:, :, 1]))
cV = np.sqrt(np.square(cV[:, :, 0]) + np.square(cV[:, :, 1]))
cD = np.sqrt(np.square(cD[:, :, 0]) + np.square(cD[:, :, 1]))


titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']


fig = plt.figure(figsize=(8, 6))

for i, a in enumerate([cA, cH, cV, cD]):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()



