import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

def pca(X, D):
    """
    PCA
    Input:
        X - An (1024, 1024) numpy array
        D - An int less than 1024. The target dimension
    Output:
        X - A compressed (1024, 1024) numpy array
    """
    u, e, v = np.linalg.svd(np.dot(X.T, X))
    eigen = v.T[:, 0:D]
    final = np.dot(X, eigen)
    #return np.dot(final, eigen.T)
    return final


if __name__ == '__main__':
    X = pd.read_csv("normdataBAL0715.txt", sep = '\t', skiprows= range(1001, 45015))
    print(X)
    D = 2


# if __name__ == '__main__':
#     D = 256

#     a = Image.open('data/20180108_171224.jpg').convert('RGB')
#     ax1 = plt.subplot(1, 2, 1)
#     ax1.set_title('Original')
#     ax1.imshow(a)
#     b = np.array(a)
#     c = b.astype('float') / 255.
#     for i in range(3):
#         x = c[:, :, i]
#         mu = np.mean(x)
#         x = x - mu
#         x_true = sklearn_pca(x, D)
#         x = pca(x, D)
#         assert np.allclose(x, x_true, atol=0.05)  # Test your results
#         x = x + mu
#         c[:, :, i] = x

#     b = np.uint8(c * 255.)
#     a = Image.fromarray(b)
#     ax2 = plt.subplot(1, 2, 2)
#     ax2.set_title('Compressed')
#     ax2.imshow(a)
#     plt.show()

