import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import PIL

filename = 'train-images.idx3-ubyte'
f = open(filename, 'rb')

index = 0
buf = f.read()

f.close()

magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
index += struct.calcsize('>IIII')

imlist = []

for i in xrange(numImages):
#for i in xrange(60000):
    im = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')
    im = np.array(im)
    im = im.reshape(28, 28)
    imlist.append(im)

imlist = np.array(imlist)
plt.imshow(imlist[1], cmap='gray')
plt.show()

