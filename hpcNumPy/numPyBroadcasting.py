from numba import vectorize
import math as m
from matplotlib import pyplot as plt
import imageio
import numpy as np
import os 
os.chdir("/Users/qw19176/Documents/Courses/hpcNumPy")

"""
####### BROADCASTING WITH NUMPY #######
"""

A = np.arange(100).reshape((10,10))
B = np.arange(10,20,0.1).reshape((10,10))

A+B

B+100

"""
###### WEIGHTING COLUMNS & ROWS ######
"""

values = np.arange(16).reshape((4, 4))
column_weights = np.arange(0, 400, 100)

values * column_weights

row_weights = np.arange(0,40,10).reshape((4,1))
values * row_weights

print("values:", values.shape)
print("column_weights:", column_weights.shape)
print("row_weights:", row_weights.shape)

# Indices are matched up from right to left.
# Can't broadcast together (4,2) with (4,4) for example but (4,1) is fine.

"""
####### RECTANGULAR ACTION ########
"""

rectangular_values = np.arange(6).reshape((3, 2))
two_vector = np.asarray([1, 10])
rectangular_values * two_vector

two_vector = np.asarray([1, 10, 100])
rectangular_values * two_vector

three_vector_column = np.asarray([1, 10, 100]).reshape((3, 1))
rectangular_values * three_vector_column

values_4d = np.arange(120).reshape((2,3,4,5))
values_4d * row_weights
row_weights.shape 
values_4d.shape

matrix_weights = np.expand_dims(np.expand_dims(
    [[0, 2, 0], [1, 0, 3]], axis=2), axis=3)
values_4d * matrix_weights

matrix_weights = np.asarray(
    [[0,2,0], [1,0,3]])[:, :, np.newaxis, np.newaxis]
matrix_weights * values_4d

matrix_weights = np.expand_dims(np.expand_dims(
    rectangular_values, axis=2), axis=3).swapaxes(0, 1)
values_4d * matrix_weights

"""
####### Array broadcasts for image manipulation ########
"""

image = imageio.imread('ace.jpg') / 256
plt.imshow(image)
plt.show()

supressor = np.zeros([image.shape[0],image.shape[1]])
imagetest = image
imagetest[:,:,2] = supressor
plt.imshow(imagetest)

#Fade to black on right
image2 = image * np.linspace(1, 0, image.shape[1])[:, np.newaxis]
plt.imshow(image2)
plt.show()

#Fade to white on bottom
image3 = 1 - (1 - image) * np.linspace(
    1, 0, image.shape[0]
)[:, np.newaxis, np.newaxis]
plt.imshow(image3)
plt.show()

#Overlay a mask. Use a variant of the peaked_function in the previous episode to 
# multiply all colour channels of the image.
x_range = np.linspace(-30, 30, image.shape[1])
y_range = np.linspace(-30, 30, image.shape[0])
x_values, y_values = np.meshgrid(x_range, y_range, sparse=False)
peaked_function = (np.sin(x_values**2 + y_values**2) /
                   (x_values**2 + y_values**2) ** 0.25)

#scale the output to be between 0 and 1.0, not doing this causes errors on some systems
peaked_function = peaked_function / 2
peaked_function = peaked_function + 0.5

peaked_function = peaked_function[:, :, np.newaxis]
plt.imshow(peaked_function * image)
plt.show()

#Do the same, but applying it only to the red channel.
peaked_function_5 = peaked_function * [1, 0, 0] + [0, 1, 1]
plt.imshow(peaked_function_5 * image)
plt.show()


"""
########## UNIVERSAL FUNCTIONS ########### 
NumPy provides a just-in-time compiler so we don't need to switch between
C and python in one project.
"""

def trig(a, b):
    return m.sin(a ** 2) * m.exp(b)

%env OMP_NUM_THREADS=1
%env NUMBA_NUM_THREADS=1

a = np.ones((5, 5))
b = np.ones((5, 5))

trig(a, b)
# If we try calling this function on a Numpy array, we correctly get an error, 
# since the math library doesn’t know about Numpy arrays, only single numbers.

@vectorize
def trig(a, b):
    return m.sin(a ** 2) * m.exp(b)

a = np.ones((5, 5))
b = np.ones((5, 5))

trig(a, b)  

def numpy_trig(a, b):
    return np.sin(a ** 2) * np.exp(b)


a = np.random.random((1000, 1000))
b = np.random.random((1000, 1000))

%timeit numpy_trig(a, b)
%timeit trig(a, b)


@vectorize
def discriminant(a, b, c):
    return b**2 - 4 * a * c

