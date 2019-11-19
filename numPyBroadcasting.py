from matplotlib import pyplot as plt
import imageio
import numpy as np

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


%matplotlib inline
image = imageio.imread('cat.jpg') / 256
plt.imshow(image)
plt.show()
