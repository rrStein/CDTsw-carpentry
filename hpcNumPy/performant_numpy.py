from random import random
import matplotlib.pyplot as plt
import numpy as np

def add_two_numbers(x, y):
    return x + y

# In python we can call this function with strings 
add_two_numbers('a','b')

"""######## TIMING ########"""
%timeit result = some_function(argument1, argument2)
# will report the time taken to perform the operation on the same line as the %timeit magic. Meanwhile, the cell magic

# In bash:
""" 
$ python -m timeit --setup='import numpy; x = numpy.arange(1000)' 'x ** 2'
"""

A = np.random.random(10000)
B = np.random.random(10000)
C = np.empty_like(A)

# The old waes of doing it (doesn't map well to NumPy):
%%timeit
for i in range(10_000): # Can write 10000 as 10_000 to increase readability in python 3.6+
    C[i] = A[i] + B[i]

# Optimal way for numpy:
%%timeit
C = A + B

""" 
Using timeit you find that doing stuff element-wise is up to 1000x slower in this case
"""

def naive_dist(p,q):
    square_distance = 0
    for p_i, q_i in zip(p, q):
        square_distance += (p_i - q_i) ** 2
    return square_distance ** 0.5

# zip function example:
"""
a = [1,2,3]
b = [4,5,6,7]
for e1, e2 in zip(a,b):
    print(e1,e2)
"""

p = [i for i in range(1000)]
q = [i + 2 for i in range(1000)]

# Benchmark for numPy to beat:
%timeit naive_dist(p, q)

"""
NumPy has a way to do multiplication across the whole array
"""

def simple_numpy_dist(p, q):
    return (np.sum((p - q) ** 2)) ** 0.5

p = np.arange(1000)
q = np.arange(1000) + 2

%timeit simple_numpy_dist(p, q)
# NumPy is about 100x faster in this case. It is not being interpreted and 
# the data type is not being inspected as in python loop. Each element in np array
# comes after the other in memory so it is easier to "predict" for the cpu.
# It also uses vector units which can do one calculation on more than 1 number
# at a time so it is much faster.

def numpy_norm_dist(p,q):
    return np.linalg.norm(p-q)

%timeit numpy_norm_dist(p,q)
# Fewer operations, more optimized, much faster performance.

def naive_dists(ps, qs):
    return [naive_dist(p,q) for p,q in zip(ps, qs)]

ps = [[i + 1000 * j for i in range(1000)] for j in range(1000)]
qs = [[i + 1000 * j + 2 for i in range(1000)] for j in range(1000)]

%timeit naive_dists(ps,qs)

# Use numpy method, summing over one axis only
def simple_numpy_dists(ps, qs):
    return np.sum((ps - qs) ** 2, axis=1) ** 0.5

ps = np.arange(1000000).reshape((1000, 1000))
qs = np.arange(1000000).reshape((1000, 1000)) + 2

%timeit simple_numpy_dists(ps, qs)
# ~100x performance increase, early lunch
# Can we do better with the norm function?
def numpy_norm_dists(ps, qs):
    return np.linalg.norm(ps - qs, axis=1)

%timeit numpy_norm_dists(ps, qs)
# This time it is slower than the explicit computation! (unlike in the 1D case)
# norm function is probably most optimized for 1D arrays because that is when it is most used.

def einsum_dists(ps, qs):
    difference = ps - qs
    return np.einsum('ij,ij->i', difference, difference) ** 0.5

%timeit einsum_dists(ps,qs)
# Performance increase but readability suffered quite a bit.
# Important to understand where your program spends most of its time.

"""
######################### AVOIDING CONDITIONALS #################################
"""

x_range = np.arange(-30, 30.1, 0.1)
y_range = np.arange(-30, 30.1, 0.1)
x_values, y_values = np.meshgrid(x_range, y_range, sparse=False)
peaked_function = (np.sin(x_values**2 + y_values**2) /
                   (x_values**2 + y_values**2) ** 0.25)
plt.style.use("seaborn")
plt.imshow(peaked_function)
plt.show()

# If only interested in the highest points
high_points = peaked_function > 0.8
plt.imshow(high_points)
plt.show()

""" 
########### MULTIPLICATION ############ 
"""

mask = peaked_function >= 0
plt.imshow(mask * peaked_function)
plt.show()

""" 
########### FANCY INDEXING ############ 
"""

mask = peaked_function > 0.9
print('Values of peaked_function greater than 0.9:')
print(peaked_function[mask])

peaked_function[peaked_function < 0] = 0
peaked_function = np.where(peaked_function < 0, 0, peaked_function)

""" 
########### MASKING ARRAYS ############ 
"""

mask = peaked_function > 0.9
masked_peaks = np.ma.masked_array(peaked_function, mask=~mask)

%%timeit
mask_mean = np.mean(masked_peaks)

np.mean(peaked_function)

##### TESTING ######
#Fancy indexing
%%timeit
# np.mean(peaked_function[mask])
peaked_function[mask].mean()

# Multiplication
%%timeit
np.sum(mask*peaked_function) / np.sum(mask)

"""
############# CALCULATING PI ###########
"""

def naive_pi(number_of_samples):
    within_circle_count = 0

    for _ in range(number_of_samples):
        x = random()
        y = random()

        if x ** 2 + y ** 2 < 1:
            within_circle_count += 1

    return within_circle_count / number_of_samples * 4

n = 1_000_000
%%timeit
naive_pi(n)
n=3
def numpy_pi(n):
    x = np.random.random(n)
    y = np.random.random(n)
    s = np.square(x) + np.square(y)
    within = np.sum(s<1)
    return within/n * 4
    
n = 1000000
%%timeit
numpy_pi(n)


### Another way to do it:
def numpy_pi_1(number_of_samples):
    # Generate all of the random numbers at once to avoid loops
    samples = np.random.random(size=(number_of_samples, 2))

    # Use the same np.einsum trick that we used in the previous example
    # Since we are comparing with 1, we don't need the square root
    squared_distances = np.einsum('ij,ij->i', samples, samples)

    # Identify all instances of a distance below 1
    # "Sum" the true elements to count them
    within_circle_count = np.sum(squared_distances < 1)
    return within_circle_count / number_of_samples * 4

%%timeit
numpy_pi_1(n)


def mandelbrot(c, maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return 0


def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter):
    real_range = np.linspace(xmin, xmax, width)
    imaginary_range = np.linspace(ymin, ymax, height)
    result = np.empty((width, height))
    for i in range(width):
        for j in range(height):
            result[i, j] = mandelbrot(real_range[i] +
                                    1j*imaginary_range[j], maxiter)
    return results












