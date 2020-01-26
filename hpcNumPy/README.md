Compiled vs interpreted languages

Explicit vs implicit programming languages
- Python is implicit as you don't need to specify what type of variable you are declaring. For explicit typing the compiler can create an optimised way of storing the variable based on it's type.

Global interpreter lock (GIL):
Running python in parallel with multiple threads then only one thread can execute at once due to the lock. 1 thread owns the lock and therefore parallelization is not great. Numpy releases the GIL or works around it.

Program can split into two with out separating the address space. 

NumPy is written in C to get more performance but it is a python library. It is less flexible because of this. It maps naturally 