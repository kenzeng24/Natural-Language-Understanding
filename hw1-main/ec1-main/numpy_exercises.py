"""
Code for Problems 3 and 4 of EC 1. Do not import any additional libraries.
"""
import time
from typing import List

import numpy as np

"""
Problem 3: Complex Operations

Implement the following functions in one line of code using NumPy 
operations. Your one line of code must be at most 80 characters wide
(76 characters excluding indentation).

Use the type hints for each function to see what kinds of inputs the
function accepts and what kinds of outputs (if any) the function must
return. Read the docstrings to find out what each function must do.
"""


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Problem 3a: Implement this function.

    Applies the sigmoid function to each item in an array.

    :param x: An array
    :return: The sigmoid of x
    """
    # Your one line code cannot be wider than this comment ####################
    return 1 / (1+np.exp(-x))


def zero_center(x: np.ndarray) -> np.ndarray:
    """
    Problem 3b: Implement this function.

    Given an array of row vectors, subtracts the mean value of each row
    from the row.

    :param x: An array of shape (n, m), where each row is an m-
        dimensional row vector
    :return: The zero-centered version of x
    """
    # Your one line code cannot be wider than this comment ####################
    return x - x.mean(axis=1).reshape(-1,1)


def even_rows(x: np.ndarray) -> np.ndarray:
    """
    Problem 3c: Implement this function.

    Return the rows (i.e., items along axis 0) with an even index from
    an array.

    :param x: An array
    :return: The items of x where the index along axis 0 are even
    """
    # Your one line code cannot be wider than this comment ####################
    return x[np.arange(x.shape[0]) % 2 == 0]


def mask(x: np.ndarray, mask_val: float, replace_with: float = -1.):
    """
    Problem 3d: Implement this function.

    Mask all instances of a certain value from an array by replacing
    them with some default value. Do this in-place; i.e., by changing
    the value of x instead of producing a return value.

    :param x: The array that the mask will be applied to
    :param mask_val: The value that will be masked out
    :param replace_with: The value to replace mask_val with
    :return: None
    """
    # Your one line code cannot be wider than this comment ####################
    x[x == mask_val] = replace_with


def accuracy(logits: np.ndarray, labels: np.ndarray) -> np.float64:
    """
    Problem 3e: Implement this function.

    Computes the accuracy of a multi-class classifier given its logit
    scores for a batch of inputs and the gold labels for those inputs.

    :param logits: An array of shape (n, m) containing logit scores
        computed by an m-class classifier for n examples
    :param labels: An array of shape (n,) containing the gold labels for
        the n examples
    :return: The proportion of examples for which the highest logit
        score in logits matches the classification given by labels
    """
    # Your one line code cannot be wider than this comment ####################
    return np.mean(logits.argmax(axis=1) == labels)


"""
Problem 4: Analysis of Matrix Multiplication

In NumPy, matrices are represented as 2-dimensional arrays. Without 
NumPy, matrices are represented as lists of lists of numbers. For
example, the list [[1., 2., 3], [4., 5., 6.]] represents a 2-by-3 matrix
where the first row contains 1, 2, and 3, and the second row contains
4, 5, and 6.

In this part of the assignment, you will compare the performance of 
NumPy arrays against lists of lists of numbers. First, fill in the 
matmul_pure_python function with your own implementation of matrix
multiplication. Then, run the measure_time function to compare the
running time of your function with NumPy's @ operator. Record and turn
in the two running times you have measured.

Which implementation of matrix multiplication is faster: the pure Python
version or the NumPy version?
"""

Matrix = List[List[float]]  # This is called a "type alias"


def matmul_pure_python(a: Matrix, b: Matrix) -> Matrix:
    """
    Problem 4a: Implement this function. You may use more than one line
    of code for this problem.

    Multiplies two matrices using only Python built-in functions. Do not
    use NumPy or any other external library.

    :param a: A matrix
    :param b: Another matrix
    :return: a @ b, the product of a and b under matrix multiplication
    """
    n,l = len(a), len(a[0])
    l,m = len(b), len(b[0])

    M = [[0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            M[i][j] = sum([a[i][k] * b[k][j] for k in range(l)])
    return M 


def matmul_numpy(a: Matrix, b: Matrix) -> np.ndarray:
    """
    Multiplies two matrices using NumPy.

    :param a: A matrix
    :param b: Another matrix
    :return: a @ b, the product of a and b under matrix multiplication
    """
    return np.array(a) @ np.array(b)


def measure_time():
    """
    Problem 4b: Run this function to measure the performance of
    matmul_pure_python against NumPy's implementation of matrix
    multiplication.

    Runs matmul_pure_python and matmul_numpy on two randomly generated
    large matrices, and reports the time elapsed during each function.

    :return: None
    """
    # Generate some random matrices
    a = np.random.rand(200, 300).tolist()
    b = np.random.rand(300, 400).tolist()

    # Record time for matmul_pure_python
    start_time = time.time()
    _ = matmul_pure_python(a, b)
    end_time = time.time()

    elapsed = end_time - start_time
    print("Matrix multiplication in pure Python took {:.3f} "
          "seconds.".format(elapsed))

    # Record time for matmul_numpy
    start_time = time.time()
    _ = matmul_numpy(a, b)
    end_time = time.time()

    elapsed = end_time - start_time
    print("Matrix multiplication in NumPy took {:.3f} "
          "seconds.".format(elapsed))


# The if __name__ == "__main__": statement contains code that is run
# when a Python file is executed, but not when it is imported as a
# module. You can use this part of the file to test your code and run
# the measure_time function. We will not grade this part of the code.
if __name__ == "__main__":
    measure_time()
