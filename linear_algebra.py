from typing import List

Vector = List[float]

height_weight_age = [70,    # inches
                     170,   # pounds
                     40]    # years

grades = [95,   # exam1
          80,   # exam2
          70,   # exam3
          62 ]  # exam4

def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements."""
    assert len(v) == len(w), "vectors must the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements."""
    assert len(v) == len(w), "vectors must the same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"\
    
    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]

def scalar_multiple(c: float, v: Vector) -> Vector:
    """Multiplies every element in a vector by c"""
    return [c * v_i for v_i in v]

assert scalar_multiple(2, [1, 2, 3]) == [2, 4, 6]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise avearage of a list of vectors"""
    n = len(vectors)
    return scalar_multiple(1/n, vector_sum(vectors))

assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]

def dot(v: Vector, w: Vector) -> float:
    """
    Computes v_1 * w_1 + .... + v_i * w_i

    The dot product computes how far the v vector extends in the direction of 
    the w vector. Basically the length of the vector we would get when
    projecting v onto w.
    """
    assert len(v) == len(w), "vectors must be the same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1,2,3], [4,5,6]) == 32 # 1 * 4 + 2 * 5 + 3 * 6 = 32

def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_i * v_i"""
    return dot(v, v)

assert sum_of_squares([1,2,3]) == 14    # 1*1 + 2*2 + 3*3

import math

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))   # math.sqrt is square root function

assert magnitude([3, 4]) == 5

# We have now have all the tools to compute the distance between two vectors

# def squared_distance(v: Vector, w: Vector) -> float:
#     """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
#     return sum_of_squares(subtract(v, w))

# def distance(v: Vector, w: Vector) -> float:
#     """Computes the distance between v and w"""
#     return math.sqrt(squared_distance(v, w))

# Can also write the distance function as 
def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))

#Another type alias
Matrix = List[List[float]]

A = [[1,2,3],
     [3,4,5]]
B = [[1,2],
     [3,4],
     [5,6]]

# Print the shape of the matrix by using len(A) and len(A[0])

print(f"Shape: {len(A)}, {len(A[0])}")

# Going against mathmatical convention in the naming of rows and columns
# usually start with one but because python is zero indexing we start at 0

from typing import Tuple

def shape(A: Matrix) -> Tuple:
    """Returns (# of rows, # of columns)"""
    num_rows = len(A)
    num_columns = len(A[0]) # number of elements in first row
    return num_rows, num_columns

assert shape(A) == (2, 3)

def get_row(A: Matrix, j: int) -> Vector:
    """Retuns the i-th row of A (as a Vector)"""
    return [A_i[j]          # jth element of row A_i
            for A_i in A]   # for each row A_i

from typing import Callable

def make_matrix(num_rows: int,
                num_columns: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)
    """

    return [[entry_fn(i, j)                 # given i create a list
             for j in range(num_columns)]   # [entry_fn(i, 0), ...]
             for i in range(num_rows)]      # create one list for each i

def identity_matrix(n: int) -> Matrix:
    """Returns a n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]