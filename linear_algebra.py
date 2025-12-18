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