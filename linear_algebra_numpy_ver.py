"""
In this file I will be recreating the funcitons that we created to do linear 
algebra, but instead of using base python objects like lists I will be using
numpy. This will increase performance and make the functions operate smoother
when used in machine learning applications. 

Insert a quick word on Numpy here:
"""

import numpy as np
from typing import List

Vector = np.array(List[float])

# Define vectors for testing our functions
foo = np.array([1,2,3])
bar = np.array([1,2,3])

def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements within np.arrays designated as vectors."""
    assert v.size == w.size, "vectors must be the smae length"
    return np.array([v_i + w_i for v_i, w_i in zip(v,w)])

assert add(foo, bar).all() == np.array([2,4,6]).all()

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements within np.arrays desginated as vectors."""
    assert v.size == w.size, "vecotrs must be the same length."
    return np.array([
        v_i - w_i for v_i, w_i in zip(v, w)
    ])

assert subtract(foo, bar).all() == np.array([0,0,0]).all()

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"

    # Check the vectors are all the same size
    num_elements = vectors[0].size
    assert all(v.size == num_elements for v in vectors), "different sizes!"

    # the i-th element of the result is the sum every vector[i]
    return np.array([sum(vector[i] for vector in vectors)
            for i in range(num_elements)])

assert vector_sum(
    [
        np.array([1,2]), np.array([3,4]), np.array([5,6]), np.array([7,8])
    ]
).all() == np.array([16,20]).all()

def scalar_multiple(c: float, v: Vector) -> Vector:
    """Multiplies evert element in a vector by c"""
    return np.array(
        [c * v_i for v_i in v]
    )

assert scalar_multiple(2, foo).all() == np.array([2,4,6]).all()

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average of a list of vectors"""
    n = len(vectors)
    return np.array([scalar_multiple(1/n, vector_sum(vectors))])

assert vector_mean(
    [np.array([1,2]), np.array([3,4]), np.array([5,6])]
).all() == np.array([3,4]).all()

def dot(v: Vector, w: Vector) -> float:
    """
    Computes v_1 * w_1 + ... + v_i * w_i

    The dot product computes how far the v vector extends in the direction of 
    the w vector. Basically the length of the vector we would get when
    projecting v onto w.
    """
    assert v.size == w.size, "vectors must be the same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot(np.array([1,2,3]), np.array([4,5,6])) == 32 # 1 * 4 + 2 * 5 + 3 * 6 = 32

def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_i * v_i"""
    return dot(v, v)

assert sum_of_squares(np.array([1,2,3])) == 14 # 1*1 + 2*2 + 3*3

import math

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))

assert magnitude(np.array([3, 4])) == 5

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

assert distance(foo, bar) == 0