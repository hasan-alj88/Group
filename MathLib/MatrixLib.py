from typing import List

import numpy as np
from itertools import product

from numpy.typing import NDArray


def finite_field_elements(q) -> list:
    """
    Generates elements of the finite field F_q (assuming integers mod q).
    For q as prime, these are simply {0, 1, ..., q-1}.
    :param q: The size of the finite field.
    :return: A list of elements in the finite field.
    """
    return list(range(q))


def is_invertible(matrix: NDArray, q: int) -> bool:
    """
    Checks if a matrix is invertible over F_q.
    A matrix is invertible if its determinant (mod q) is non-zero.
    :param matrix: The matrix to check.
    :param q: The size of the finite field.
    :return: True if the matrix is invertible, False otherwise
    """
    det = round(np.linalg.det(matrix))  # Calculate determinant
    return det % q != 0  # Check if determinant is non-zero mod q

def ensure_identity_matrix_at_first_position(matrix_list: List[NDArray]) -> List[NDArray]:
    # find the index of the identity matrix
    identity_matrix = np.eye(matrix_list[0].shape[0], dtype=int)
    identity_index = [i for i, matrix in enumerate(matrix_list) if np.array_equal(matrix, identity_matrix)][0]
    # swap the identity matrix with the first position
    matrix_list[0], matrix_list[identity_index] = matrix_list[identity_index], matrix_list[0]
    return matrix_list


def generate_gl_elements(n: int, q: int)-> List[NDArray]:
    """
    Generates all invertible n x n matrices over the finite field F_q.
    :param n: The size of the matrix.
    :param q: The size of the finite field.
    :return: A list of invertible n x n matrices.
    """
    field_elements = finite_field_elements(q)
    all_matrices = product(field_elements, repeat=n * n)  # All n x n matrices
    gl_elements = []

    for matrix_values in all_matrices:
        # Convert to an n x n matrix
        matrix = np.array(matrix_values).reshape((n, n))
        if is_invertible(matrix, q):
            gl_elements.append(matrix)

    return ensure_identity_matrix_at_first_position(gl_elements)


def generate_sl_elements(n: int, q: int) -> List[NDArray]:
    """
    Generates all n x n matrices with determinant 1 over the finite field F_q.
    These matrices form the Special Linear Group SL(n,F_q).

    :param n: The size of the matrices (nxn)
    :param q: The size of the finite field
    :return: A list of matrices in SL(n,F_q)
    """
    field_elements = finite_field_elements(q)
    all_matrices = product(field_elements, repeat=n * n)  # All n x n matrices
    sl_elements = []

    for matrix_values in all_matrices:
        # Convert to an n x n matrix
        matrix = np.array(matrix_values).reshape((n, n))
        # Calculate determinant and reduce modulo q
        det = round(np.linalg.det(matrix)) % q
        # Check if determinant is 1 (mod q)
        if det == 1:
            sl_elements.append(matrix)

    return ensure_identity_matrix_at_first_position(sl_elements)


def generate_orthogonal_group(n: int, q: int, bilinear_form: NDArray = None) -> List[NDArray]:
    """
    Generates the elements of the finite orthogonal group O(n, F_q).
    These are n x n matrices preserving a given bi-linear form B over F_q.

    :param n: The size of the matrices (nxn)
    :param q: The size of the finite field
    :param bilinear_form: The bi-linear form matrix B (default: identity matrix)
    :return: A list of matrices in O(n, F_q)
    """
    if bilinear_form is None:
        # Default bi-linear form is the identity matrix
        bilinear_form = np.eye(n, dtype=int)

    field_elements = finite_field_elements(q)
    all_matrices = product(field_elements, repeat=n * n)  # All n x n matrices
    orthogonal_elements = []

    for matrix_values in all_matrices:
        # Convert to an n x n matrix
        matrix = np.array(matrix_values).reshape((n, n))
        # Check if the orthogonality condition A^T B A = B is satisfied
        if np.array_equal(matrix.T @ bilinear_form @ matrix % q, bilinear_form % q):
            orthogonal_elements.append(matrix)

    return ensure_identity_matrix_at_first_position(orthogonal_elements)


def matrix_mult(A: np.ndarray, B: np.ndarray, field_size:int) -> np.ndarray:
    return (A @ B) % field_size


def generate_unitary_group(n: int, q: int, hermitian_form: NDArray = None) -> List[NDArray]:
    """
    Generates the elements of the finite unitary group U(n, F_{q^2}).
    These are n x n matrices that preserve the Hermitian form H over F_{q^2}.

    :param n: The size of the matrices (nxn).
    :param q: The base size of the finite field (q^2 is the extension field size).
    :param hermitian_form: The Hermitian form matrix H (default: identity matrix).
    :return: A list of matrices in U(n, F_{q^2}).
    """
    if hermitian_form is None:
        # Default Hermitian form is the identity matrix
        hermitian_form = np.eye(n, dtype=int)

    # Generate all elements of F_{q^2}
    field_elements = finite_field_elements(q)
    field_extension = [
        a + b * q for a, b in product(field_elements, repeat=2)
    ]  # Generate F_{q^2} as {a + b * sqrt(q)}

    def conjugate(element):
        """
        Computes the conjugate of an element in F_{q^2}.
        For F_{q^2}, the conjugate of (a + b * sqrt(q)) is (a - b * sqrt(q)).
        """
        a, b = divmod(element, q)
        return a - b * q

    def is_unitary(matrix: NDArray) -> bool:
        """
        Checks if a matrix satisfies the unitary condition A^dagger H A = H.
        """
        conjugate_transpose = np.vectorize(conjugate)(matrix.T)
        return np.array_equal(
            conjugate_transpose @ hermitian_form @ matrix % q, hermitian_form % q
        )

    # Generate all possible n x n matrices over F_{q^2}
    all_matrices = product(field_extension, repeat=n * n)
    unitary_elements = []

    for matrix_values in all_matrices:
        matrix = np.array(matrix_values).reshape((n, n))
        if is_unitary(matrix):
            unitary_elements.append(matrix)

    return ensure_identity_matrix_at_first_position(unitary_elements)


def sort_2d_array(arr: NDArray) -> NDArray:
    assert arr[0, 0] == 0  # Validate constraint
    # Sort columns based on first row (excluding first column)
    col_idx = np.concatenate(([0], 1 + np.argsort(arr[0, 1:])))
    arr = arr[:, col_idx]

    # Sort rows based on first column (excluding first row)
    row_idx = np.concatenate(([0], 1 + np.argsort(arr[1:, 0])))
    arr = arr[row_idx]
    return arr