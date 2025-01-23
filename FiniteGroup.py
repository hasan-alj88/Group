from functools import cached_property, partial
from itertools import product
from typing import TypeVar, List, Callable, Set, Dict, Tuple

import numpy as np
from joblib import Parallel, delayed, Memory
from numpy.typing import NDArray
from pydantic.v1 import BaseModel, Field, validator

from GroupFamily import GroupFamily
from MathLib.MatrixLib import generate_gl_elements, generate_sl_elements, sort_2d_array, generate_orthogonal_group, \
    matrix_mult, generate_unitary_group
from MathLib.PrimeLib import powerset_unions, is_prime_number, is_factorial
from MathLib.Quaternion import Quaternion
from Permutation import Permutation
from log_configuration import log_decorator, create_logger

logger = create_logger(__name__)

T = TypeVar('T')


class FiniteGroup(BaseModel):
    """A mathematical group represented by its Cayley table with identity at index 0."""

    cayley_table: NDArray[int] = Field(
        description="Cayley table representing the group operation"
    )

    class Config:
        arbitrary_types_allowed = True  # Allows numpy arrays
        validate_assignment = True      # Validate on assignment
        allow_mutation = False          # Make the model immutable
        keep_untouched = (cached_property,)  # This is the key addition!

    @validator('cayley_table')
    @log_decorator(logger)
    def validate_cayley_table(cls, v: NDArray[int]) -> NDArray[int]: # noqa
        """Validate the Cayley table satisfies group axioms and has identity at 0."""
        v = np.array(v).astype(int)
        logger.debug(f"Validating Cayley table:\n{v}")

        if v.ndim != 2:
            raise ValueError(f"Cayley table must be 2-dimensional. Got {v.ndim}-dimensional array")

        if v.shape[0] != v.shape[1]:
            raise ValueError(f"Cayley table must be square. Got shape {v.shape}")

        n = v.shape[0]
        elements = range(n)

        # Group Axiom #1 - Closure
        if not np.all((v >= 0) & (v < n)):
            logger.debug(f"Group Axiom violation (Closure): All elements must be integers in range [0, {n - 1}]")
            raise ValueError(
                f"Group Axiom violation (Closure): "
                f"All elements must be integers in range [0, {n - 1}]"
            )
        if not np.all(np.isin(v, elements)):
            invalid_element = v[~np.isin(v, elements)][0]
            errmsg = f"Group Axiom violation (Closure): Found element {invalid_element} not in group"
            logger.debug(errmsg)
            raise ValueError(errmsg)

        elif v.shape[0] != len(set(v.flatten())):
            errmsg = (f"Group Axiom violation (Closure): Cayley table must contain all group elements."
                      f" Expected {n} unique elements, got {len(set(v.flatten()))}\n first row: {v[0]}")
            logger.debug(errmsg)
            raise ValueError(errmsg)

        logger.debug(f"Group Axiom #1 (Closure) satisfied")

        # Group Axiom #2 - Identity
        # Verify identity is at 0,0
        if v[0, 0] != 0:
            errmsg = f"Group Axiom violation (Identity): Identity must be at index [0,0], got {v[0, 0]}"
            logger.debug(errmsg)
            raise ValueError(errmsg)
        v = sort_2d_array(v)
        if not np.array_equal(v[0], np.arange(n)):
            errmsg = f"Group Axiom violation (Identity): First row must be identity permutation [0,1,...,{n - 1}]."
            logger.debug(errmsg)
            raise ValueError(errmsg)

        if not np.array_equal(v[:, 0], np.arange(n)):
            errmsg = f"Group Axiom violation (Identity): First column must be identity permutation [0,1,...,{n - 1}]."
            logger.debug(errmsg)
            raise ValueError(errmsg)

        logger.debug(f"Group Axiom #2 (Identity) satisfied")

        # Group Axiom #3 - Associativity
        # Verify associativity (a*b)*c = a*(b*c)
        for a,b,c in product(elements, repeat=3):
            if v[v[a, b], c] != v[a, v[b, c]]:
                raise ValueError(
                    f"Group Axiom violation (Associativity): "
                    f"({a}*{b})*{c} = {v[v[a, b], c]} != {a}*({b}*{c}) = {v[a, v[b, c]]}"
                )
        logger.debug(f"Group Axiom #3 (Associativity) satisfied")

        # Group Axiom #4 - Inverse elements
        # Verify inverses exist (we already know e is at 0)
        # Every row and column must contain 0 exactly once
        has_right_inverse = np.any(v == 0, axis=1)
        has_left_inverse = np.any(v == 0, axis=0)

        if not np.all(has_right_inverse):
            first_violation = np.where(~has_right_inverse)[0][0]
            logger.debug(f"Group Axiom violation (Inverse): Element {first_violation} has no right inverse")
            raise ValueError(
                f"Group Axiom violation (Inverse): "
                f"Element {first_violation} has no right inverse"
            )

        if not np.all(has_left_inverse):
            first_violation = np.where(~has_left_inverse)[0][0]
            logger.debug(f"Group Axiom violation (Inverse): Element {first_violation} has no left inverse")
            raise ValueError(
                f"Group Axiom violation (Inverse): "
                f"Element {first_violation} has no left inverse"
            )

        logger.debug(f"Group Axiom #4 (Inverse) satisfied")
        return v

    @classmethod
    @log_decorator(logger)
    def from_binary_operation(
            cls,
            elements: List[T],
            operation: Callable[[T, T], T]
    ) -> 'FiniteGroup':
        """
        Create a group from an ordered list of elements and a binary operation.
        The first element (index 0) must be the identity element.

        :param elements: An ordered list of elements where elements[0] is the identity.
        :param operation: Binary operation that combines two elements.
        :return: New group instance.
        :raises GroupValidationError: If the elements and operation don't form a valid group.
        :raises ValueError: If the input list is empty.
        """
        if not elements:
            raise ValueError("Elements list cannot be empty")

        # Use from_binary_operation_numpy for numpy arrays elements
        if isinstance(elements[0], np.ndarray):
            return cls.from_binary_operation_numpy(elements, operation)

        n = len(elements)
        identity = elements[0]
        table = np.zeros((n, n), dtype=int)

        # Verify the designated identity element actually acts as identity
        for i, x in enumerate(elements):
            result_right = operation(x, identity)
            result_left = operation(identity, x)
            if result_right != x or result_left != x:
                logger.debug(f"Group Axiom violation (Identity): Element at index 0 "
                             f"is not identity: "
                             f"{identity}*{x} = {result_right}, {x}*{identity} = {result_left}, "
                             f"expected {x}")
                raise ValueError(
                    f"Group Axiom violation (Identity): "
                    f"Element at index 0 is not identity: {identity}*{x} = {result_right}, "
                    f"{x}*{identity} = {result_left}, expected {x}"
                )

        # Create the table using numpy operations where possible
        for i, j in product(range(n), repeat=2):
            result = operation(elements[i], elements[j])
            try:
                idx = elements.index(result)
            except ValueError:
                logger.debug(f"Group Axiom violation (Closure): "
                             f"Operation result "
                             f"{result} = {elements[i]} * {elements[j]} "
                             f"not in element list")
                raise ValueError(
                    f"Group Axiom violation (Closure): "
                    f"Operation result {result} = {elements[i]} * {elements[j]} "
                    f"not in element list"
                )
            table[i, j] = idx

        return cls(cayley_table=np.array(table, dtype=int))

    @classmethod
    def from_binary_operation_numpy(
            cls,
            elements: List[np.ndarray],
            operation: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> 'FiniteGroup':
        """
        Create a group from an ordered list of numpy arrays and a binary operation.
        The first element (index 0) must be the identity element.

        :param elements: An ordered list of numpy arrays where elements[0] is the identity.
        :param operation: Binary operation that combines two arrays.
        :return: New group instance.
        :raises GroupValidationError: If the elements and operation don't form a valid group.
        :raises ValueError: If the input list is empty.
        """
        if not elements:
            raise ValueError("Elements list cannot be empty")

        n = len(elements)
        identity = elements[0]
        table = np.zeros((n, n), dtype=int)

        def find_array_index(arr: np.ndarray) -> int:
            """Find the index of a numpy array in the element list using array equality."""
            for idx, elem in enumerate(elements):
                if np.array_equal(elem, arr):
                    return idx
            raise ValueError(f"Array {arr} not found in elements list")

        # Verify the designated identity element actually acts as identity
        for i, x in enumerate(elements):
            result_right = operation(x, identity)
            result_left = operation(identity, x)

            if not (np.array_equal(result_right, x) and np.array_equal(result_left, x)):
                error_msg = (
                    f"Group Axiom violation (Identity): Element at index 0 is not identity:\n"
                    f"For element at index {i}:\n"
                    f"identity         =\n{identity}\n\n"
                    f"{x} \n* identity =\n{result_right}\n, expected:\n{x}\n\n"
                    f"identity \n* {x} =\n{result_left}\n, expected\n{x}\n\n"
                )
                logger.debug(error_msg)
                raise ValueError(error_msg)

        # Create the Cayley table
        for i, j in product(range(n), repeat=2):
            try:
                result = operation(elements[i], elements[j])
                # Check if a result matches any existing element (using array equality)
                index = find_array_index(result)
                table[i, j] = index
            except ValueError as e:
                error_msg = (
                    f"Group Axiom violation (Closure): Operation result not in element list\n"
                    f"Elements[{i}] * Elements[{j}] = {result}\n"
                    f"Elements[{i}] = {elements[i]}\n"
                    f"Elements[{j}] = {elements[j]}"
                )
                logger.debug(error_msg)
                raise ValueError(error_msg) from e

        return cls(cayley_table=np.array(table, dtype=int))

    @classmethod
    def trivial(cls) -> 'FiniteGroup':
        """
        Create the trivial group with only the identity element.

        :return: Trivial group
        """
        logger.debug("Creating trivial group")
        return cls(cayley_table=np.array([[0]], dtype=int))


    @classmethod
    @log_decorator(logger)
    def cyclic(cls, n: int) -> 'FiniteGroup':
        """
        Create cyclic group of order n.

        :param n: Order of the cyclic group.
        :return: Cyclic group of order n
        :raises ValueError: If n < 1
        """
        logger.debug(f"Creating cyclic group of order {n}")
        if n < 1:
            raise ValueError("Group order must be positive")

        # A*B = (A + B) mod n
        table = np.add.outer(range(n), range(n)).astype(int) % n
        return cls(cayley_table=table)

    @classmethod
    def MultiplicativeMod(cls, n: int) -> 'FiniteGroup':
        """
            MultiplicativeMod class method for constructing a finite group with
            multiplicative modulo operation for integers.

            This method generates a finite group by defining a set of integers
            in the range 1 to n-1 (inclusive). Whereas an operation that computes
            the product of two numbers under modulo n. The resulting structure
            adheres to the group properties by employing a binary operation
            for closure within the specified set of elements.

            Args:
                n (int): A positive integer representing the modulus for the
                group operation and the range upper limit for constructing
                the group elements.

            Returns:
                FiniteGroup: An instance of FiniteGroup where the elements
                and the operation define a multiplicative modulo group.
        """
        return cls.from_binary_operation(
            elements=list(range(1, n)),
            operation=lambda x, y: x * y,
        )


    @classmethod
    @log_decorator(logger)
    def from_permutations(cls, permutations: List[Permutation]) -> 'FiniteGroup':
        """
        Create a group from a list of permutations.
        The first permutation must be the identity permutation.

        :param permutations: List of permutations where permutations[0] is identity
        :return: New group instance
        :raises GroupValidationError: If the permutations don't form a group
        :raises ValueError: If the input list is empty
        """
        if not permutations:
            raise ValueError("Permutations list cannot be empty")
        permutations_set_str = '['+'\n'.join(str(p) for p in permutations)+']'
        logger.debug(f"Creating group from permutations: {permutations_set_str}")

        if Permutation() not in permutations:
            logger.debug(f"Group Axiom violation (Identity): Identity permutation not found in list")
            raise ValueError(
                f"Group Axiom violation (Identity): "
                "Identity permutation not found in list"
            )

        # make Identity permutation the first element
        identity_index = permutations.index(Permutation())
        permutations[0], permutations[identity_index] = permutations[identity_index], permutations[0]

        return cls.from_binary_operation(
            elements=permutations,
            operation=lambda p, q: p * q
        )


    @classmethod
    @log_decorator(logger)
    def complete_the_permutation_group(cls, permutations: Set[Permutation]) -> 'FiniteGroup':
        """
        Complete the permutation group to a full symmetric group.
        This is done by ensuring closure under composition.

        :param permutations: Set of permutations to complete
        :return: a completed group
        """
        # Ensure identity is present
        if Permutation() not in permutations:
            permutations.add(Permutation())

        # Find all possible compositions
        new_permutations = set(permutations)
        while True:
            new_permutations_len = len(new_permutations)
            for p, q in product(new_permutations, repeat=2):
                new_permutations.add(p * q)
            if len(new_permutations) == new_permutations_len:
                break

        return cls.from_permutations(list(new_permutations))


    @classmethod
    @log_decorator(logger)
    def dihedral(cls, n: int) -> 'FiniteGroup':
        """
        Create a dihedral group D_n of order 2n.


        :param n: Parameter n in D_n (group will have order 2n)
        :return: Dihedral group D_n
        :raises ValueError: If n < 1
        """
        logger.debug(f"Creating dihedral group D_{n}")
        if n < 1:
            raise ValueError("Group order must be positive")
        elif n == 1:
            return cls.cyclic(2)  # D₁ ≅ C₂

        # Initialize Cayley table of size 2n x 2n
        size = 2 * n
        table = np.zeros((size, size), dtype=int)

        # Fill the table
        for i, j in product(range(size), repeat=2):
            if i < n and j < n:  # Rotation followed by rotation
                table[i, j] = (i + j) % n
            elif i < n <= j:  # Rotation followed by reflection
                table[i, j] = n + (j - n - i) % n
            elif i >= n > j:  # Reflection followed by rotation
                table[i, j] = n + (i - n + j) % n
            else:  # Reflection followed by reflection
                table[i, j] = (2 * n + (j - n) - (i - n)) % n

        return cls(cayley_table=table)


    @classmethod
    @log_decorator(logger)
    def symmetric(cls, n: int) -> 'FiniteGroup':
        """
        Create a symmetric group of order n!.

        :param n: Order of the symmetric group.
        :return: Symmetric group of order n!.
        :raises ValueError: If n < 1
        """
        if n < 1:
            raise ValueError("Group order must be positive")

        return cls.from_permutations(list(Permutation.generator(n)))


    @classmethod
    @log_decorator(logger)
    def alternating(cls, n: int) -> 'FiniteGroup':
        """
        Create an alternating group of order n.
        Group of even permutations in the symmetric group.

        :param n: Order of the alternating group.
        :return: Alternating group of order n.
        :raises ValueError: If n < 1
        """
        if n < 1:
            raise ValueError("Group order must be positive")

        return cls.from_permutations(
            [p for p in Permutation.generator(n) if p.is_even]
        )


    @classmethod
    @log_decorator(logger)
    def general_linear(cls, n: int, field_size: int) -> 'FiniteGroup':
        """
        Create general linear group GL(n,F) of invertible n×n matrices
        over finite field of given size.

        :param n: Matrix dimension.
        :param field_size: Size of finite field.
        :return: General linear group GL(n,F).
        :raises ValueError: If field_size is n < 1
        """

        if n < 1:
            raise ValueError("Matrix dimension must be positive")

        # Generate all possible matrices and filter for invertible ones
        elements = generate_gl_elements(n, field_size)

        return cls.from_binary_operation(
            elements=elements,
            operation=partial(matrix_mult, field_size=field_size)
        )


    @classmethod
    @log_decorator(logger)
    def special_linear(cls, n: int, field_size: int) -> 'FiniteGroup':
        """
        Create special linear group SL(n,F) of n×n matrices with
        determinant 1 over finite field of given size (must be prime).

        :param n: Matrix dimension.
        :param field_size: Size of finite field (must be prime).
        :return: Special linear group SL(n,F).
        :raises ValueError: If field_size is not prime or n < 1.
        """

        if n < 1:
            raise ValueError("Matrix dimension must be positive")
        if not is_prime_number(field_size):
            raise ValueError("Field size must be prime")

        # Generate matrices with determinant 1
        elements = generate_sl_elements(n, field_size)

        return cls.from_binary_operation(
            elements=elements,
            operation=partial(matrix_mult, field_size=field_size)
        )

    @classmethod
    def Orthogonal(cls, n: int, q: int) -> 'FiniteGroup':
        """
        Group of the elements of the finite orthogonal group O(n, F_q).
        These are n x n matrices preserving a given bi-linear form B over F_q.

        :param n: Matrix dimension.
        :param q: Order of the symmetric group.
        :return: Orthogonal group of n x n matrices.
        """
        elements = generate_orthogonal_group(n, q)
        return cls.from_binary_operation(
            elements=elements,
            operation=partial(matrix_mult, field_size=n)
        )
    
    @classmethod
    def Unitary(cls, n: int, q: int) -> 'FiniteGroup':
        """
        Create the Unitary group U(n, q), the set of n×n unitary matrices
        over a finite field of order q^2 that preserve a Hermitian form.
    
        The Unitary group is defined as the subgroup of GL(n, q^2) that satisfies
        M* H M = H, where H is a Hermitian matrix, M is a matrix from GL(n, q^2),
        and M* is the conjugate transpose of M.
    
        :param n: Dimension of the matrices.
        :param q: Order of the base field (q must be a prime power).
        :return: Unitary group U(n, q).
        :raises ValueError: If n < 1 or q is not a prime power.
        """
        elements = generate_unitary_group(n, q)
        return cls.from_binary_operation(
            elements=elements,
            operation=partial(matrix_mult, field_size=n)
        )
        

    @classmethod
    def Quaternion(cls, n: int) -> 'FiniteGroup':
        """
        Create a quaternion group of order 2^n.

        :param n: Order of the quaternion group.
        :return: Quaternion group of order 2^n.
        :raises ValueError: If n < 1
        """
        if n < 3:
            raise ValueError("Quaternion group requires n >= 3")

        elements = Quaternion.generator(n)
        elements = [Quaternion(a=1,b=0,c=0,d=0)] + list(elements - {Quaternion(a=1,b=0,c=0,d=0)})

        return cls.from_binary_operation(
            elements=elements,
            operation=lambda q1, q2: q1 * q2
        )


    def __len__(self) -> int:
        return int(self.cayley_table.shape[0])

    def __hash__(self):
        return hash(self.cayley_table.tobytes())

    def __getitem__(self, key: tuple[int, int]) -> int:
        """
        The Group operation.
        :param key: Tuple of two integers representing the indices of the elements.
        :return: The result of the group operation.
        :raises TypeError: If the index is not a tuple of two integers.
        :raises TypeError: If the index contains non-integer elements.
        :raises IndexError: If the index is out of range.
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Index must be a tuple of two integers")
        if not all(isinstance(i, int) for i in key):
            raise TypeError(f"Index must be a tuple of two integers. Got {[type(i) for i in key]}")
        if not all(0 <= i < self.order for i in key):
            raise IndexError(f"Index out of range for group of order {self.order}."
                           f"Got {key}, expected range [0, {self.order}]")
        a, b = key
        return int(self.cayley_table[a, b])

    def __mul__(self, H: 'FiniteGroup') -> 'FiniteGroup':
        """
        Compute the direct product of this group with another group.
        :param H: The other group to compute the direct product with.
        :return: Direct product group.
        """
        return self.direct_product(H)

    def __truediv__(self, N: 'FiniteGroup') -> 'FiniteGroup':
        """
        Compute the factor group G/N given a normal subgroup N.
        :param N: The normal subgroup to factor out.
        :return: Factor group G/N.
        :raises ValueError: If the subgroup is not normal.
        """
        return self.factor_group(N)

    def __eq__(self, other: 'FiniteGroup') -> bool:
        """
        Group equality mathematically means the groups are isomorphic.
        :param other: The other group to compare with.
        :return: True if the groups are isomorphic.
        :raises TypeError: If the other object is not a FiniteGroup.
        """
        if not isinstance(other, FiniteGroup):
            raise TypeError(f"Cannot compare FiniteGroup with {type(other)}")
        return self.is_isomorphic(other)

    @cached_property
    def order(self) -> int:
        """Order of the group."""
        return len(self)

    @cached_property
    def elements(self) -> Set[int]:
        """Set of elements in the group."""
        return set(range(self.order))


    @cached_property
    @log_decorator(logger)
    def _inverses(self) -> Dict[int, int]:
        """Cached array of inverse elements."""
        # Since identity is at 0, find indices where multiplication gives 0
        inverse_mask = self.cayley_table == 0
        # Convert to an int type and get just the array of indices
        inv_indices = np.where(inverse_mask)[1]
        # Create dictionary with explicit integer types
        return {i: int(inv) for i, inv in enumerate(inv_indices)}

    def inverse(self, element: int) -> int:
        """Find the inverse of an element."""
        if not 0 <= element < self.order:
            raise IndexError(f"Element {element} not in group of order {self.order}")
        return int(self._inverses[element])


    @cached_property
    def center(self) -> Set[int]:
        """Center of the group."""
        is_central = np.all(self.cayley_table == self.cayley_table.T, axis=1)
        return set(np.where(is_central)[0])
    
    @cached_property
    def center_group(self) -> 'FiniteGroup':
        
        if len(self.center) < 2:
            return self.trivial()
        
        if self.is_subgroup_elements(self.center):
            return self.form_a_subgroup(self.center)
        raise ValueError(f'{self.center} is not a subgroup of G (|G|={self.order})')

    @cached_property
    def is_abelian(self) -> bool:
        """Whether the group is abelian."""
        return np.array_equal(self.cayley_table, self.cayley_table.T)

    @log_decorator(logger)
    def is_homomorphic(self, H: 'FiniteGroup', phi: Permutation) -> bool:
        """
        Check if the group is homomorphic to another group.
        G(x,*), H(y,#) are homomorphic if there exists a mapping φ: G → H
        such that φ(x * y) = φ(x) # φ(y) for all x,y in G.

        Edge cases:
        - Empty groups are only homomorphic to empty groups
        - Single element groups are homomorphic to any group (trivial homomorphism)

        :param H: The other group to check homomorphism against
        :param phi: The mapping between the groups
        :return: True if the group is homomorphic to the other group
        """
        # Handle empty groups
        if self.order < 1 and H.order < 1:
            return True

        # Handle single element groups (trivial case)
        if self.order == 1:
            return phi[0] == 0

        # identity must be preserved
        if phi[0] != 0:
            return False

        # Check if the mapping preserves the group operation
        for x, y in product(range(self.order), repeat=2):
            if phi[self[x, y]] != H[phi[x], phi[y]]:
                return False
        return True

    @log_decorator(logger)
    def is_isomorphic(self, H: 'FiniteGroup') -> bool:
        """
        Check if the group is isomorphic to another group.
        G(x,*), H(y,#) are isomorphic if there exists a bijective mapping phi: G → H
        such that phi(x * y) = phi(x) # phi(y) for all x, y in G.

        Edge cases:
        - Empty groups are only isomorphic to empty groups
        - Single element groups are only isomorphic to single element groups

        :param H: The other group to check isomorphism against.
        :return: True if the group is isomorphic to the other group.
        """
        if self.order != H.order:
            return False
        elif self.order in {0, 1}:
            return True

        # Check for isomorphism by trying all permutations
        for phi in Permutation.generator(self.order):
            if self.is_homomorphic(H, phi):
                return True
        return False

    @cached_property
    def family(self) -> List[GroupFamily]:
        if self.order < 1:
            return [GroupFamily.Trivial]

        family = []
        # cyclic group
        if self.cyclic(self.order).is_isomorphic(self):
            family.append(GroupFamily.Trivial)

        if self.order % 2 == 0:
            if self.dihedral(self.order // 2).is_isomorphic(self):
                family.append(GroupFamily.Dihedral)

        if self.is_abelian:
            family.append(GroupFamily.Abelian)

        if self.is_solvable:
            family.append(GroupFamily.Solvable)


        if is_factorial(self.order):
            if self.symmetric(self.order).is_isomorphic(self):
                family.append(GroupFamily.Symmetric)
        if not family:
            family.append(GroupFamily.Special)

        return family



    @cached_property
    @log_decorator(logger)
    def automorphism_mappings(self) -> Set[Permutation]:
        """
        The computationally expensive way to find all automorphisms of the group.
        Using parallel processing

        Edge cases:
        - an Empty group has no automorphisms (empty set)
        - a Single element group has only the identity automorphism

        :return: Automorphisms of the group.
        """
        # Handle an empty group
        if self.order == 0:
            return set()

        # Handle a single element group
        if self.order == 1:
            return {Permutation()}  # Identity permutation only

        memory = Memory(location='.cache', verbose=1)

        @memory.cache
        def is_valid_automorphism(perm):
            return self.is_homomorphic(self, phi=perm)

        automorphisms = Parallel(n_jobs=-1, prefer="threads")(
            delayed(is_valid_automorphism)(p)
            for p in Permutation.generator(self.order)
        )

        return automorphisms


    @cached_property
    @log_decorator(logger)
    def automorphism_group(self) -> 'FiniteGroup':
        """
        Compute the automorphism group of the group.

        Edge cases:
        - an Empty group has empty automorphism group
        - a Single element group has a single element automorphism group

        :return: Automorphism group of the group.
        """
        if self.order in {0, 1, 2}:
            return self

        if not self.is_simple:
            # Aut(G) = Aut(N) x Aut(G/N)
            N = sorted(self.proper_normal_subgroups, key=len, reverse=True)[0]
            Q = self / N
            return N.automorphism_group * Q.automorphism_group

        if GroupFamily.Dihedral in self.family:
            n = self.order // 2
            if is_prime_number(n):
                return self.dihedral(n)
            return self.cyclic(2) * self.cyclic(n)

        if GroupFamily.Cyclic in self.family:
            return FiniteGroup.MultiplicativeMod(n=self.order)


        # if there are no special cases do the computationally expensive thing and final all automorphisms
        return FiniteGroup.from_permutations(list(self.automorphism_mappings))

    @log_decorator(logger)
    def semi_direct_product(self, H: 'FiniteGroup', phi: Dict[int, Permutation]) -> 'FiniteGroup':
        """
        Compute the semi-direct product of this group with another group.
        Let 'self' be the group K(k,*) and h be the group H(h,#)
        with homomorphism φ: K → Aut(K).
        For groups G(g,*) and H(h,#) with homomorphism φ: H → Aut(G),
        G = K ⋊φ H = {(k,h) | k ∈ K, h ∈ H} with operation
        (k₁,h₁)·(k₂,h₂) = (k₁ * φ(h₁)(k₂), h₁#h₂)

        :param H: The group H to compute the semi-direct product with
        :param phi: The homomorphism φ mapping elements of H to automorphisms of G
        :return: Semi-direct product group G = K ⋊φ H
        :raises ValueError: If φ is not a valid homomorphism to Aut(G)
        """
        if not isinstance(phi, dict):
            raise TypeError(f"φ must be a dictionary of permutations. Got {type(phi)}")
        if all(not isinstance(p, Permutation) for p in phi.values()):
            raise TypeError(f"φ must be a dictionary of permutations. Got {[type(p) for p in phi.values()]}")

        # First check: Each φ(h) must be an automorphism of G
        aut_group = self.automorphism_group
        for h, phi_h in phi.items():
            if phi_h not in aut_group.permutation_elements:
                valid_automorphisms = '\n'.join(str(p) for p in aut_group.permutation_elements)
                phi_str = '\n'.join(f"φ({h}) = {phi_h}" for h, phi_h in phi.items())
                raise ValueError(f"φ must be a valid automorphism of K."
                                 f"\nValid automorphisms: \n{valid_automorphisms}"
                                 f"\nGot φ = \n{phi_str}")

        n, m = self.order, H.order
        elements = [(i, j) for i, j in product(range(n), range(m))]

        def pair_operation(pair1: tuple[int, int], pair2: tuple[int, int]) -> tuple[int, int]:
            """
            G = (K,*) ⋊φ (H,#) = (k₁,h₁)·(k₂,h₂) = (k₁ * φ(h₁)(k₂), h₁#h₂)
            """
            k1, h1 = pair1  # First element (k₁,h₁)
            k2, h2 = pair2  # Second element (k₂,h₂)

            # φ(h₁)
            phi_h1 = phi[h1]

            # apply φ(h₁) to k₂
            phi_h1_k2 = phi_h1[k2]

            if not 0 <= phi_h1_k2 < n:
                raise IndexError(f"φ({h1})({k2}) = {phi_h1_k2} not in group of order {n}."
                                 f"φ({h1}) = {phi_h1}, k₂ = {k2}")

            return (
                self[k1, phi_h1_k2],  # k₁ * φ(h₁)(k₂)
                H[h1, h2]  # h₁ # h₂
            )

        return FiniteGroup.from_binary_operation(
            elements=elements,
            operation=pair_operation
        )

    @log_decorator(logger)
    def direct_product(self, H: 'FiniteGroup') -> 'FiniteGroup':
        """
        Compute the direct product of this group with another group.
        For groups K and H, G = K × H = {(k,h) | k ∈ K, h ∈ H} with operation
        (kg₁,h₁)·(k₂,h₂) = (k₁*k₂, h₁#h₂)

        :param H: The other group to compute the direct product with.
        :return: Direct product group
        """
        return self.semi_direct_product(
            H=H,
            phi={i: Permutation() for i in range(self.order * H.order)} # trivial homomorphism
        )

    # @log_decorator(logger)
    def conjugate(self, g: int, h: int) -> int:
        """
        Find the conjugate of an element by another element.
        Operation g * h * g⁻¹
        :param g: Element to conjugate by.
        :param h: Element to conjugate.
        :return: Conjugate element.
        """
        return self[g, self[h, self.inverse(g)]]


    @cached_property
    @log_decorator(logger)
    def conjugacy_classes(self) -> Set[frozenset[int]]:
        """Return the set of conjugacy classes of the group."""
        seen = set()
        conjugacy_classes_set = set()

        for h in self.elements:
            if h in seen or h == 0:
                continue

            # Calculate conjugacy class for element h
            conj_class = frozenset(self.conjugate(g,h) for g in self.elements)
            conjugacy_classes_set.add(conj_class)
            seen.update(conj_class)

        return conjugacy_classes_set


    @cached_property
    @log_decorator(logger)
    def orbits(self) -> Dict[int, List[int]]:
        """
        All orbits of the group as ordered lists showing powers of elements.
        X -> [e, x, x², ...]
        """
        seen = set()
        unique_orbits = []
        for element in range(self.order):
            if element in seen:
                continue
            orbit = [0]  # Start with identity
            new_element = element
            while new_element != 0 or len(orbit) == 1:  # Continue until we get back to identity
                if new_element in orbit:
                    break
                orbit.append(new_element)
                new_element = self[new_element, element]
            seen.update(orbit)
            unique_orbits.append(orbit)
        orbits = sorted(unique_orbits, key=lambda x: x[0])  # Sort by first element (which is always 0)
        return dict(enumerate(orbits))
    
    @cached_property
    def generators(self) -> Set[frozenset[int]]:
        """
        Compute all possible generator sets of the group.
        A generator set forms the entire group when all combinations
        of its powers (closure) produce the group's elements.
    
        :return: Set of generator sets, each represented as a frozenset of orbit keys.
        """
        # Get the orbits as sets of elements
        orbit_elements = {k: set(v) for k, v in self.orbits.items()}
    
        # Check all combinations of the orbit sets
        generators = set()
        for subset_keys in powerset_unions(orbit_elements.keys()):
            # Union of the selected subsets
            combined_elements = set.union(
                *[orbit_elements[k] for k in subset_keys]
            )
            # Check if the combined set equals the group elements
            if combined_elements == self.elements:
                generators.add(frozenset(subset_keys))
        return generators
        

    @log_decorator(logger)
    def is_subgroup_elements(self, elements: Set[int]) -> bool:
        """Check if a set of elements forms a subgroup."""
        # must not be empty
        if not elements:
            return False

        # must contain identity
        if 0 not in elements:
            return False

        # must be a subset of the group elements
        if not set(elements).issubset(self.elements):
            return False

        # must be closed under group operation
        for a, b in product(elements, repeat=2):
            if self[a, b] not in elements:
                return False
        return True


    @cached_property
    @log_decorator(logger)
    def permutation_elements(self) -> List[Permutation]:
        """List of permutations corresponding to group elements."""
        return [Permutation.of_the_array(p) for p in self.cayley_table]

    @log_decorator(logger)
    def form_a_subgroup(self, elements: Set[int]) -> 'FiniteGroup':
        """Form a subgroup from a set of elements."""
        if not self.is_subgroup_elements(elements):
            raise ValueError(f"Input elements {elements} do not form a subgroup")
        return FiniteGroup.from_binary_operation(
            elements=[0]+list(elements - {0}),
            operation=lambda a, b: self[a, b]
        )


    @cached_property
    @log_decorator(logger)
    def proper_normal_subgroups_data(self) -> List[Tuple[set[int], 'FiniteGroup']]:

        normal_subgroups = []
        for normal_set in powerset_unions(self.conjugacy_classes):
            normal_set = set(normal_set).union({0})
            if len(normal_set) in [1, self.order]:
                continue
            if self.is_subgroup_elements(normal_set):
                normal_subgroups.append(
                    (set(normal_set),
                     self.form_a_subgroup(normal_set)
                    )
                )
        return normal_subgroups

    @cached_property
    def proper_normal_subgroups(self) -> Set['FiniteGroup']:
        return {sg[1] for sg in self.proper_normal_subgroups_data}


    @cached_property
    @log_decorator(logger)
    def is_simple(self) -> bool:
        """Check if the group is simple."""
        return len(self.proper_normal_subgroups) == 0


    @cached_property
    @log_decorator(logger)
    def is_solvable(self) -> bool:
        """Check if the group is solvable."""
        logger.debug(f"Checking if group is solvable:\n{self}")
        if self.order in [1, 2]:
            logger.debug("Groups of order 1 or 2 are solvable")
            return True
        if self.is_abelian:
            logger.debug("Abelian groups are solvable")
            return True
        for normal_subgroup in self.proper_normal_subgroups:
            if normal_subgroup.is_solvable:
                return True
        logger.debug(f"Group is [[NOT]] solvable\n{self}")
        return False

    @log_decorator(logger)
    def create_cosets(self, core_coset: List[int], left: bool=True) -> NDArray[int]:
        """
        Creates left co-sets mapping for a given co-set in a group.

        A left coset of a subgroup H in a group G is the set gH = {gh : h ∈ H} for g ∈ G.
        This method computes all left cosets for the given core_coset (subgroup).
        :param core_coset: List of elements forming a subgroup (H) of the main group.
        :param left: If True, create left cosets. If False, create right cosets.
        :return: A 2D array where each row represents a left coset.  The First row is the core_coset itself.
                         Shape is (|G|/|H|, |H|) where |G| is group order and |H| is subgroup order.
        :raises ValueError: If core_coset is not a valid subgroup
        :raises ValueError: If core_coset doesn't contain the identity element (0).
        :raises ValueError: If core_coset's order doesn't divide the group's order.
        :raises ValueError: If the resulting cosets do not partition the group.
        """

        # Input validation
        core_coset_set = set(core_coset)

        if not self.is_subgroup_elements(core_coset_set):
            raise ValueError(f"Core coset {core_coset} is not a subgroup")

        if 0 not in core_coset_set:
            raise ValueError("Core coset must contain the identity element (0)")

        group_order = len(self)
        subgroup_order = len(core_coset)

        if group_order % subgroup_order != 0:
            raise ValueError(
                f"Core coset order ({subgroup_order}) must divide group order ({group_order})"
            )

        # Initialize coset array
        num_cosets = group_order // subgroup_order
        left_cosets = np.full((num_cosets, subgroup_order), -1, dtype=int)

        # Normalize core_coset to always have identity first
        normalized_core_coset = [0] + sorted(list(core_coset_set - {0}))
        left_cosets[0] = normalized_core_coset

        # Keep track of used elements to ensure we get a partition
        used_elements: Set[int] = set(normalized_core_coset)
        current_coset = 1

        # Find remaining cosets
        for element in self.elements:
            if element in used_elements:
                continue

            # Compute new coset
            new_coset = [self[element, h] if left else self[h, element]
                         for h in normalized_core_coset]

            # Check for intersection with existing cosets
            new_elements = set(new_coset)
            if not new_elements.isdisjoint(used_elements):
                continue

            # Add new coset
            left_cosets[current_coset, :] = new_coset
            used_elements.update(new_elements)
            current_coset += 1

            if current_coset >= num_cosets:
                break

        # Verify we found all cosets
        if current_coset != num_cosets:
            raise ValueError(
                f"Failed to find all cosets. Found {current_coset} of {num_cosets} expected cosets"
            )

        # Verify we used all elements
        if len(used_elements) != group_order:
            raise ValueError(
                f"Cosets didn't form a partition. Used {len(used_elements)} of {group_order} elements"
            )
        return left_cosets

    def is_normal_subgroup(self, H: 'FiniteGroup') -> bool:
        """
        Check if this group is a normal subgroup of another group.

        :param H: The other group to check against.
        :return: True if this group is a normal subgroup of the other group.
        """
        if H.order > self.order:
            return False
        elif H.order % self.order != 0:
            return False
        elif H.order < 2:
            return True

        for sg in self.proper_normal_subgroups:
            if sg.is_isomorphic(H):
                return True
        return False

    def factor_group(self, normal_subgroup: 'FiniteGroup') -> 'FiniteGroup':
        """
        Compute the factor group G/N given a normal subgroup N.
        :param normal_subgroup: The normal subgroup to factor out.
        :return: Factor group G/N.
        :raises ValueError: If the subgroup is not a normal subgroup.
        """
        if normal_subgroup.order <2 :
            return self

        # find normal group subset elements
        for sg_elements, sg in self.proper_normal_subgroups_data:
            if sg.is_isomorphic(normal_subgroup):
                normal_subgroup_elements = sg_elements
                break
        else:
            raise ValueError(f"Subgroup is not a normal subgroup."
                             f"\nGroup:\n {self}\nSubgroup:\n{normal_subgroup}")
        # Get cosets
        cosets = self.create_cosets(list(normal_subgroup_elements))

        def coset_operation(a: int, b: int) -> int:
            # Get representatives of cosets
            a_coset_element = int(cosets[a][0])
            b_coset_element = int(cosets[b][0])

            # Multiply any elements from the coset
            # Result will be in the same coset regardless of choice
            ab = self[a_coset_element, b_coset_element]

            # Find which coset contains the product
            for i, coset in enumerate(cosets):
                if ab in coset:
                    return i

            raise ValueError("Product not found in any coset")

        # Create a factor group with coset indices as elements
        return FiniteGroup.from_binary_operation(
            elements=list(range(len(cosets))),
            operation=coset_operation
        )



    @log_decorator(logger)
    def plot(self):
        """Plot the group as a Cayley table."""
        import matplotlib.pyplot as plt
        plt.imshow(self.cayley_table, cmap='YlGnBu_r', interpolation='nearest')
        plt.colorbar()
        plt.title(f'{self.order}-element group')
        plt.show()

