from functools import reduce
from itertools import product
from typing import Set

import numpy as np
from pydantic.v1 import BaseModel


class Quaternion(BaseModel):
    a: float
    b: float
    c: float
    d: float

    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(
            a=self.a + other.a,
            b=self.b + other.b,
            c=self.c + other.c,
            d=self.d + other.d
        )

    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(
            a=self.a - other.a,
            b=self.b - other.b,
            c=self.c - other.c,
            d=self.d - other.d
        )

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(
            a=self.a * other.a - self.b * other.b - self.c * other.c - self.d * other.d,
            b=self.a * other.b + self.b * other.a + self.c * other.d - self.d * other.c,
            c=self.a * other.c - self.b * other.d + self.c * other.a + self.d * other.b,
            d=self.a * other.d + self.b * other.c - self.c * other.b + self.d * other.a
        )

    def __truediv__(self, other: 'Quaternion') -> 'Quaternion':
        return self * other.inverse()

    def __str__(self) -> str:
        return f'{self.a} + {self.b}i + {self.c}j + {self.d}k'

    def __repr__(self) -> str:
        return f'Quaternion(a={self.a}, b={self.b}, c={self.c}, d={self.d})'

    def __eq__(self, other: 'Quaternion', tol=1e-10) -> bool:
        return (abs(self.a - other.a) < tol and
                abs(self.b - other.b) < tol and
                abs(self.c - other.c) < tol and
                abs(self.d - other.d) < tol)

    def __ne__(self, other: 'Quaternion') -> bool:
        return not self.__eq__(other)

    def __neg__(self) -> 'Quaternion':
        return Quaternion(
            a=-self.a,
            b=-self.b,
            c=-self.c,
            d=-self.d
        )

    def __pow__(self, power: int) -> 'Quaternion':
        if power == 0:
            return Quaternion(a=1, b=0, c=0, d=0)
        if power < 0:
            return self.inverse() ** -power
        return reduce(lambda x, y: x * y, [self for _ in range(power)])

    def __hash__(self):
        return hash((self.a, self.b, self.c, self.d))

    def __abs__(self):
        return self.norm()

    def __round__(self, n=None):
        return Quaternion(
            a=round(self.a, n),
            b=round(self.b, n),
            c=round(self.c, n),
            d=round(self.d, n)
        )

    def norm(self) -> float:
        return (self.a ** 2 + self.b ** 2 + self.c ** 2 + self.d ** 2) ** 0.5

    def inverse(self) -> 'Quaternion':
        norm = self.norm() ** 2
        return Quaternion(
            a=self.a / norm,
            b=-self.b / norm,
            c=-self.c / norm,
            d=-self.d / norm
        )

    def conjugate(self) -> 'Quaternion':
        return Quaternion(
            a=self.a,
            b=-self.b,
            c=-self.c,
            d=-self.d
        )



    @staticmethod
    def generator(n: int) -> Set['Quaternion']:
        """
        Generate the set of elements of the quaternion group Q_{2^n}.
        This static method computes rotation quaternions corresponding to 
        evenly spaced angles for rotations around the i, j, and k axes, 
        based on the input parameter `n`. The set of rotations forms the 
        quaternion group Q_{2^n}.
        
        :param n: Number of rotations to generate.
        :return: Set of rotations.
        """
        if n < 3:
            raise ValueError("n must be >= 3")

        expected_size = 2 ** (n + 1)  # Order of Q_{2^n}
        step = 2 * np.pi / (2 ** (n - 2))
        elements = set()

        # Generate base rotations
        for k in range(2 ** (n - 2)):
            angle = k * step
            h = angle / 2
            cos_h = np.cos(h)
            sin_h = np.sin(h)

            rotations = [
                Quaternion(a=cos_h, b=sin_h, c=0, d=0),
                Quaternion(a=cos_h, b=0, c=sin_h, d=0),
                Quaternion(a=cos_h, b=0, c=0, d=sin_h)
            ]
            elements.update(rotations)

        # Add products and inverses until we reach the expected size
        while len(elements) < expected_size:
            products = {p1 * p2 for p1, p2 in product(elements, elements)}
            inverses = {q.inverse() for q in elements}

            # Safely update the element set
            elements.update(products)
            elements.update(inverses)
            print(f'elements size: {len(elements)}')

        return {round(q, 10) for q in elements}



if __name__ == '__main__':
    # Create quaternions
    q1 = Quaternion(a=1, b=2, c=3, d=4)
    q2 = Quaternion(a=2, b=3, c=4, d=5)

    # Test operations
    assert q1 + q2 == Quaternion(a=3, b=5, c=7, d=9)
    assert q1 * q2 != q2 * q1  # Non-commutative
    assert q1 * q1.inverse() == Quaternion(a=1, b=0, c=0, d=0)  # Identity
    assert (q1 * q2).conjugate() == q2.conjugate() * q1.conjugate()  # Conjugate property

    # Test generator
    for i, element in enumerate(Quaternion.generator(4)):
        print(f'Element {i}: {element}')