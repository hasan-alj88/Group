import itertools
from collections import Counter
from math import factorial
from random import randint
from typing import List, Tuple, Self

from pydantic import BaseModel, Field, ConfigDict


class Permutation(BaseModel):
    """
    This object holds the permutation of the array [0, 1, 2, 3, ... (n-1)]
    where 'n' is the array length
    """
    swaps: List[Tuple[int, int]] = Field(default=None, alias="Swaps")

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True
    )

    def model_post_init(self, __context) -> None:
        if self.swaps is None:
            self.swaps = []

    def __len__(self) -> int:
        return len(self.permuted_array)

    def __str__(self) -> str:
        cycles_list = self.cycles
        if not cycles_list:
            return '()'

        return ''.join(
            f"({','.join(map(str, cycle))})"
            for cycle in cycles_list
        )

    def __repr__(self) -> str:
        return f"Permutation[{self.__str__()}]"

    def __hash__(self) -> int:
        return hash(str(self))

    def __add__(self, other: Self) -> Self:
        combination = Permutation(swaps=self.swaps + other.swaps)
        return Permutation.of_the_array(combination.permuted_array)

    def __sub__(self, other: Self) -> Self:
        return self + other.inv

    def __eq__(self, other: Self) -> bool:
        return hash(self) == hash(other)

    @classmethod
    def of_the_array(cls, arr: List[int]) -> Self:
        """
        Permutation constructor by entering the permuted array
        :param arr: permuted array input
        [!] the array must be an ordering of the sequence 0,...,(n-1)
        :return: permutation object of the permuted array
        """
        arr = list(map(abs, arr))
        min_index = min(arr, default=0)
        # insert the missing elements
        if min_index > 0:
            arr = list(range(min_index)) + arr
        assert (Permutation.array_validation(arr))

        # check if the array is already in identity form
        if arr == list(range(len(arr))):
            return cls(swaps=[])

        swaps = []
        current = arr.copy()
        target = list(range(len(arr)))

        # Work backwards from arr to identity
        while current != target:
            for i in range(len(current)):
                if current[i] != target[i]:
                    # Find the position containing the number we want
                    j = current.index(target[i])
                    # Add the swap
                    swaps.append((min(i, j), max(i, j)))
                    # Apply the swap
                    current[i], current[j] = current[j], current[i]
                    break

        # Swaps were built going from arr to identity
        # So we need to reverse them to go from identity to arr
        swaps.reverse()
        return cls(swaps=swaps)

    @classmethod
    def of_cyclic_notation(cls, cyclic_input: str) -> Self:
        if cyclic_input == '()':
            return cls(swaps=[])

        # remove all kinds of white spaces
        cyclic_input = cyclic_input.replace(' ', '')

        # validate the cyclic notation (should only contain digits, commas and parentheses)
        if not all(c in '0123456789,()' for c in cyclic_input):
            raise ValueError(f'Invalid cyclic notation: {cyclic_input}')

        # replace ')(' with ; to split the cycles
        cyclic_input = cyclic_input.replace(')(', ';')

        # remove the parentheses
        cyclic_input = cyclic_input.replace('(', '').replace(')', '')

        # split the cycles
        cycles = cyclic_input.split(';')

        swaps = []
        for cycle in cycles:
            elements = [int(e) for e in cycle.split(',')]
            swap_pairs = cls.cycle2swaps(elements)
            swaps.extend(swap_pairs)

        return cls(swaps=swaps)

    @staticmethod
    def cycle2swaps(cycle: List[int]) -> List[Tuple[int, int]]:
        """Convert a cycle to a list of swaps"""
        def minmax(a, b):
            return min(a, b), max(a, b)

        if len(cycle) < 2:
            raise ValueError(f'Cycle must have at least 2 elements: {cycle}')
        else:
            swaps = [minmax(cycle[i], cycle[i + 1])
                     for i in range(len(cycle)-1)]
        return swaps

    @staticmethod
    def array_validation(arr: List[int]) -> bool:
        """Validate the index array"""
        for element in arr:
            if element not in range(len(arr)):
                raise ValueError(f'All elements must be integers and between 0 and (n-1).'
                                 f'\n{arr}\n{Counter(arr)}')

        if len(arr) > len(set(arr)):
            raise ValueError(f'All array elements must be Unique.\n{arr}\n{Counter(arr)}')
        return True

    def add_swap(self, a: int, b: int) -> None:
        """Add a swap to the permutation"""
        self.swaps.append((min(a, b), max(a, b)))

    @property
    def max_index(self) -> int:
        """Get the maximum index"""
        return max(map(max, self.swaps), default=0)

    @property
    def permuted_array(self) -> List[int]:
        """Get the permuted array"""
        if len(self.swaps) == 0:
            return []

        arr = self.permuted_array_of_length(self.max_index)

        # Find the effective size of the permutation
        effective_size = self.max_index + 1
        for i in range(len(arr) - 1, -1, -1):
            if arr[i] != i or i < effective_size:
                effective_size = i + 1
                break

        # Return the array up to the effective size
        result = arr[:effective_size]
        if len(result) < 2:
            return []
        self.array_validation(result)
        return result

    def permuted_array_of_length(self, array_size: int = 0) -> List[int]:
        """Get the permuted array of a specific length"""
        max_index = self.max_index
        if array_size == 0:
            array_size = max_index
        if array_size < max_index:
            raise ValueError(f'Permuted array size must be >= {max_index}')

        # Start with identity permutation
        arr = list(range(array_size + 1))

        # Apply swaps to construct the permutation
        for i, j in self.swaps:
            arr = self.apply_swap(arr, i, j)

        return arr

    @property
    def cycles(self) -> List[List[int]]:
        """Get the cycles in the permutation"""
        cycles_list = []
        arr = self.permuted_array
        seen = set()

        # Iterate through array indices
        for x in range(len(arr)):
            # Skip if already seen or maps to itself
            if x in seen or x == arr[x]:
                continue

            cycle = []
            current = x
            # Follow the cycle until we return to start
            while current not in cycle:
                cycle.append(current)
                seen.add(current)
                current = arr[current]

            if len(cycle) > 1:
                cycles_list.append(cycle)

        # Sort the cycles by their minimum element
        cycles_list.sort(key=lambda c: min(c))

        # For each cycle, rotate until minimum element is first
        for c in cycles_list:
            min_element = min(c)
            while c[0] != min_element:
                c.append(c.pop(0))  # rotate left until min element is first

        return cycles_list

    @property
    def inv(self) -> Self:
        """Get the inverse permutation"""
        p = Permutation(swaps=[])
        arr = self.permuted_array.copy()

        for index1, element in enumerate(arr):
            if element == index1:
                continue
            index2 = arr.index(index1)
            p.add_swap(index1, index2)
            arr = self.apply_swap(arr, index1, index2)

        return p

    @staticmethod
    def rotate_array(arr: List) -> List:
        """Rotate an array"""
        return [arr[-1]] + arr[:-1]

    @staticmethod
    def apply_swap(arr: List[int], index1: int, index2: int) -> List[int]:
        """Apply a swap to an array"""
        arr_copy = arr.copy()
        arr_copy[index1], arr_copy[index2] = arr_copy[index2], arr_copy[index1]
        return arr_copy

    @staticmethod
    def generator(n: int):
        """Generate all permutations of size n"""
        arr = list(range(n))
        for p in itertools.permutations(arr):
            yield Permutation.of_the_array(list(p))

    @staticmethod
    def random(size: int) -> Self:
        """Generate a random permutation of given size"""
        n = randint(1, factorial(size))
        for ind, p in enumerate(Permutation.generator(size)):
            if ind == n:
                return p

if __name__ == '__main__':
    arr = [1, 0, 2, 4, 3]
    p = Permutation.of_the_array(arr=arr)
    print(f'p = {p}')
    print(f'permuted array: {p.permuted_array}')
    print(f'{p.permuted_array} == {arr} ... [{p.permuted_array == arr}]' )