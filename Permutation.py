import itertools
from collections import Counter
from math import factorial
from random import randint
from typing import List, Tuple, Self
from functools import cached_property
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
        return self.permuted_array == other.permuted_array

    def __getitem__(self, item) -> int:
        assert isinstance(item, int), f'Index must be an integer, not {type(item)}'
        try:
            return self.permuted_array[item]
        except IndexError:
            return item

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

    @cached_property
    def max_index(self) -> int:
        """Get the maximum index"""
        return max(map(max, self.swaps), default=0)

    @cached_property
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
            arr[i], arr[j] = arr[j], arr[i]

        return arr

    @cached_property
    def cycles(self) -> List[List[int]]:
        """Get the cycles in the permutation using improved algorithm"""
        arr = self.permuted_array
        if not arr:
            return []

        cycles_list = []
        seen = set()

        # Use array indexing instead of the following cycles
        next_indices = {i: arr[i] for i in range(len(arr))}

        for start in range(len(arr)):
            if start in seen or next_indices[start] == start:
                continue

            cycle = []
            current = start
            while current not in seen:
                cycle.append(current)
                seen.add(current)
                current = next_indices[current]

            if len(cycle) > 1:
                # Rotate to minimum element
                min_pos = cycle.index(min(cycle))
                cycle = cycle[min_pos:] + cycle[:min_pos]
                cycles_list.append(cycle)

        return sorted(cycles_list, key=lambda c: c[0])

    @cached_property
    def inv(self) -> Self:
        """Get the inverse permutation"""
        p = Permutation(swaps=[])
        arr = self.permuted_array.copy()

        for index1, element in enumerate(arr):
            if element == index1:
                continue
            index2 = arr.index(index1)
            p.add_swap(index1, index2)
            arr[index1], arr[index2] = arr[index2], arr[index1]

        return p

    @cached_property
    def shifted_by(self) -> int:
        """Get the shift of the permutation"""
        return min(map(min, self.swaps), default=0)

    @cached_property
    def shift(self, shift_by: int) -> Self:
        """Shift the permutation by a given number"""
        swaps = [(a+shift_by, b+shift_by) for a, b in
                 self.swaps]
        return Permutation(swaps=swaps)

    @cached_property
    def standard_form(self) -> Self:
        """Get the standard form of the permutation"""
        return self.shift(-self.shifted_by)

    @staticmethod
    def generator(n: int):
        """Generate all permutations of size n"""
        arr = list(range(n))
        for p in itertools.permutations(arr):
            yield Permutation.of_the_array(list(p))

    @staticmethod
    def random(size: int) -> Self:
        """Generate a random permutation of given size using Fisher-Yates shuffle"""
        arr = list(range(size))
        for i in range(size - 1, 0, -1):
            j = randint(0, i)
            arr[i], arr[j] = arr[j], arr[i]

        if arr == list(range(size)):
            return Permutation(swaps=[])
        elif arr[-1] == size-1:
            # if the last element is in its place, swap it with random element
            j = randint(0, size-2)
            arr[-1], arr[j] = arr[j], arr[-1]

        return Permutation.of_the_array(arr)

if __name__ == '__main__':
    arr = [1, 0, 2, 4, 3]
    p = Permutation.of_the_array(arr=arr)
    print(f'p = {p}')
    print(f'permuted array: {p.permuted_array}')
    print(f'{p.permuted_array} == {arr} ... [{p.permuted_array == arr}]' )