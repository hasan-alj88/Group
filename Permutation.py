import itertools
from collections import Counter
from math import factorial
from random import randint
# from StringIO import StringIO # if sys.version_info[0] < 3
from typing import TypeVar, List

import pandas as pd

Permutation_Type = TypeVar('Permutation_Type', bound='Permutation')


# from log_configuration import logger, log_decorator


class Permutation(object):
    """
    This object holds the permutation of the array of
    [0 ,1 ,2 ,3 ,... (n-1)]
    where 'n' is the array length
    """

    def __init__(self):
        self.Swaps = list()

    def __len__(self):
        return max(map(max, self.Swaps)) if len(self.Swaps) > 0 else 0

    def __str__(self):
        string = ''
        for Cycle in self.cycles:
            string += '('
            for index in Cycle:
                if string[-1] != '(':
                    string += ' ,'
                string += str(index)
            string += ')'
        return string if len(string) > 0 else '()'

    def __add__(self, other: Permutation_Type) -> Permutation_Type:
        combination = Permutation()
        combination.Swaps = self.Swaps + other.Swaps
        return Permutation.of_the_array(combination.permutated_array)

    def __sub__(self, other: Permutation_Type) -> Permutation_Type:
        return self + other.inv

    def __eq__(self, other: Permutation_Type) -> Permutation_Type:
        return self.permutated_array == other.permutated_array

    @classmethod
    # @log_decorator(logger)
    def of_the_array(cls, arr: List[int])-> Permutation_Type:
        """
        Permutation constructor by entering the permutated array
        :param arr: permutated array input
        [!] the array must be an ordering of the sequence 0,...,(n-1)
        :return: permutation object of the permutated array
        """
        p = cls()
        p.Swaps = list()
        assert (p.array_validation(arr))
        for index1, element in enumerate(arr):
            if element == index1:
                continue
            index2 = [i for i, x in enumerate(arr) if x == index1][0]
            p.add_swap(index1, index2)
            arr = Permutation.apply_swap(arr, index1, index2)
        p.Swaps.reverse()
        return p

    @classmethod
    # @log_decorator(logger)
    def of_input_array(cls, arr: list, sorted_arr: list)->Permutation_Type:
        """
        his will Create the permutation  object where its permutation is relative to a sorted array
        [!] All the elements must be present in both arrays.
            ie set(arr) intersection set(sorted_arr) == set(arr)
        :type sorted_arr: list
        :param arr: The disarranged array
        :param sorted_arr: The elements in a sorted order
        :return:
        """
        try:
            input_array = list(map(sorted_arr.index, arr))
        except ValueError as e:
            raise ValueError(str(e) + ' of elements which violate closure axiom.\nList elements are\n{}'
                             .format(sorted_arr))
        return cls.of_the_array(input_array)

    @classmethod
    def of_input_series_object(cls, series: pd.Series) -> Permutation_Type:
        series.replace(to_replace=series.index, value=list(series), inplace=True)
        return Permutation.of_input_array(list(series), series.index)

    @classmethod
    def of_cyclic_notation(cls, cyclic_input: str)->Permutation_Type:
        """
         This will create the permutation presented in the cyclic notation.
        :param cyclic_input: string of the cyclic notation.
        [!] it must be integers separated by commas ',' enclosed by brackets (0,8,5) for example.
        [i] The input can be many cycles (0,5,8)(10,2,9)...
        :return: Permutation object of the input Cyclic notation
        """

        p = Permutation()
        if cyclic_input == '()':
            return p
        cyclic_input = cyclic_input.strip().replace(' ', '')
        for cycle in cyclic_input.split(')('):
            cycle = cycle.replace('(', '')
            cycle = cycle.replace(')', '')
            indices = list(map(int, cycle.split(',')))
            for index1, index2 in zip(indices, indices[1:]):
                p.add_swap(index1, index2)
            p.Swaps.reverse()
            p = Permutation.of_the_array(p.permutated_array)
        return p

    def rotate(self, n: int=1)->Permutation_Type:
        """
        rotate the permutated array n times.
        [!] note that the size of the array is the largest permuted element.
        :param n:
        :return:
        """
        arr = []
        for i in range(n):
            arr = list(Permutation.rotate_array(self.permutated_array))
        return Permutation.of_the_array(arr)

    @staticmethod
    def rotate_array(arr: List)->List:
        return [arr[-1]] + arr[:-1]

    def apply(self, array: List)->List:
        for x, y in self.Swaps:
            array = Permutation.apply_swap(array, x, y)
        return array

    @staticmethod
    def array_validation(arr: List[int]) -> bool:
        """
        Test
        :param arr: Input array where :-
        All elements must be integers
        All elements must be between 0 and (n-1)
        :return:
        """
        for element in arr:
            if element not in range(len(arr)):
                raise ValueError('All elements must be integers and between 0 and (n-1)')
        # All array elements must be Unique
        if len(arr) > len(set(arr)):
            raise ValueError('All array elements must be Unique.\n' +
                             '{}\n{}'.format(arr, Counter(arr)))
        return True

    def add_swap(self, a: int, b: int):
        self.Swaps.append((min(a, b), max(a, b)))
        return

    # @log_decorator(logger)
    def permutated_array_of_length(self, array_size: int = 0)->List[int]:
        if array_size == 0:
            array_size = len(self)
        if array_size < len(self):
            raise ValueError('Permuted array size must be >= {}'.format(len(self)))
        arr = list(range(array_size + 1))
        for i, j in self.Swaps:
            arr = Permutation.apply_swap(arr, i, j)
        return arr

    @property
    def permutated_array(self) -> List[int]:
        return self.permutated_array_of_length(len(self))

    def __getitem__(self, index: int) -> int:
        try:
            return self.permutated_array[index]
        except IndexError:
            return index

    @property
    def cycles(self)->List[List[int]]:
        """
        A cycle is set the set of elements that have been rotated/Cycled
        their position.
        :return: list of all the elements cycles in the permutation
        """
        cycles_list = list()
        arr = self.permutated_array
        seen = list()
        for x in range(len(arr)):
            if x in seen or x == arr[x]:
                continue
            cycle = list()
            y = arr[x]
            while y not in cycle:
                seen.append(y)
                cycle.append(y)
                y = arr[y]

            if len(cycle) > 1:
                cycle.reverse()
                cycles_list.append(cycle)
        for c in cycles_list:
            while min(c) != c[0]:
                c = Permutation.rotate_array(c)
        cycles_list.sort(key=lambda e: e[0])
        return cycles_list

    @property
    def is_idel(self)->bool:
        """
        :return:
        """
        p = Permutation.of_the_array(self.permutated_array)  # Refresh
        self.Swaps = p.Swaps
        return len(self.Swaps) == 0

    @property
    def inv(self) -> Permutation_Type:
        """
        if x' is the permutation inverse of x then
        x * x' = () idel permutation
        :return: The permutation inverse
        """
        p = Permutation()
        arr = self.permutated_array
        for index1, element in enumerate(arr):
            if element == index1:
                continue
            index2 = [i for i, x in enumerate(arr) if x == index1][0]
            p.add_swap(index1, index2)
            arr = Permutation.apply_swap(arr, index1, index2)
        return p

    def is_derangement_of(self,
                          other: Permutation_Type,
                          n: int = 0) -> bool:
        """
        Compare this Permutation object with 'other' Permutation object
         and confirms if they a derangement of each other.
         IE there no element maps to its self.
        :param other: The other Permutation object
        :param n:
        :return:
        """
        n = max(len(self), len(other)) if n == 0 else max([len(self), len(other), n])
        arr1, arr2 = self.permutated_array_of_length(n), other.permutated_array_of_length(n)
        for x, y in zip(arr1, arr2):
            if x == y:
                return False
        else:
            return True

    def elements_in_place(self, other: Permutation_Type) -> List[int]:
        """
        Compare this Permutation object with 'other' Permutation object and returns the elements
         whom are the same on both permutation objects.
        :param other: the elements whom are the same on both permutation objects
        :return:
        """
        ans = []
        for x, y in zip(self.permutated_array, other.permutated_array):
            if x == y:
                ans.append(x)
        return ans

    @staticmethod
    def generator(n: int)->Permutation_Type:
        """
        Generate Permutation objects of an array size (n)
        :param n: array max size
        :return: Python Iterator of Permutation objects of an array size (n)
        """
        arr = list(range(n))
        for p in itertools.permutations(arr):
            yield Permutation.of_the_array(list(p))

    @staticmethod
    def apply_swap(arr: List[int], index1: int, index2: int) -> List[int]:
        """
        function returns the array 'arr' with the elements of
        indeices index1 & index2 swapped
        :rtype: list
        :return:
        :param arr: the array to be swapped.
        :param index1: first index of the swap element
        :param index2: second index of the swap element
        :return: a copy of input array with elements in index1 & index2 swapped
        """
        arr[index1], arr[index2] = arr[index2], arr[index1]
        return arr

    @staticmethod
    def random(size: int) -> Permutation_Type:
        n = randint(1, factorial(size))
        for ind, p in enumerate(Permutation.generator(size)):
            if ind == n:
                return p

    @staticmethod
    def latin_square_generator(size: int)->pd.DataFrame:
        def no_duplicates(grid: List[Permutation_Type])->bool:
            for row1, row2 in itertools.combinations(grid, r=2):
                if not row1.is_derangement_of(row2, n=size):
                    return False
            else:
                return True
        for row in range(1, size):
            square = [Permutation()]
            for permu in Permutation.generator(size+1):
                # print('Checking {}'.format(permu.permutated_array_of_length(size)))
                if no_duplicates(square + [permu]):
                    # print('Added {}'.format(permu.permutated_array_of_length(size)))
                    square.append(permu)
            yield pd.DataFrame([_.permutated_array_of_length(size) for _ in square])





