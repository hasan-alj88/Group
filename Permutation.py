import itertools
import json
from collections import Counter
# from StringIO import StringIO # if sys.version_info[0] < 3
from io import StringIO
from typing import TypeVar

import pandas as pd

from log_configuration import logger

Table_object = TypeVar('Table_object', bound='table')
Permutation_object = TypeVar('Permutation_object', bound='Permutation')


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

    def __add__(self, other: Permutation_object) -> Permutation_object:
        combination = Permutation()
        combination.Swaps = self.Swaps + other.Swaps
        return Permutation.of_the_array(combination.permutated_array)

    def __sub__(self, other: Permutation_object) -> Permutation_object:
        return self + other.inv

    def __eq__(self, other: Permutation_object) -> Permutation_object:
        assert (isinstance(Permutation, other))
        return self.permutated_array == other.permutated_array

    @classmethod
    # @log_decorator(logger)
    def of_the_array(cls, arr):
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
    def of_input_array(cls, arr: list, sorted_arr: list):
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
    def of_input_series_object(cls, series: pd.Series) -> Permutation_object:
        series.replace(to_replace=series.index, value=list(series), inplace=True)
        return Permutation.of_input_array(list(series), series.index)

    @classmethod
    def of_cyclic_notation(cls, cyclic_input: str):
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

    def rotate(self, n=1):
        arr = []
        for i in range(n):
            arr = list(Permutation.rotate_array(self.permutated_array))
        return Permutation.of_the_array(arr)

    @staticmethod
    def rotate_array(arr):
        return [arr[-1]] + arr[:-1]

    def apply(self, array):
        for x, y in self.Swaps:
            array = Permutation.apply_swap(array, x, y)
        return array

    @staticmethod
    def array_validation(arr: list) -> bool:
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
    def permutated_array_of_length(self, array_size: int = 0):
        if array_size == 0:
            array_size = len(self)
        if array_size < len(self):
            raise ValueError('Permuted array size must be >= {}'.format(len(self)))
        arr = list(range(array_size + 1))
        for i, j in self.Swaps:
            arr = Permutation.apply_swap(arr, i, j)
        return arr

    @property
    def permutated_array(self) -> list:
        return self.permutated_array_of_length(len(self))

    def __getitem__(self, index: int) -> int:
        try:
            return self.permutated_array[index]
        except IndexError:
            return index

    @property
    def cycles(self):
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
    def is_idel(self):
        """

        :return:
        """
        p = Permutation.of_the_array(self.permutated_array)  # Refresh
        self.Swaps = p.Swaps
        return len(self.Swaps) == 0

    @property
    def inv(self) -> Permutation_object:
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

    def is_derangement_of(self, other: Permutation_object) -> bool:
        """
        Compare this Permutation object with 'other' Permutation object
         and confirms if they a derangement of each other.
         IE there no elements in the same palace in both of them.
        :param other: The other Permutation object
        :return:
        """
        assert (isinstance(other, Permutation))
        arr1, arr2 = self.permutated_array, other.permutated_array
        for x, y in zip(arr1, arr2):
            if x == y:
                return False
        else:
            return True

    def elements_in_place(self, other: Permutation_object) -> list:
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
    def generator(n: int):
        """
        Generate Permutation objects of an array size (n)
        :param n: array max size
        :return: Python Iterator of Permutation objects of an array size (n)
        """
        arr = list(range(n))
        for p in itertools.permutations(arr):
            yield Permutation.of_the_array(list(p))

    @staticmethod
    def apply_swap(arr: list, index1: int, index2: int) -> list:
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


class Table(object):

    def __init__(self):
        self.table = pd.DataFrame()
        self.labels = pd.Series()

    def __len__(self):
        return max(self.table.shape)

    @classmethod
    def from_file(cls, filename: str, delimiter: str = '\t'):
        table = cls()
        table.table = pd.read_csv(filename, sep=delimiter, header=None)
        table.normalize_table()
        table.labels = table.table.iloc[0, :]
        return table

    @classmethod
    def from_str(cls, input_str: str, delimiter: str = '\t'):
        table = cls()
        table.table = pd.read_csv(StringIO(input_str), sep=delimiter)
        table.normalize_table()
        return table

    @classmethod
    def from_data_frame(cls, table_input: pd.DataFrame) -> Table_object:
        table = cls()
        table.table = table_input
        table.normalize_table()
        return table

    def normalize_table(self):
        """
        The standard form is the first row and column are in sorted order
        This method set the table to standard form.
        """
        logger.debug('Table in non standard form:-\n{}'.format(self.table))
        p = Permutation.of_input_array(list(self.table.iloc[:, 0]),
                                       list(self.table.iloc[0, :]))
        self.table.index = p.permutated_array_of_length(len(self) - 1)
        self.table = self.table.reindex(index=sorted(self.table.index))
        logger.debug('Table after reindexing:-\n{}'.format(self.table))
        self.labels = self.row(0)
        logger.debug('Element labels:-\n{}'.format(self.labels))
        self.table = self.table.replace(value=list(self.labels.index), to_replace=list(self.labels))
        logger.debug('Table raw data:-\n{}'.format(self.table))
        return

    def __str__(self):
        return self.table.to_string(index=None, header=None)

    @property
    def to_html(self) -> str:
        self.normalize_table()
        return self.table.to_html

    def get_item(self, x: int, y: int):
        try:
            return self.table[x][y]
        except Exception as e:
            logger.error(e)
            logger.error('There is no Entry in table[{}][{}].'.format(x, y))
            raise e

    @property
    def is_latin_grid(self) -> bool:
        for row1, row2, col1, col2 in itertools.product(range(len(self)), repeat=4):
            if (row1 == row2) or (col1 == col2):
                continue
            p1 = Permutation.of_input_series_object(self.row(row1))
            p2 = Permutation.of_input_series_object(self.row(row2))
            p3 = Permutation.of_input_series_object(self.row(col1))
            p4 = Permutation.of_input_series_object(self.row(col2))
            logger.debug('row#{} Permutation: {}'.format(row1, p1))
            logger.debug('row#{} Permutation: {}'.format(row2, p2))
            logger.debug('row#{} Permutation: {}'.format(col1, p3))
            logger.debug('row#{} Permutation: {}'.format(col2, p4))
            if not p1.is_derangement_of(p2):
                logger.debug('row#{} and row#{} are not derangement of each other'.format(row1, row2))
                logger.debug('They have {} elements in common'.format(p1.elements_in_place(p2)))
                return False
            elif not p3.is_derangement_of(p4):
                logger.debug('col#{} and col#{} are not derangement of each other'.format(col1, col2))
                logger.debug('They have {} elements in common'.format(p2.elements_in_place(p3)))
                return False
        return True

    @property
    # @log_decorator(logger)
    def is_caley_table(self):
        # Must be a latin square
        condition1 = self.is_latin_grid
        # Must be a square table
        x, y = self.table.shape
        condition2 = x == y
        return condition1 and condition2

    def row(self, row: int) -> pd.Series:
        return self.table.ix[row, :]

    def column(self, col: int) -> pd.Series:
        return self.table.ix[:, col]

    def difference_map(self, other: Table_object) -> list:
        ret = list()
        for row_self, row_other in itertools.zip_longest(self.table, other.Table):
            row = list()
            for col_self, col_other in itertools.zip_longest(row_self.permutated_array,
                                                             row_other.permutated_array):
                if None in [col_self, col_other]:
                    row.append(1)
                else:
                    row.append(0 if col_self == col_other else 1)
            ret.append(row)
        return ret

    @property
    def export_data_to_dict(self):
        table_dict = dict()
        table_dict['Valid Caley table'] = self.is_caley_table
        permutated_array_data = [t.export_data_to_dict_of_length(len(self))['permutatedArray'] for t in self.table]
        cyclic_notation = [t.export_data_to_dict['cyclicNotation'] for t in self.table]
        table_rows, table_cyclic = dict(), dict()
        for row, x, y in zip(range(len(permutated_array_data)), permutated_array_data, cyclic_notation):
            table_rows['row#{}'.format(row)] = array1d_to_str(x)
            table_cyclic['row#{}'.format(row)] = y
        table_dict['table'] = table_rows
        table_dict['cyclicNotation'] = table_cyclic
        return table_dict

    def export_data_to_json_file(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.export_data_to_dict, file)
        return


class GroupAxiomsViolationClosure(Exception):
    def __init__(self, non_member_element, location):
        super().__init__('The element {} in {} is not a Group member.'.format(non_member_element, location))


class CaleyTableViolation(Exception):
    def __init__(self, non_unique_element, location):
        super().__init__(
            """The element {} in {} is not a unique member.
        As such the multiplication table is not a latin grid.
        """.format(non_unique_element, location))


def array1d_to_str(array1d):
    string = ''
    for x in array1d:
        string += str(x) + '    '
    return string[:-1]


def array2d_to_str(array2d):
    string = ''
    for row in array2d:
        string += array1d_to_str(row) + '\t'
        string = string[:-1]
        string += '\n'
    return string


def minmax(a, b):
    return min(a, b), max(a, b)
