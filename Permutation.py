import itertools
import json
from collections import Counter
from log_configuration import logger, log_decorator
from typing import Type, TypeVar

Table_object = TypeVar('Table_object', bound='Table')
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

    def __add__(self, other: Permutation_object)-> Permutation_object:
        combination = Permutation()
        combination.Swaps = self.Swaps + other.Swaps
        return Permutation.of_the_array(combination.permutated_array)

    def __sub__(self, other: Permutation_object)-> Permutation_object:
        return self + other.inv

    def __eq__(self, other: Permutation_object)-> Permutation_object:
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
            arr = p.apply_swap(arr, index1, index2)
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

    @staticmethod
    def apply_swap(arr: list, index1: int, index2: int)-> list:
        """
        apply_swap function returns the array 'arr' with the elements of
        indeices index1 & index2 swapped
        :param arr: the array to be swapped.
        :param index1: first index of the swap element
        :param index2: second index of the swap element
        :return: a copy of input array with elements in index1 & index2 swapped
        """
        arr[index1], arr[index2] = arr[index2], arr[index1]
        return arr

    @staticmethod
    def array_validation(arr: list)-> bool:
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
    def permutated_array_of_length(self, array_size: int =0):
        if array_size == 0:
            array_size = len(self)
        if array_size < len(self):
            raise ValueError('Permuted array size must be >= {}'.format(len(self)))
        arr = list(range(array_size + 1))
        for i, j in self.Swaps:
            arr = Permutation.apply_swap(arr, i, j)
        return arr

    @property
    def permutated_array(self)-> list:
        return self.permutated_array_of_length(len(self))

    def __getitem__(self, index: int)-> int:
        try:
            return self.permutated_array[index]
        except IndexError:
            return index

    @property
    def cycles(self):
        """

        :return:
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
    def inv(self)-> Permutation_object:
        """

        :return:
        """
        p = Permutation()
        arr = self.permutated_array
        for index1, element in enumerate(arr):
            if element == index1:
                continue
            index2 = [i for i, x in enumerate(arr) if x == index1][0]
            p.add_swap(index1, index2)
            arr = p.apply_swap(arr, index1, index2)
        return p

    def is_derangement_of(self, other: Permutation_object)-> bool:
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

    def elements_in_place(self, other: Permutation_object)->list:
        """
        Compare this Permutation object with 'other' Permutation object and returns the elements whom are the same on both permutation objects.
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

    @property
    def export_data_to_dict(self):
        """
        Export the permutation object data
        :return: Python dictionary object with 'permutatedArray' & 'cyclicNotation' keys.
        """
        permutation_dict = dict()
        permutation_dict['permutatedArray'] = self.permutated_array
        permutation_dict['cyclicNotation'] = str(self)
        return permutation_dict

    def export_data_to_dict_of_length(self, n:int):
        """
        Export the permutation object data
        :return: Python dictionary object with 'permutatedArray' & 'cyclicNotation' keys.
        """
        permutation_dict = dict()
        permutation_dict['permutatedArray'] = self.permutated_array_of_length(n)
        permutation_dict['cyclicNotation'] = str(self)
        return permutation_dict


class Table(object):

    def __init__(self):
        self.Table = list()

    def __len__(self):
        return max(map(len, self.Table)) if len(self.Table) else 0

    @classmethod
    def from_file(cls, filename: str, delimiter: str ='\t'):
        table = cls()
        with open(filename, 'r') as file:
            lables = file.readline().split(delimiter)
            lables = list(map(str.strip, lables))  # Remove white space
            file.seek(0)
            for row in file:
                line = row.split(delimiter)
                line = list(map(str.strip, line))  # Remove white space
                table.Table.append(Permutation.of_input_array(line, lables))
            table.Table.sort(key=lambda x: x.permutated_array[0])
        return table

    @classmethod
    def from_str(cls, input_str: str, delimiter: str ='\t'):
        table = cls()
        labels = None
        input_str = input_str.split('\n')
        for index, row in enumerate(input_str):
            line = row.split(delimiter)
            line = list(map(str.strip, line))  # Remove white space
            if index == 0:
                labels = line
            table.Table.append(Permutation.of_input_array(line, labels))
        table.Table.sort(key=lambda x: x.permutated_array()[0])
        return table

    @classmethod
    def from_2d_list(cls, table_input):
        table = cls()
        for row in table_input:
            table.Table.append(Permutation.of_input_array(row, table_input[0]))
        table.Table.sort(key=lambda x: x.permutated_array[0])
        return table

    def __str__(self):
        output = 'Table:-\n'
        for row in self.Table:
            row_list = row.permutated_array_of_length(len(self))
            for item in row_list:
                output += str(item) + '\t'
            output = output[:-1]
            output += '\n'
        return output

    def get_item(self, x: int, y: int)->int:
        try:
            return self.Table[x][y]
        except Exception as e:
            logger.error(e)
            logger.error('Function Input are [{}][{}] and Table length is {}'.format(x, y, len(self)))
            raise e

    @property
    def is_latin_grid(self):
        ret = True
        for row1, row2 in itertools.product(self.Table, repeat=2):
            if row1 is row2:
                continue
            if not row1.is_derangement_of(row2):
                ret = False
                logger.debug('The following two rows are not derangement of each other\n{}\n{}\n'.format(row1, row2))
        return ret

    @property
    # @log_decorator(logger)
    def is_caley_table(self):
        # Must be a latin square
        condition1 = self.is_latin_grid
        # Must be a square table
        condition2 = len(self) == len(self.Table)
        return condition1 and condition2

    def row(self, row: int)->list:
        return self.Table[row].permutated_array(len(self))

    def column(self, col: int)->list:
        return [self.Table[col][x] for x in range(len(self) + 1)]

    def difference_map(self, other: Table_object)->list:
        ret = list()
        for row_self, row_other in itertools.zip_longest(self.Table, other.Table):
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
        permutated_array_data = [t.export_data_to_dict_of_length(len(self))['permutatedArray'] for t in self.Table]
        cyclic_notation = [t.export_data_to_dict['cyclicNotation'] for t in self.Table]
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


def array1d_to_str(array1d):
    string = ''
    for x in array1d:
        string += str(x) + '    '
    return string[:-1]


def array2d_to_str(array2d):
    string = ''
    for row in array2d:
        string += array1d_to_str(row)+'\t'
        string = string[:-1]
        string += '\n'
    return string


def minmax(a, b):
    return min(a, b), max(a, b)
