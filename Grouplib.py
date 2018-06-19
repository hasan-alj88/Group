import sqlite3
from contextlib import contextmanager

import seaborn as sns

from PrimeLib import *
from log_configuration import log_decorator
from mapping import *

Group_Type = TypeVar('Group_Type', bound='Group')


@contextmanager
@log_decorator(logger)
def group_db():
    path = 'group_db.db'
    with sqlite3.connect(path) as con:
        yield con


def label_handling_decorator(number_of_element_input: int):
    def label_handling(func):
        def label_handling_wrapper(*args, **kwargs):
            """
            Group methods use integers to represent elements
            This decorator function is used to handel element labels as input
            :param args:
            :param kwargs:
            :return:
            """

            def element_input(element_in):
                if isinstance(element_in, str):
                    return args[0].label_of(element_in)
                else:
                    return element_in

            args = list(args)
            for el in range(1, number_of_element_input + 1):
                try:
                    args[el] = element_input(args[el])
                except IndexError:
                    raise ValueError('There is no {}th parameter'.format(el))
            args = tuple(args)

            ret = func(*args, **kwargs)

            if args[0].display_labels:
                if isinstance(ret, pd.DataFrame):
                    ret.replace(args[0].element_labels.to_dict(), inplace=True)
                    ret.fillna(value='', inplace=True)
                else:
                    ret = args[0].label_of(ret)
            return ret

        return label_handling_wrapper

    return label_handling


# @log_decorator(logger)
class Group(object):
    def __init__(self,
                 cayley_table: pd.DataFrame,
                 name: str = 'UnnamedGroup',
                 display_labels: bool = True):
        self._order = None
        self.cayley_table = cayley_table
        self._element_labels = None
        self.name = name
        self._reference = None
        self._is_abeailan = None
        self._is_simple = None
        self._is_solvable = None
        self._subgroups = None
        self._display_labels = display_labels
        self.normalize_table()

    def __str__(self):
        s = 'Group Name:\t{}\n'.format(self.name)
        s += 'Group order:\t{}\n'.format(self.order)
        s += 'Abeailan group:\t{}\n'.format(self.is_abeailan)
        s += 'solvable : {}\n'.format(self.is_solvable)
        s += 'Element Lables:-\n'
        for el, label in self.element_labels.to_dict().items():
            s += '{}\t->\t{}\n'.format(el, label)
        return s

    def __mul__(self, other):
        return Group.direct_product(self, other)

    def __len__(self):
        return self.order - 1

    # Validation methods
    def normalize_table(self):
        """
        The standard form is the first row and column are in sorted order
        This method set the table to standard form.
        """
        logger.debug('Group order is {}'.format(self.order))
        logger.debug('Table in non standard form:-\n{}'.format(self.cayley_table))
        p = Permutation.of_input_array(list(self.cayley_table.iloc[:, 0]),
                                       list(self.cayley_table.iloc[0, :]))
        self.cayley_table.index = p.permutated_array_of_length(len(self))
        self.cayley_table = self.cayley_table.reindex(index=sorted(self.cayley_table.index))
        logger.debug('Table after reindexing:-\n{}'.format(self.cayley_table))
        # Set Element lables
        self.element_labels = pd.Series(self.cayley_table.iloc[0, :])
        logger.debug('Element labels:-\n{}'.format(self.element_labels))
        self.cayley_table = self.cayley_table.replace(value=list(self.element_labels.index),
                                                      to_replace=list(self.element_labels))
        logger.debug('Table raw data:-\n{}'.format(self.cayley_table))
        self.validate_table()
        return

    def validate_table(self):

        # No Duplicate in all rows and columns
        for row_ind, row in self.cayley_table.iterrows():
            row_duplicates = row.duplicated()
            if row_duplicates.any():
                raise ValueError('The Group Multiplication table grid must be a latin Square.\n' +
                                 'The following elements are duplicated {}'.format(row[row_duplicates]))
        for col_ind, col in self.cayley_table.iteritems():
            col_duplicates = col.duplicated()
            if col_duplicates.any():
                raise ValueError('The Group Multiplication table grid must be a latin Square.\n' +
                                 'The following elements are duplicated {}'.format(col[col_duplicates]))

        # All the table entries must be in group element list (closure Group property)
        group_set = set(self.element_set)
        for row_ind, row in self.cayley_table.iteritems():
            elements_not_in_group = set(row.values).difference(group_set)
            if len(elements_not_in_group) > 0:
                raise ValueError(
                    'The element(s) found in {}th row {} are not in the group set'
                        .format(row_ind, elements_not_in_group))
        for col_ind, col in self.cayley_table.iteritems():
            elements_not_in_group = set(col.values).difference(group_set)
            if len(elements_not_in_group) > 0:
                raise ValueError(
                    'The element(s) found in {}th column {} are not in the group set'
                        .format(col_ind, elements_not_in_group))
        # A Group must be associative (a*b)*c == a*(b*c)
        if not self.is_associative:
            raise ValueError('A Group must be associative (a*b)*c == a*(b*c)')
        return

    # class methods
    @classmethod
    @log_decorator(logger)
    def from_file(cls, filename: str, delimiter: str = '\t', name: str = 'Unnamed group'):
        """
        Reads cayley_table table (Multiplication table) from a file.
        [!] The Elements are separated by the delimiter (default : Tab) and row every new line.
        [!] The table must obey the Group Axiom. ie the table must be a latin grid.
        :param name :(optional) group name
        :param filename: The file path and name
        :param delimiter: string that separate the group elements in a row, default Tab.
        :return: Group Object
        """
        return cls(cayley_table=pd.read_csv(filename, sep=delimiter, header=None),
                   name=name)

    @classmethod
    def from_definition(cls, operation: callable, element_set: iter, parse: callable = str,
                        name: str = 'Unnamed group'):
        """
        Create a Group object from list of python objects and a binary operation.
        :param operation: The binary operation of the group.
        :param element_set: The group's elements list.
        :param parse: The elements of the group label will be registered by output of this function.
        :param name: Group name
        By default it will the __str__() of the object. However, parsing is needed sometime
         such as the following example:-
         The result of __str__() of complex(0,1) can be '1j', '0+1j' or '-0+1j'
         Of each will be registered as find source different element where they should be the same.
         suggested parse function is
         lambda x: '{:+}{:+}j'.format(0.0+x.real, 0.0+x.imag)
        :return:
        """
        # logger.info(('Creating a group of below elements under the operation {.__name__}'
        #              + ' with the following definition:-\n{}\nElements are {}')
        #             .format(operation, getsource(operation), element_set))
        order = len(element_set)
        mul_table = pd.DataFrame(index=range(order), columns=range(order))
        logger.debug('Constructing the multiplication table:-')
        for (xi, x), (yi, y) in itertools.product(enumerate(element_set), repeat=2):
            z = parse(operation(x, y))
            # logger.debug('Multiplication table[{}][{}] = {}'.format(xi, yi, z))
            mul_table[xi][yi] = z
        logger.info('The result multiplication table:\n{}'.format(mul_table))
        return cls(cayley_table=mul_table, name=name)

    @classmethod
    def semidirect_product(cls,
                           group_g: Group_Type,
                           group_h: Group_Type,
                           phi: Mapping_Type) -> Group_Type:
        gl, hl = group_g.order, group_h.order
        if (gl, hl) != phi.size:
            raise ValueError(
                """Mapping mismatch.
                Mapping size is {}
                groups sizes are {}, {}""".format(phi.size, gl, hl))

        logger.debug('G Element set = {}'.format(list(group_g.element_set)))
        logger.debug('H Element set = {}'.format(list(group_h.element_set)))
        logger.debug('phi = {}'.format(phi))
        logger.debug('|G x. H| = |G|x.|H| = {} x. {} = {}'.format(gl, hl, gl * hl))

        def semidirect_multiply(x: int, y: int) -> int:
            x1, x2 = x // gl, x % gl
            y1, y2 = y // hl, y % hl
            z1 = group_g.multiply(x2, y1)
            z2 = group_h.multiply(x1, y2)
            z3 = phi[z2]
            n = hl * z1 + z3
            logger.info(
                '({}*{} , Phi[{}*{}]) = ({}, Phi[{}]) = ({}, {})'.format(
                    x2, y1, x1, y2, z1, z2, z1, z3))
            return n

        element_labels = list()
        for g, h in itertools.product(group_g.element_set, group_h.element_set):
            element_labels.append('({},{})'.format(g, h))

        def parse_label(el: int) -> str:
            try:
                return element_labels[el]
            except IndexError:
                raise IndexError('{} element label is not in {}'.format(el, element_labels))

        return Group.from_definition(semidirect_multiply, list(range(gl * hl)), parse_label)

    @classmethod
    def direct_product(cls,
                       group_g: Group_Type,
                       group_h: Group_Type) -> Group_Type:
        gl, hl = group_g.order, group_h.order
        if gl < hl:
            return Group.direct_product(group_h, group_g)
        mapping = Mapping.by_function([i for i in range(gl)],
                                      lambda x: x % hl)
        return Group.semidirect_product(group_g, group_h, mapping)

    @classmethod
    def from_db(cls, group_ref: str) -> Group_Type:
        # will be implemented
        pass

    @classmethod
    def cyclic(cls, order: int, name: str = 'Unnamed group') -> Group_Type:
        """
        Creates a Cyclic Group of order n (G,+) aka Z mod n
        Modular addition of set of size n in mod n group
        :param order: Group order (element set size)
        :param name: group name
        :return: Cyclic Group
        """
        return Group.from_definition(lambda x, y: (x + y) % order, list(range(order)), name=name)

    @classmethod
    def symmetric(cls, n: int, name: str = None) -> Group_Type:
        """
        Creates the symmetric group of the all permutations of n elements
        :param n:
        :param name: group name
        :return:
        """
        if name is None:
            name = 'S{}'.format(n)
        element_list = [p for p in Permutation.generator(n)]
        return Group.from_definition(operation=lambda x, y: x + y,
                                     element_set=element_list,
                                     name=name)

    @classmethod
    def dihedral(cls, n: int):
        """

        :param n:
        :return:
        """
        element_list = []
        for element in range(2 * n):
            if element // n == 0:
                element_list.append('e' if element % n == 0 else 'r' * (element % n))
            else:
                element_list.append('f' if element % n == 0 else 'f' + 'r' * (n - element % n))

        def reduce_expression(expression: str) -> str:
            while expression not in element_list:
                expression = expression.replace('ee', 'e')
                expression = expression.replace('ef', 'f')
                expression = expression.replace('er', 'r')
                expression = expression.replace('fe', 'f')
                expression = expression.replace('re', 'r')
                expression = expression.replace('ff', '')
                expression = expression.replace('rf', 'f' + ('r' * (n - 1)))
                expression = expression.replace('r' * n, '')
                expression = 'e' if expression == '' else expression
            return expression

        D = Group.from_definition(lambda x, y: reduce_expression(x + y),
                                  element_set=element_list,
                                  parse=lambda x: x)
        for ind, el in enumerate(element_list):
            r_count = el.count('r')
            f_count = el.count('f')
            if r_count > 1:
                element_list[ind] = 'r^{}'.format(r_count) if f_count == 0 else 'fr^{}'.format(r_count)
        D.element_labels = pd.Series(element_list)
        return D

    # properties
    @property
    def multiplication_table(self) -> pd.DataFrame:
        """
        Outputs the multiplication table with the element tables
        :return: pandas.DataFrame: multiplication table
        """
        if self.display_labels:
            return self.cayley_table.replace(self.element_labels.to_dict())
        else:
            return self.cayley_table

    # Element Iterators
    @property
    def element_set(self):
        return [i for i in range(self.order)]

    # @log_decorator(logger)
    @label_handling_decorator(2)
    def multiply(self, x: int, y: int) -> int:
        return self.cayley_table.iat[x, y]

    def multiply_left_to_right(self, elements: iter) -> int:
        return functools.reduce(function=self.multiply,
                                sequence=elements,
                                initial=0)

    # Group Properties
    @property
    def order(self):
        if self._order is None:
            # Table dimensions
            d1, d2 = self.cayley_table.shape
            if d1 != d2:
                raise ValueError('The Group Multiplication table grid must be a latin Square.\n' +
                                 'The Input table dimension was ({}, {})'.format(d1, d2))
            self._order = d1
        return self._order

    @property
    def is_simple(self) -> bool:
        return len(self.subgroups) < 2

    @property
    def is_solvable(self):
        if self._is_solvable is not None:
            return 'feature will available in a later version'
        return 'feature will available in a later version'

    @property
    # @log_decorator(logger)
    def is_abeailan(self) -> bool:
        """
        A Group is abeailan if following condition is met:-
        let x,y are Group elements of (G,*)
        then
        x * y = y * x must be True for all group elements
        :return: bool : is abeailan (True/False)
        """
        # to avoid double calculation
        if self._is_abeailan is not None:
            return self._is_abeailan

        self._is_abeailan = True
        for x, y in itertools.combinations(self.element_set, r=2):
            if x is y:
                continue
            if self.multiply(x, y) != self.multiply(y, x):
                logger.info('{} * {} != {} * {} \tThus the Group is non-abeailan'.format(x, y, y, x))
                self._is_abeailan = False
                break
        return self._is_abeailan

    @property
    def is_associative(self):
        for x, y, z in itertools.product(self.element_set, repeat=3):
            if self.multiply(self.multiply(x, y), z) != self.multiply(x, self.multiply(y, z)):
                logger.debug('For ({},{},{})'.format(x, y, z))
                logger.debug(
                    '{} != {}'.format(self.multiply(self.multiply(x, y), z), self.multiply(x, self.multiply(y, z))))
                return False
        return True

    # Element methods
    Identity = 0  # Standard for all Groups

    def inverse(self, element):
        # logger.debug('Getting the inverse of the element {}'.format(element))
        if element not in self.element_set:
            raise ValueError('{} is not a Group element.'.format(element))
        for x in self.element_set:
            if self.multiply(element, x) == Group.Identity:
                return x
        raise ValueError('All elements must have inverse (Group Axiom)')

    @label_handling_decorator(2)
    def conjugate(self, a: int, b: int) -> int:
        """

        :param a:
        :param b:
        :return: b * a * b^-1
        """
        return self.multiply_left_to_right([b, a, self.inverse(b)])

    @property
    @label_handling_decorator(0)
    def center(self) -> pd.DataFrame:
        center_list = set()
        for a in self.element_set:
            for b in self.element_set:
                if self.multiply(a, b) != self.multiply(b, a):
                    break
            else:
                center_list.add(a)
        center_list = pd.DataFrame(data=[_ for _ in center_list], columns=['Center'])
        return center_list

    @label_handling_decorator(1)
    def conjugacy_class_of(self, a: int) -> pd.DataFrame:
        cl = pd.DataFrame(data=[a])
        for b in self.element_set:
            for g in self.element_set:
                if self.multiply(a, g) == self.multiply(g, b) and a != b:
                    cl = cl.append(pd.DataFrame([b]), ignore_index=True)
                    break
        m = min(cl.values.flat)
        header = 'CL({})'.format(self.label_of(m) if self.display_labels else m)
        cl.columns = [header]
        cl.drop_duplicates(inplace=True, keep='first')
        cl.dropna(inplace=True)
        cl = cl.reindex(index=list(range(len(cl))))
        return cl

    @property
    def conjugacy_classes(self) -> pd.DataFrame:
        """

        :return:
        """
        all_cl = self.center
        all_cl.columns = ['Center']
        done = set(map(self.element_of, self.center.values.flat))
        for el in self.element_set:
            print('Doing {}'.format(el))
            print(all_cl)
            print(done)
            if el in done:
                print('{} is exiting'.format(el))
                continue
            print('Doing {}'.format(el))
            cl = self.conjugacy_class_of(el)
            print(cl)
            done |= set(map(self.element_of, cl.values.flat))
            all_cl = pd.concat([all_cl, cl], axis=1)
        return all_cl

    # @log_decorator(logger)
    @property
    def orbits(self) -> pd.DataFrame:
        """
        Lists all Element orbits where The Element 'x' orbit is:-
        [x, x^2, x^3, ...,x^n]
        Where x^n=0 (Identity)
        :return: list of all the element orbits
        """

        all_orbits = pd.DataFrame()
        for element in self.element_set:
            all_orbits = pd.concat([all_orbits, self.orbit(element)], axis=1)
        all_orbits = pd.DataFrame(all_orbits)

        if self.display_labels:
            all_orbits.replace(self.element_labels.to_dict(), inplace=True)
            all_orbits.fillna(value='', inplace=True)
        return all_orbits

    def orbit(self, element: int) -> pd.Series:
        """
        The Element 'x' orbit is:-
        [x, x^2, x^3, ...,x^n]
        Where x^n=0 (Identity)
        :param element: int: An element in the group
        :return: element orbit
        """
        ele = element
        orbit = [ele]
        while ele != Group.Identity:
            ele = self.multiply(ele, element)
            orbit.append(ele)
        header = '<{}>'.format(self.label_of(element)
                               if self.display_labels
                               else '<{}>'.format(element))
        orbit = pd.Series(orbit).rename(header)
        if self.display_labels:
            orbit.replace(self.element_labels.to_dict(), inplace=True)
        return orbit

    def element_order(self, element: int) -> int:
        """
        Element x have order n where x^n=e (group Identity)
        :param element: element in the Group
        :return: element order -> int
        """
        return len(self.orbit(element).values)

    @property
    def elements_order(self) -> pd.DataFrame:
        order_list = dict()
        for el in self.element_set:
            order = self.element_order(el)
            if order not in order_list.keys():
                order_list[order] = [el]
            else:
                order_list[order].append(el)
        order_list = pd.DataFrame(order_list)
        if self.display_labels:
            order_list.replace(self.element_labels.to_dict(), inplace=True)
        return order_list

    # Element Labeling
    @property
    def display_labels(self):
        return self._display_labels

    @display_labels.setter
    def display_labels(self, value: bool):
        assert value in [True, False]
        self._display_labels = value
        return

    @label_handling_decorator(2)
    def test(self, a: int, b: int) -> pd.DataFrame:
        print('({}, {})'.format(a, b))
        return pd.DataFrame({'a': [a], 'b': [b], 'a*b': [self.multiply(a, b)]})

    def test2(self, a: int, b: int) -> pd.DataFrame:
        print('({}, {})'.format(a, b))
        return pd.DataFrame({'a': [a], 'b': [b], 'a*b': [self.multiply(a, b)]})

    @property
    def element_labels(self):
        return self._element_labels

    @element_labels.setter
    def element_labels(self, labels: pd.Series):
        if len(self) > len(labels):
            # Lables does incloude all elements
            raise ValueError('Group elements are {} and given lables are {}'
                             .format(self.order, len(labels) + 1))
        if len(set(labels)) != len(labels):
            # Lables must be Unique
            raise ValueError('Lables must be Unique')
        self._element_labels = labels

    def label_of(self, element):
        if isinstance(element, str):
            try:
                return self.element_labels.index[self.element_labels == element][0]
            except IndexError:
                raise IndexError('There is no element with label "{}'.format(element))
        else:
            try:
                return self.element_labels[element]
            except IndexError:
                raise IndexError('The {}th Element is not in the element set.'.format(element))

    def element_of(self, element):
        if isinstance(element, str):
            return self.label_of(element)
        else:
            if element in self.element_set:
                return element
            else:
                raise ValueError('element {} is not in group element set'.format(element))

    # Group to Group method
    @log_decorator(logger)
    def find_homomorphic_mapping(self, group_h: Group_Type) -> list:  # to be implemented
        if self.order < group_h.order:
            return group_h.is_homomorphic(self)
        elif self.order % group_h.order != 0:
            return []  # The bigger order of G & H must devides the other
        # check all possible mapping
        mapping_list = list()
        for gens in self.generators:
            for mapping in Mapping.mapping_generator(len(gens), group_h.Order):
                logger.debug('mapping generators to element in H_group.')
                logger.debug(['\nphi[{}] = {}'.format(_, mapping[_]) for _ in gens])
                for gen in gens:  # iterate though all generator list
                    for g in gen:  # iterate though the generators
                        h = mapping[g]
                        g_orbit = self.orbit(g)
                        h_orbit = group_h.orbit(h)
                        gl, hl = len(g_orbit), len(h_orbit)
                        if max(gl, hl) % min(gl, hl) != 0:
                            # orbit lengths of g in Group G and h in group H
                            # are not divisible to each other
                            # find other mapping
                            break
                        # repeat the smaller orbit to match the bigger obit length
                        if gl > hl:
                            h_orbit = [_ for _ in itertools.repeat(h_orbit, gl // hl)]
                            h_orbit = [_ for _ in itertools.chain(*h_orbit)]
                        else:
                            g_orbit = [_ for _ in itertools.repeat(g_orbit, hl // gl)]
                            g_orbit = [_ for _ in itertools.chain(*g_orbit)]
                        # add mapping links
                        for g1, h1 in zip(g_orbit, h_orbit):
                            mapping.add_link(g1, h1)
                    else:
                        # No break hence add the mapping to te list
                        logger.debug('Checking homomorphism condition with the following mapping:-\n{}'
                                     .format(mapping))
                        if self.is_homomorphic(group_h, mapping):
                            mapping_list.append(mapping)
                        mapping_list.append(mapping)
                        continue
                    # break has occurred hence skip this mapping scheme
                    continue
            return mapping_list

    @log_decorator(logger)
    def is_homomorphic(self, group_h: Group_Type, mapping: Mapping_Type = None):
        if self.order < group_h.order:
            return group_h.is_homomorphic(self)
        elif self.order % group_h.order:
            return False  # The bigger order of G & H must devides the other

        for g1, g2 in itertools.product(self.element_set, repeat=2):
            try:
                logger.debug('Checking the homomorphism condition on ({}, {})'
                             .format(g1, g2))
                lhs = mapping[self.multiply(g1, g2)]
                rhs = group_h.multiply(mapping[g1], mapping[g2])
                homomorphism_condition = lhs == rhs
                if not homomorphism_condition:
                    logger.debug('phi[ {} * {}] =? phi[{}] # phi[{}] for (G,*) , (H,#)'
                                 .format(g1, g2, g1, g2))
                    logger.debug('h[{}] =? {} # {}'
                                 .format(self.multiply(g1, g2), mapping[g1], mapping[g2]))
                    logger.debug('{} = {}\t[{}]'.format(lhs, rhs, homomorphism_condition))
                    return False
            except IndexError:
                logger.debug('homomorphism condition failed due to mapping discrepancy')
                return False
        else:
            return True

    # @log_decorator(logger)
    def find_isomorphic_mapping(self, group_h: Group_Type) -> (bool, Mapping_Type):
        """

        :param group_h:
        :return:
        """
        if len(self) != len(group_h):
            return False, None
        else:
            return self.find_homomorphic_mapping(group_h)

    @property
    @log_decorator(logger)
    def subgroups(self):
        """List the proper subgroups of the Group G"""
        # to avoid double calculation
        if self._subgroups is not None:
            return self._subgroups

        logger.debug('Finding subgroups of group of order {}'.format(self.order))
        if prime_test(self.order):
            logger.debug('Since the Group order {} is prime there no subgroups.'.format(self.order))
            return list()
        divisors = find_divisors(self.order)
        subgroup_list = []
        for subgroup_size in divisors:
            if subgroup_size < 2 or subgroup_size == self.order:
                logger.debug('Not considering subgroup of size {} since only interested on proper subgroups.'
                             .format(subgroup_size))
                continue  # Only proper subgroups
            logger.debug('Finding Subgroups of size {}'.format(subgroup_size))
            for subgroup_elements in itertools.combinations(self.element_set, r=subgroup_size):
                logger.debug('Checking if elements {} form a subgroup.'.format(subgroup_elements))
                if Group.Identity not in subgroup_elements:
                    logger.debug('interested on subgroups only excluding cosets.')
                    continue
                try:
                    sg = Group.from_definition(
                        lambda x, y: str(self.multiply(x, y)),
                        subgroup_elements)
                    subgroup_list.append(sg)
                    logger.debug(
                        '{} of size {} makes a VAILD subgroup'.format(subgroup_elements, len(subgroup_elements)))
                except Exception as excep:
                    logger.debug('{} of size {} is not subgroup'.format(subgroup_elements, len(subgroup_elements)))
                    logger.error(excep)
                    continue
        self._subgroups = subgroup_list
        return self._subgroups

    @property
    # @log_decorator(logger)
    @label_handling_decorator(0)
    def generators(self) -> list:
        print('Start')
        display_label_state = self.display_labels
        self.display_labels = False

        def generated_elements(gens: list):
            if len(gens) < 1:
                return []
            elif len(gens) == 1:
                return self.orbit(gens[0]).drop_duplicates().values.tolist()
            elif len(gens) == 2:
                [x, y] = gens
                generated_ele = set()
                for xi, yi in itertools.product(self.orbit(x).values.tolist(),
                                                self.orbit(y).values.tolist()):
                    generated_ele.add(self.multiply(xi, yi))
                generated_ele = list(generated_ele)
                generated_ele.sort()
                return generated_ele
            else:
                return functools.reduce(lambda a, b: set(generated_elements([a, b])), gens, {0})

        generators_list = []
        print('{} divisors are {}'.format(self.order, [_ for _ in find_divisors(self.order)]))
        for n in find_divisors(self.order):
            print('n = {}'.format(n))
            for g in itertools.combinations(self.element_set, r=n):
                print('g = {}'.format(g))
                print('generated elements are \n{}'.format(generated_elements(g)))
                for g_existing in generators_list:
                    if set(g).issubset(set(g_existing)):
                        print('generators already existing')
                        break
                else:
                    if generated_elements(g) == self.element_set:
                        print('Added')
                        generators_list.append(g)
                        continue
                    print('discarded')
        self.display_labels = display_label_state
        return generators_list

    @property
    def to_html(self) -> str:
        def tag(tag_name: str, inner_text: str = None, attributes: dict = None):
            output = '<' + tag_name
            if attributes is not None:
                for att, value in attributes.items():
                    output += ' ' + att + '=' + '"' + value + '"'
            output += '>'
            if inner_text is not None:
                output += inner_text + '</' + tag_name + '>'
            return output

        group_properties = pd.DataFrame(
            columns=['property'],
            index=['order', 'Abeailan', 'Simple', 'solvable'],
            data=[self.order, self.is_abeailan, self.is_simple, self.is_solvable]
        )
        content = tag('h1', 'Group Name:')
        content += tag('p', '{}'.format(self.name))
        content += tag('h1', 'Group Reference:')
        content += tag('p', '{}'.format(self.reference))
        content += tag('h1', 'Group properties:')
        content += group_properties.to_html()
        content += tag('h1', 'Element labels:')
        label_list = self.cayley_table.labels.to_frame()
        label_list.columns = ['Label']
        content += label_list.to_html()
        content += tag('h1', 'Multiplication table:-')
        content += tag('p', self.cayley_table.to_html())

        html_output = tag('Title', self.name)
        html_output += tag(tag_name='links', attributes={'rel': 'stylesheet', 'href': 'stylesheet.css'})
        html_output = tag('head', html_output)
        html_output += tag('body', content)
        return '<!DOCTYPE html>' + tag('html', html_output)

    def export_to_html(self, filename):
        with open(filename, 'w') as html_file:
            html_file.write(self.to_html)
            print('HTML File have been written in ' + filename + '.\n')

    def plot(self, annot=True):
        return sns.heatmap(self.cayley_table,
                           annot=annot,
                           fmt="g",
                           cmap='viridis',
                           cbar_kws={"ticks": self.element_set})

    # ## Database methods ## #
    @property
    def reference(self):
        if self._reference is None:
            with group_db() as db:
                counter = 0
                group_ref = 'G{}-{}'.format(self.order, counter)
                refs = db.cursor().execute('SELECT GroupRef FROM Groups')
                for ref in refs:
                    if group_ref != ref or ref is None:
                        self._reference = group_ref
                        break
                    counter += 1
        return self._reference

    def export_to_db(self):
        with group_db() as db:
            add_mul_table = 'Create'
            self.cayley_table.to_sql(add_mul_table, db.connection)
