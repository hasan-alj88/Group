import pandas as pd
import numpy as np
import itertools as itr
import functools
from Permutation import Permutation
from log_configuration import logger
import PrimeLib
from typing import TypeVar, List, Tuple
from Permutation import Permutation_Type

Group_Type = TypeVar('Group_Type', bound='Group')


class Group:
    """

    """

    def __init__(self, cayley_table: pd.DataFrame, name: str = 'UnnamedGroup'):

        self._order = None
        self._is_abelian = None
        self._is_simple = None
        self._is_solvable = None
        self._orbits = None
        self._center = None
        self._generators = None
        self._subgroups = None
        self._automorphism = None
        self._multiplication_table = None
        self.name = name
        # logger.debug('Creating Group with the following Table:\n{}'.format(cayley_table))
        self.cayley_table = cayley_table
        self.labels = pd.Series(self.cayley_table.iloc[0].values)
        # logger.debug('Group elements labels are :-\n{}'.format(self.labels))
        self.cayley_table = self.cayley_table.replace(to_replace=self.labels.values,
                                                      value=np.arange(len(self.labels)))
        self.cayley_table.astype(np.int)
        self.cayley_table.sort_values(0, inplace=True)
        # logger.debug('Using numeric elements only :-\n{}'.format(self.cayley_table))
        self.cayley_table.set_index(self.cayley_table.iloc[0].values, inplace=True)
        # Validate Table
        try:
            # Test for Closure
            q = self.cayley_table[self.cayley_table.applymap(lambda x: isinstance(x, str))]
            q = pd.Series(q.values.flatten()).dropna().tolist()
            if len(q) > 0:
                error_message = 'Closure Axiom Violation. The following element are not in the set {}'.format(q)
                raise GroupAxiomsViolationClosure(error_message)
            # Must be square
            table_shape = self.cayley_table.shape
            if len(table_shape) != 2 and table_shape[0] != table_shape[1]:
                error_message = 'Group Table must be a square where entered cayley_table deminsions is {}'.format(
                    table_shape)
                raise NotValidGroupTable(error_message)
            self._order = self.cayley_table.shape[0]

            # No duplicates
            for rowcol in self.element_set:
                if any(self.cayley_table.iloc[rowcol].duplicated()):
                    error_message = 'No duplicates allowed in Table. Duplicate detected in row #{}'.format(rowcol)
                    raise NotValidGroupTable(error_message)
                if any(self.cayley_table.iloc[:, rowcol].duplicated()):
                    error_message = 'No duplicates allowed in Table. Duplicate detected in column #{}'.format(rowcol)
                    raise NotValidGroupTable(error_message)
            for a, b, c in itr.combinations(self.element_set, 3):
                if self.multiply(a, self.multiply(b, c)) != self.multiply(self.multiply(a, b), c):
                    error_message = 'Associativity Axiom Violation.'
                    error_message += 'where it does not apply on the following elements {}'.format([a, b, c])
                    raise GroupAxiomsViolationAssociativity(error_message)
        except Exception as e:
            print(e)
            self.cayley_table = pd.DataFrame([0])

    def __mul__(self, other: Group_Type) -> Group_Type:
        """
        Direct product between this group and the 'other' group
        :param other: The other group
        :return Group_Type: Direct product between this group and the 'other' group
        """
        return Group.semi_direct_product(self, other)

    def __str__(self) -> str:
        out = '____________________'
        out += '\nGroup Name\t:{}\n'.format(self.name)
        out += 'Order\t:{}\n '.format(self.order)
        out += 'Abelian\t:[{}]\n'.format(self.is_abelian)
        out += 'Caley Table:-\n{}\n'.format(self.cayley_table)
        out += 'Elements :-\n{}\n'.format(self.labels)
        out += '____________________'
        return out

    # Class methods  (constructors)
    @classmethod
    def from_file(cls,
                  filename: str,
                  delimiter: str = '\t',
                  name: str = 'UnnamedGroup') -> Group_Type:
        """
        Reads cayley_table cayley_table (Multiplication cayley_table) from a file.
        [!] The Elements are separated by the delimiter (default : Tab) and row every new line.
        [!] The cayley_table must obey the Group Axiom. ie the cayley_table must be a latin grid.
        :param name :(optional) group name
        :param filename: The file path and name
        :param delimiter: string that separate the group elements in a row, default Tab.
        :return: Group Object
        """
        return cls(cayley_table=pd.read_csv(filename,
                                            sep=delimiter,
                                            header=None),
                   name=name)

    @classmethod
    def from_definition(cls, operation: callable,
                        element_set: iter,
                        parse: callable = str,
                        name: str = 'Unnamed group') -> Group_Type:
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
        order = len(element_set)
        # logger.debug('Creating Table of size {}'.format(order))
        mul_table = pd.DataFrame(index=range(order), columns=range(order))
        for (xi, x), (yi, y) in itr.product(enumerate(element_set), repeat=2):
            # logger.debug('({},{})={}*{}'.format(xi, yi, x, y))
            mul_table.iloc[xi, yi] = parse(operation(x, y))
            # logger.debug('\t={}'.format(parse(operation(x, y))))
        # logger.debug('Table created by defined function is:-\t{}'.format(mul_table))
        return cls(cayley_table=mul_table, name=name)

    @classmethod
    def cyclic(cls, order: int) -> Group_Type:
        """
        Creates a Cyclic Group of order n (G,+) aka Z mod n
        Modular addition of set of size n in mod n group
        :param order: Group order (element set size)
        :return: Cyclic Group
        """
        name = 'Z_{}'.format(order)
        return Group.from_definition(lambda x, y: (x + y) % order, list(range(order)), name=name)

    @classmethod
    def symmetric(cls, n: int) -> Group_Type:
        """
        Creates the symmetric group of the all permutations of n elements
        :param n:
        :return:
        """
        element_list = [p for p in Permutation.generator(n)]
        return Group.from_definition(operation=Permutation.__add__,
                                     element_set=element_list,
                                     name='S_{}'.format(n))

    @classmethod
    def dihedral(cls, n: int) -> Group_Type:
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

        element_labels = pd.Series(element_list)
        for ind, el in enumerate(element_list):
            r_count = el.count('r')
            f_count = el.count('f')
            if r_count > 1:
                element_labels[ind] = 'r^{}'.format(r_count) if f_count == 0 else 'fr^{}'.format(r_count)
        return Group.from_definition(
            lambda x, y: reduce_expression(x + y),
            element_set=element_list,
            parse=lambda x: element_labels[x], name='D_(2*{})'.format(n))

    @classmethod
    def semi_direct_product(cls,
                            group_g: Group_Type,
                            group_h: Group_Type,
                            phi: Permutation_Type = Permutation()) -> Group_Type:
        """

        :param group_g:
        :param group_h:
        :param phi:
        :return:
        """
        gl, hl = group_g.order, group_h.order
        if len(phi) > gl:
            raise ValueError('Invalid phi permutation on group G')
        logger.debug('G Element set = {}'.format(list(group_g.element_set)))
        logger.debug('H Element set = {}'.format(list(group_h.element_set)))
        logger.debug('phi = {}'.format(phi))
        logger.debug('|G x. H| = |G|x.|H| = {} x. {} = {}'.format(gl, hl, gl * hl))

        def product_element(gx: int, hx: int) -> int:
            return gx + hx * gl

        def product_element_pair(prod_element: int):
            return prod_element % gl, prod_element // gl

        def semi_direct_multiply(el_x: int, el_y: int) -> int:
            x1, x2 = product_element_pair(el_x)
            y1, y2 = product_element_pair(el_y)
            logger.debug('{}*{}=({},{})*({},{})'.format(el_x, el_y, x1, y1, x2, y2))
            logger.debug('\t=({}*phi[{}], {}*{})'.format(x1, y1, x2, y2))
            logger.debug('\t=({}*{}, {}*{})'.format(x1, phi[y1], x2, y2))
            z1 = group_g.multiply(x1, phi[y1])
            z2 = group_h.multiply(x2, y2)
            n = product_element(z1, z2)
            logger.debug('\t=({},{})={}'.format(z1, z2, n))
            return n

        prod_lables = []
        for prod_n in range(gl * hl):
            logger.debug('Getting label of {}'.format(prod_n))
            x, y = product_element_pair(prod_n)
            logger.debug('{}=>({},{})'.format(prod_n, x, y))
            x_label = group_g.labels[x]
            logger.debug('x = {} => {}'.format(x, x_label))
            y_label = group_h.labels[y]
            logger.debug('y = {} => {}'.format(y, y_label))
            logger.debug('Label of {} => ({},{})\n'.format(prod_n, x_label, y_label))
            prod_lables.append('({},{})'.format(x_label, y_label))

        def prod_element_label(n: int) -> str:
            return prod_lables[n]

        return Group.from_definition(semi_direct_multiply,
                                     list(range(gl * hl)),
                                     prod_element_label)

    def multiply(self, x: int, y: int) -> int:
        try:
            return self.cayley_table.iloc[x, y]
        except IndexError:
            error_message = 'single positional indexer is out-of-bounds.'
            error_message += '\nGroup order is {} where (x,y) = ({}, {})'.format(self.order, x, y)
            raise IndexError(error_message)

    def label_of(self, element: int) -> str:
        try:
            return self.labels.iloc[element]
        except IndexError:
            return '#INVALID!'

    @property
    def order(self) -> int:
        if self._order is None:
            self._order = self.cayley_table.shape[0]
        return self._order

    @property
    def element_set(self) -> List[int]:
        return [el for el in range(self.order)]

    @property
    def is_abelian(self) -> bool:
        """
        A Group is abeailan if following condition is met:-
        let x,y are Group elements of (G,*)
        then
        x * y = y * x must be True for all group elements
        :return: bool : is abeailan (True/False)
        """
        if self._is_abelian is None:
            for a, b in itr.combinations(self.element_set, 2):
                if self.multiply(a, b) != self.multiply(b, a):
                    self._is_abelian = False
                    break
            else:
                self._is_abelian = True
        return self._is_abelian

    @property
    def automorphism(self) -> Group_Type:
        """
        Group automorphism is the homomorphism of the Group to it self.
        As in all the rewiring of the group that keep the result multiplication table consistent.
        :return: automorphism Group
        """
        if self._automorphism is None:
            automorphism_elements = list()
            # logger.debug('Getting automorphism of group G')
            for phi in Permutation.generator(self.order):
                for a, b in itr.product(self.element_set, repeat=2):
                    if self.multiply(phi[a], phi[b]) != phi[self.multiply(a, b)]:
                        break
                else:
                    automorphism_elements.append(phi)
                    logger.debug('Added {}'.format(phi))
            logger.debug('AutomorphismGroup elements are \n{}'.format([str(i) for i in automorphism_elements]))
            return Group.from_definition(Permutation.__add__, automorphism_elements)
        else:
            return self._automorphism

    def subgroup_with_elements(self,
                               subgroup_elements: List[int],
                               name: str = None) -> Group_Type:
        """
        Creating a group from a group subset
        :param subgroup_elements: subset of group elements
        :param name: (optional) name of the subgroup
        :return: A subgroup if the elements forms a valid group otherwise trivial group
        """
        try:
            return Group.from_definition(self.multiply,
                                         subgroup_elements,
                                         name=name if name is not None else 'subgroup of {}'.format(self.name),
                                         parse=lambda x: self.labels[x])
        except ValueError:
            raise

    @property
    def subgroups(self) -> List[Group_Type]:
        """
        List the proper subgroups of the Group G
        :return: List the proper subgroups of the Group G
        """
        if self._subgroups is None:
            self._subgroups = list()
            for subgroup_order in PrimeLib.find_divisors(self.order):
                if subgroup_order in [1, self.order]:
                    # logger.debug('Skip Group of size {}'.format(subgroup_order))
                    continue
                for subgroup_elements in itr.combinations(self.element_set, r=subgroup_order):
                    try:
                        subgroup = self.subgroup_with_elements(subgroup_elements)
                        if subgroup.order == 1:
                            raise ValueError
                        # logger.debug('Added a subgroup of size {}'.format(subgroup_order))
                        self._subgroups.append(subgroup)
                        logger.debug(subgroup)
                        print(subgroup)
                    except ValueError:
                        # logger.debug('Skip')
                        continue

        return self._subgroups

    def is_homomorphic(self, group_h: Group_Type) -> Tuple[bool, str]:
        if group_h.order < self.order:
            return group_h.is_homomorphic(self)
        elif group_h.order % self.order == 0:
            # logger.debug('Determining if g and h is a homomorphism')
            for phi in Permutation.generator(group_h.order):
                print('phi = {}'.format(phi))
                for a, b in itr.product(self.element_set, repeat=2):
                    if phi[self.multiply(a, b)] != group_h.multiply(phi[a], phi[b]):
                        break
                else:
                    return True, str(phi)
            return False, ''
        else:
            return False, ''

    def is_isomorphic(self, group_h: Group_Type) -> bool:
        if self.order == group_h.order:
            return self.is_homomorphic(group_h)[0]
        else:
            return False

    def orbit(self, element: int) -> List[int]:
        """
        The Element 'x' orbit is:-
        [x, x^2, x^3, ...,x^n]
        Where x^n=0 (Identity)
        :param element: int: An element in the group
        :return: element orbit
        """
        orbit = [element]
        while 0 not in orbit:
            orbit.append(self.multiply(orbit[-1], element))
        return orbit

    @property
    def orbits(self) -> List[List[int]]:
        """
        Lists all Element orbits where The Element 'x' orbit is:-
        [x, x^2, x^3, ...,x^n]
        Where x^n=0 (Identity)
        :return: list of all the element orbits
        """
        if self._orbits is None:
            self._orbits = list()
            for el in self.element_set:
                self._orbits.append(self.orbit(el))
        return self._orbits

    def element_order(self, element) -> int:
        """
        Element x have order n where x^n=e (group Identity)
        :param element: element in the Group
        :return: element order -> int
        """
        return len(self.orbit(element))

    @property
    def generators(self) -> List[List[int]]:
        """
        the smallest subset of group elements that can generate the complete group element set.
        :return: List of group generators
        """
        if self._generators is None:
            def generated_elements(gens) -> List[int]:
                if len(gens) < 1:
                    return []
                elif len(gens) == 1:
                    return self.orbit(gens[0])
                elif len(gens) == 2:
                    element_generated = set()
                    for x, y in itr.product(self.orbits[gens[0]], self.orbits[gens[1]]):
                        element_generated.add(self.multiply(x, y))
                else:
                    element_generated = set()
                    for g1, g2 in itr.combinations(gens, r=2):
                        element_generated.union(set(generated_elements([g1, g2])))

                element_generated = list(element_generated)
                element_generated.sort()
                return element_generated

            generators_list = list()
            # logger.debug('{} divisors are {}'.format(self.order, [_ for _ in PrimeLib.find_divisors(self.order)]))
            for n in PrimeLib.find_divisors(self.order):
                # logger.debug('n = {}'.format(n))
                for g in itr.combinations(self.element_set, r=n):
                    # logger.debug('g = {}'.format(g))
                    # logger.debug('element orders = {}'.format([self.element_order(_) for _ in g]))
                    if functools.reduce(lambda x, y: x * y, [self.element_order(_) for _ in g], 1) < self.order:
                        # logger.debug('Skip')
                        continue
                    # logger.debug('generated elements are \n{}'.format(generated_elements(g)))
                    for g_existing in generators_list:
                        if set(g).issubset(set(g_existing)):
                            # logger.debug('generators already existing')
                            break
                    else:
                        if generated_elements(g) == self.element_set:
                            # logger.debug('Added')
                            generators_list.append(list(g))
                            continue
                        # logger.debug('discarded')
            self._generators = generators_list
        return self._generators

    def inverse(self, element: int) -> int:
        """
        The inverse of the element is when
        element * inverse = Identity (0)
        :param element: Group element
        :return:element inverse
        """
        if element not in self.element_set:
            raise ValueError('{} is not a Group element.'.format(element))
        for inv in self.element_set:
            if self.multiply(element, inv) == 0:
                return inv

    @property
    def center(self) -> List[int]:
        """
        All the subset of Group elements that  that commute with every element of G.
        :return: Group center elements
        """
        if self._center is None:
            center_list = set()
            for a in self.element_set:
                for b in self.element_set:
                    if self.multiply(a, b) != self.multiply(b, a):
                        break
                else:
                    center_list.add(a)
            self._center = [_ for _ in center_list]
        return self._center

    def conjugacy_class_of(self, a: int) -> List[int]:
        """
        A conjugacy class that the group element 'a' is in
        :param a: Group element 'a'
        :return: CL(a)
        """
        cl = {a}
        for b in self.element_set:
            for g in self.element_set:
                if self.multiply(a, g) == self.multiply(g, b) and a != b:
                    break
            else:
                cl.add(b)
        cl = cl - set(self.center)
        return list(cl)

    @property
    def conjugacy_classes(self) -> List[List[int]]:
        """
        A conjugacy class in a group can be defined in any of the following ways:
        It is an orbit of the group (as a set) under the action of the group on itself by conjugation
        (or as inner automorphisms) It is an equivalence class under the equivalence relation of being conjugate.
        :return: list of Group conjugacy classes
        """
        all_cl = list()
        all_cl.append(self.center)
        done = set(self.center)
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
            done |= set(cl)
            all_cl.append(cl)
        return all_cl

    # Representations
    @property
    def multiplication_table(self) -> pd.DataFrame:
        """
        Outputs the multiplication table with the element tables
        :return: pandas.DataFrame: multiplication table
        """
        if self._multiplication_table is None:
            self._multiplication_table = self.cayley_table.replace(to_replace=self.element_set,
                                                                   value=self.labels.values,
                                                                   inplace=False)
        return self._multiplication_table


class NotValidGroupTable(Exception):
    def __init__(self, message):
        super(NotValidGroupTable, self).__init__(message)


class GroupAxiomsViolationClosure(Exception):
    def __init__(self, message):
        super(GroupAxiomsViolationClosure, self).__init__(message)


class GroupAxiomsViolationAssociativity(Exception):
    def __init__(self, message):
        super(GroupAxiomsViolationAssociativity, self).__init__(message)
