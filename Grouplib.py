import sqlite3

import seaborn as sns

from PrimeLib import *
from log_configuration import log_decorator
from mapping import *

Group_object = TypeVar('Group_object', bound='Group')
from contextlib import contextmanager


@contextmanager
@log_decorator(logger)
def GroupDB():
    path = 'GroupDB.db'
    with sqlite3.connect(path) as con:
        yield con


# @log_decorator(logger)
class Group(object):
    def __init__(self):
        self.Order = 1
        self.Cayley = Table.from_data_frame(pd.DataFrame(['0']))
        self._element_labels = pd.Series(data=['e'])
        self.name = 'UnnamedGroup'
        self._reference = None
        self._is_abeailan = None
        self._is_simple = None
        self._is_solvable = None
        self._subgroups = None

    def __str__(self):
        s = 'Group Name:\t{}\n'.format(self.name)
        s += 'Group Reference:\t{}\n'.format(self.reference)
        s += 'Group Order:\t{}\n'.format(self.Order)
        s += 'Abeailan group:\t{}\n'.format(self.is_abeailan)
        s += 'solvable : {}\n'.format(self.is_solvable)
        s += 'Multiplication table raw data:-\n' + str(self.Cayley) + '\n\n'
        s += 'Element Lables:-\n{}\n'.format(self.element_labels)
        s += 'Multiplication table :-\n{}\n'.format(self.multiplication_table)
        return s

    def __mul__(self, other):
        return Group.direct_product(self, other)

    def __len__(self):
        return self.Order - 1

    # class methods
    @classmethod
    @log_decorator(logger)
    def from_file(cls, filename: str, delimiter: str = '\t'):
        """
        Reads Caylay table (Multiplication table) from a file.
        [!] The Elements are separated by the delimiter (default : Tab) and row every new line.
        [!] The table must obey the Group Axiom. ie the table must be a latin grid.
        :param filename: The file path and name
        :param delimiter: string that separate the group elements in a row, default Tab.
        :return: Group Object
        """
        group = cls()
        with open(filename, 'r') as file:
            file.seek(0)
            labels = file.readline().split(delimiter)
            labels = list(map(str.strip, labels))  # Remove white space
            group.Order = len(labels) + 1  # python counts 0 to (n-1)
            group.element_labels = labels
        caley_table = Table.from_file(filename, delimiter=delimiter)
        group = Group.from_table(caley_table)
        return group

    @classmethod
    # @log_decorator(logger)
    def from_table(cls, table: Table_Type) -> Group_object:
        """
        Create a Group with input multiplication table
        :param table: The Cayley table (aka multiplication table)
        :return: Group Object
        """
        # if not table.is_caley_table:raise ValueError
        group = cls()
        group.Order = len(table)
        group.Cayley = table
        group.element_labels = table.labels
        logger.debug('Group Order = {}'.format(group.Order))
        logger.debug('table input:-{}'.format(group.Cayley))
        return group

    @classmethod
    def from_definition(cls, operation, element_set, parse=str):
        """
        Create a Group object from list of python objects and a binary operation.
        :param operation: The binary operation of the group.
        :param element_set: The group's elements list.
        :param parse: The elements of the group label will be registered by output of this function.
        By default it will the __str__() of the object. However, parsing is needed sometime
         such as the following example:-
         The result of __str__() of complex(0,1) can be '1j', '0+1j' or '-0+1j'
         Of each will be registered as find source different element where they should be the same.
         suggested parse function is
         lambda x: '{:+}{:+}j'.format(0.0+x.real, 0.0+x.imag)
        :return:
        """
        assert (callable(operation) and callable(parse))
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
        mul_table = Table.from_data_frame(mul_table)
        group = cls.from_table(mul_table)
        return group

    @classmethod
    def semidirect_product(cls,
                           group_g: Group_object,
                           group_h: Group_object,
                           phi: Mapping_Type) -> Group_object:
        gl, hl = group_g.Order, group_h.Order
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
                    x2, y1, x1, y2, z1, z2, z1, z3
                )
            )
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
                       group_g: Group_object,
                       group_h: Group_object) -> Group_object:
        gl, hl = group_g.Order, group_h.Order
        if gl < hl:
            return Group.direct_product(group_h, group_g)
        mapping = Mapping.by_function([i for i in range(gl)],
                                      lambda x: x % hl)
        return Group.semidirect_product(group_g, group_h, mapping)

    @classmethod
    def from_db(cls, group_ref: str) -> Group_object:
        mul_table = GroupDB.get_table(group_ref)
        mul_table = Table.from_data_frame(mul_table)
        return Group.from_table(mul_table)

    @classmethod
    def cyclic(cls, order: int) -> Group_object:
        """
        Creates a Cyclic Group of order n (G,+) aka Z mod n
        Modular addition of set of size n in mod n group
        :param order:
        :return: Cyclic Group
        """
        return Group.from_definition(lambda x, y: (x + y) % order, list(range(order)))

    # properties

    @property
    def multiplication_table(self) -> pd.DataFrame:
        """
        Outputs the multiplication table with the element tables
        :return: pandas.DataFrame: multiplication table
        """
        self.Cayley.table = self.Cayley.table.astype(str)
        return self.Cayley.table.replace(
            to_replace=self.Cayley.labels.values.tolist(),
            value=list(self.Cayley.table.index))

    # Element Iterators
    @property
    def element_set(self):
        return [i for i in range(self.Order)]

    @property
    def grid(self):
        return itertools.product(self.element_set, repeat=2)

    def grid_half(self):
        return itertools.product(self.element_set,
                                 self.element_set[0:self.Order // 2])

    # Group methods
    # @log_decorator(logger)
    def multiply(self, x: int, y: int) -> int:
        return self.Cayley.get_item(x, y)

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
        for x, y in self.grid_half():
            if x is y:
                continue
            if self.multiply(x, y) != self.multiply(y, x):
                logger.info('{} * {} != {} * {} \tThus the Group is non-abeailan'.format(x, y, y, x))
                self._is_abeailan = False
                break
        return self._is_abeailan

    # Element methods
    Identity = 0  # Standard for all Groups

    def inverse(self, element):
        logger.debug('Getting the inverse of the element {}'.format(element))
        if element not in self.element_set:
            raise ValueError('{} is not a Group element.'.format(element))
        for x in self.element_set:
            if self.multiply(element, x) == Group.Identity:
                return x
        raise ValueError('All elements must have inverse (Group Axiom)')

    @property
    # @log_decorator(logger)
    def orbits(self) -> list:
        """
        Lists all Element orbits where The Element 'x' orbit is:-
        [x, x^2, x^3, ...,x^n]
        Where x^n=0 (Identity)
        :return: list of all the element orbits
        """
        all_orbits = list()
        for element in self.element_set:
            orbit = self.orbit(element)
            all_orbits.append(orbit)
        return all_orbits

    def orbit(self, element: int) -> list:
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
        return orbit

    def element_order(self, element: int) -> int:
        """
        Element x have order n where x^n=e (group Identity)
        :param element: element in the Group
        :return: element order -> int
        """
        return len(self.orbit(element))

    # Element Labeling
    @property
    def element_labels(self):
        return self._element_labels

    @element_labels.setter
    def element_labels(self, labels: pd.Series):
        if len(self) > len(labels):
            # Lables does incloude all elements
            raise ValueError('Group elements are {} and given lables are {}'
                             .format(self.Order, len(labels) + 1))
        if len(set(labels)) != len(labels):
            # Lables must be Unique
            raise ValueError('Lables must be Unique')
        self._element_labels = labels

    def label_of(self, element: int) -> str:
        try:
            return self.element_labels[element]
        except IndexError:
            return str(element)

    # Group to Group method
    # @log_decorator(logger)
    def find_homomorphic_mapping(self, group_h: Group_object) -> (bool, Mapping_Type):  # to be implemented
        if self.Order < group_h.Order:
            return group_h.is_homomorphic(self)
        elif self.Order % group_h.Order:
            return False  # The bigger order of G & H must devides the other
        # check all possible mapping
        for mapping in Mapping.mapping_generator(self.Order, group_h.Order):
            logger.debug('Checking homomorphism condition with the following mapping:-\n{}'
                         .format(mapping))
            if self.is_homomorphic(group_h, mapping):
                return True, mapping
        else:
            return False, None

    def is_homomorphic(self, group_h: Group_object, mapping: Mapping_Type = None):
        if self.Order < group_h.Order:
            return group_h.is_homomorphic(self)
        elif self.Order % group_h.Order:
            return False  # The bigger order of G & H must devides the other

        for g1, g2 in self.grid:
            try:
                logger.debug('Checking the homomorphism condition on ({}, {})'
                             .format(g1, g2))
                lhs = mapping[self.multiply(g1, g2)]
                rhs = group_h.multiply(mapping[g1], mapping[g2])
                homomorphism_condition = lhs == rhs
                logger.debug('phi[ {} * {}] =? phi[{}] # phi[{}] for (G,*) , (H,#)'
                             .format(g1, g2, g1, g2))
                logger.debug('h[{}] =? {} # {}'
                             .format(self.multiply(g1, g2), mapping[g1], mapping[g2]))
                logger.debug('{} = {}\t[{}]'.format(lhs, rhs, homomorphism_condition))
                if not homomorphism_condition:
                    return False
            except IndexError:
                logger.debug('homomorphism condition failed due to mapping discrepancy')
                return False
        else:
            return True

    # @log_decorator(logger)
    def find_isomorphic_mapping(self, group_h: Group_object) -> (bool, Mapping_Type):
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

        logger.debug('Finding subgroups of group of order {}'.format(len(self)))
        if prime_test(self.Order):
            logger.debug('Since the Group order {} is prime there no subgroups.'.format(self.Order))
            return list()
        divisors = find_divisors(self.Order)
        subgroup_list = []
        for subgroup_size in divisors:
            if subgroup_size < 2 or subgroup_size == self.Order:
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
                    subgroup_list.append(
                        {'subgroup': sg, 'subset': subgroup_elements})
                    logger.debug(
                        '{} of size {} makes a VAILD subgroup'.format(subgroup_elements, len(subgroup_elements)))
                except Exception:
                    logger.debug('{} of size {} is not subgroup'.format(subgroup_elements, len(subgroup_elements)))
                    continue
        self._subgroups = subgroup_list
        return self._subgroups

    @property
    # @log_decorator(logger)
    def generators(self):
        def generated_elements(element_set1, element_set2):
            assert iter(element_set1) and iter(element_set2)
            element_generated = set()
            for x, y in itertools.product(element_set1, element_set2):
                element_generated.add(self.multiply(x, y))
            element_generated = list(element_generated)
            element_generated.sort()
            return element_generated

        def is_new_generator(possible_gen, existing_gen):
            for g in existing_gen:
                common = set(g).intersection(set(possible_gen))
                if list(common) in existing_gen:
                    return False
            else:
                logger.debug('{} is new possible generator'.format(possible_gen))
                return True

        generators_list = list()
        for orbits in all_subsets(self.orbits[1:]):
            gens = [orbit[0] for orbit in orbits]
            if not is_new_generator(gens, generators_list):
                logger.debug('{} already have generators'.format(gens))
                continue
            elements = functools.reduce(generated_elements, orbits, [0])
            logger.debug('The elements {} have generated {}'.format(gens, elements))
            logger.debug('{} == {}\t[{}]'.format(elements, self.element_set, elements == list(self.element_set)))
            if elements == list(self.element_set):
                logger.debug('{} Added'.format(gens))
                generators_list.append(gens)
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
            index=['Order', 'Abeailan', 'Simple', 'solvable'],
            data=[self.Order, self.is_abeailan, self.is_simple, self.is_solvable]
        )
        content = tag('h1', 'Group Name:')
        content += tag('p', '{}'.format(self.name))
        content += tag('h1', 'Group Reference:')
        content += tag('p', '{}'.format(self.reference))
        content += tag('h1', 'Group properties:')
        content += group_properties.to_html()
        content += tag('h1', 'Element labels:')
        label_list = self.Cayley.labels.to_frame()
        label_list.columns = ['Label']
        content += label_list.to_html()
        content += tag('h1', 'Multiplication table:-')
        content += tag('p', self.Cayley.table.to_html())

        html_output = tag('Title', self.name)
        html_output += tag(tag_name='links', attributes={'rel': 'stylesheet', 'href': 'stylesheet.css'})
        html_output = tag('head', html_output)
        html_output += tag('body', content)
        return '<!DOCTYPE html>' + tag('html', html_output)

    def export_to_html(self, filename):
        with open(filename, 'w') as html_file:
            html_file.write(self.to_html)
            print('HTML File have been written in ' + filename + '.\n')

    def plot(self):
        return sns.heatmap(self.Cayley.table,
                           annot=True,
                           fmt="g",
                           cmap='viridis',
                           cbar_kws={"ticks": self.element_set})

    # ## Database methods ## #
    @property
    def reference(self):
        if self._reference is None:
            with GroupDB() as db:
                counter = 0
                group_ref = 'G{}-{}'.format(self.Order, counter)
                refs = db.cursor().execute('SELECT GroupRef FROM Groups')
                for ref in refs:
                    if group_ref != ref:
                        self._reference = group_ref
                        break
                    else:
                        counter += 1
        return self._reference

    def export_to_db(self):
        with GroupDB() as db:
            add_mul_table = 'Create'
            self.Cayley.table.to_sql(add_mul_table, db.connection)
