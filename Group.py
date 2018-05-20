from inspect import getsource
from typing import TypeVar

from Permutation import *
from PrimeLib import *
from log_configuration import logger, log_decorator

Group_object = TypeVar('Group_object', bound='Group')


# @log_decorator(logger)
class Group(object):
    def __init__(self):
        self.Order = 1
        self.Cayley = Table.from_data_frame(pd.DataFrame(['0']))
        self._element_labels = pd.Series(data=['e'])
        self.name = 'feature will available in a later version'
        self.reference = 'feature will available in a later version'
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

    @classmethod
    @log_decorator(logger)
    def from_file(cls, filename: str, delimiter: str = '\t'):
        """
        Reads Caylay table (Multiplication table) from a file.
        [!] The Elements are separated by the delimiter (default : Tab) and row every new line.
        [!] The table must obey the Group Axims. ie the table must be a latin grid.
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
    def from_table(cls, table: Table_object) -> Group_object:
        """
        Create a Group with input multiplication table
        :param table: The Caylay table (aka multiplication table)
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
         Of each will be registered as different element where they should be the same.
         suggested parse function is
         lambda x: '{:+}{:+}j'.format(0.0+x.real, 0.0+x.imag)
        :return:
        """
        assert (callable(operation) and callable(parse))
        logger.info(('Creating a group of below elements under the operation {.__name__}'
                     + ' with the following definition:-\n{}\nElements are {}')
                    .format(operation, getsource(operation), element_set))
        order = len(element_set)
        mul_table = pd.DataFrame(index=range(order), columns=range(order))
        logger.debug('Empty multiplication table is created:-\n{}'.format(mul_table))
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
                           phi: Permutation_object = Permutation()) -> Group_object:
        logger.debug('G Element set = {}'.format(list(group_g.element_set)))
        logger.debug('H Element set = {}'.format(list(group_h.element_set)))
        logger.debug('phi = {}'.format(phi))
        gl, hl = group_g.Order, group_h.Order
        logger.debug('|G x. H| = |G|x.|H| = {} x. {} = {}'.format(gl, hl, gl * hl))

        def semidirect_multiply(x: int, y: int) -> int:
            x1, x2 = x // gl, x % gl
            y1, y2 = y // hl, y % hl
            z1 = group_g.multiply(x2, y1)
            z2 = group_h.multiply(x1, y2)
            z3 = phi[z2]
            n = hl * z1 + z3
            logger.info('({}*{},{}*{}) = ({},{}) = ({},{}) = {}'
                        .format(x2, y1, x1, y2, z1, z2, z1, z3, n))
            return n

        element_labels = list()
        for g, h in itertools.product(group_g.element_set, group_h.element_set):
            element_labels.append('({},{})'.format(g, h))

        return Group.from_definition(semidirect_multiply, list(range(gl * hl)),
                                     lambda el: element_labels[el])

    @classmethod
    def direct_product(cls,
                       group_g: Group_object,
                       group_h: Group_object) -> Group_object:
        logger.debug('G Element set = {}'.format(list(group_g.element_set)))
        logger.debug('H Element set = {}'.format(list(group_h.element_set)))
        gl, hl = group_g.Order, group_h.Order
        logger.debug('|G x H| = |G|x|H| = {} x {} = {}'.format(gl, hl, gl * hl))

        def direct_multiply(x: int, y: int) -> int:
            x1, x2 = x // gl, x % gl
            y1, y2 = y // hl, y % hl
            z1 = group_g.multiply(x2, y1)
            z2 = group_h.multiply(x1, y2)
            n = hl * z1 + z2
            logger.info('({}*{},{}*{}) = ({},{}) = {}'.format(x2, y1, x1, y2, z1, z2, n))
            return n

        element_labels = list()
        for g, h in itertools.product(group_g.element_set, group_h.element_set):
            element_labels.append('({},{})'.format(g, h))
        return Group.from_definition(direct_multiply, list(range(gl * hl)),
                                     lambda el: element_labels[el])

    @classmethod
    def cyclic(cls, order: int):
        return Group.from_definition(lambda x, y: (x + y) % order, list(range(order)))

    def __mul__(self, other):
        return Group.direct_product(self, other)

    def __len__(self):
        return self.Order - 1

    @property
    def multiplication_table(self):
        self.Cayley.table = self.Cayley.table.astype(str)
        print(self.Cayley.table.columns.to_series().groupby(self.Cayley.table.dtypes).groups)
        print(type(self.Cayley.labels.values.tolist()))
        return self.Cayley.table.replace(
            value=self.Cayley.labels.values.tolist(),
            to_replace=list(self.Cayley.table.index))

    # Element Iterators
    @property
    def element_set(self):
        return range(len(self))

    # Group method
    # @log_decorator(logger)
    def multiply(self, x: int, y: int) -> int:
        return self.Cayley.get_item(x, y)

    @property
    def is_simple(self):
        return len(self.subgroups) < 2

    @property
    def is_solvable(self):
        if self._is_solvable is not None:
            return 'feature will available in a later version'
        return 'feature will available in a later version'

    @property
    # @log_decorator(logger)
    def is_abeailan(self):
        """
        A Group is abeailan if
        let x,y are Group elements of (G,*)
        then
        x * y = y * x must be True for all group elements
        """
        # to avoid double calculation
        if self._is_abeailan is not None:
            return self._is_abeailan

        self._is_abeailan = True
        for x, y in itertools.product(self.element_set, repeat=2):
            if x is y:
                continue
            if self.multiply(x, y) != self.multiply(y, x):
                logger.info('{} * {} != {} * {} \tThus the Group is non-abeailan'.format(x, y, y, x))
                self._is_abeailan = False
                break
        return self._is_abeailan

    # Element methods
    Identity = 0  # Standard for all Groups

    @staticmethod
    def is_identity(element):
        return element == Group.Identity

    def inverse(self, element):
        if element not in self.element_set:
            raise ValueError  # element Must be a Group element
        for x in self.element_set:
            if self.multiply(element, x) == Group.Identity:
                return x
        raise ValueError  # All elements must have inverse (Group Axiom)

    @property
    # @log_decorator(logger)
    def orbits(self):
        """
        Lists all Element orbits where The Element 'x' orbit is:-
        [x, x^2, x^3, ...,x^n]
        Where x^n=0 (Identity)
        """
        all_orbits = list()
        for element in self.element_set:
            orbit = self.orbit(element)
            all_orbits.append(orbit)
        return all_orbits

    def orbit(self, element):
        """
        The Element 'x' orbit is:-
        [x, x^2, x^3, ...,x^n]
        Where x^n=0 (Identity)
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
            raise ValueError(str(labels))
        if len(set(labels)) != len(labels):
            # Lables must be Unique
            raise ValueError('Lables must be Unique')
        self.Cayley.table.labels = labels
        self._element_labels = labels

    def label_of(self, element: int) -> str:
        try:
            return self.element_labels[element]
        except IndexError:
            return str(element)

    # Group to Group method
    # @log_decorator(logger)
    def is_homomorphic(self, group_h: Group_object) -> bool:  # to be implemented
        if max(len(self), len(group_h)) % min(len(self), len(group_h)) == 0:
            return False  # The bigger order of G & H must devides the other

    # @log_decorator(logger)
    def is_isomorphic(self, group_h: Group_object, phi: Permutation_object = Permutation()) -> bool:
        """

        :param group_h:
        :param phi
        :return:
        """
        if len(self) != len(group_h):
            return False
        for g, h in itertools.product(self.element_set, repeat=2):
            G, H = self.multiply(g, h), phi[group_h.multiply(g, h)]
            if G != H:
                logger.debug('G({}*{}={}) where H({}#{}={})'.format(g, h, G, g, h, H))
                return False
        return True

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
    def export_data_to_dict(self):
        group_data = dict()
        group_data['order'] = self.Order
        group_data['abeailan'] = self.is_abeailan
        group_data['simple'] = self.is_simple
        temp_subgroups = self.subgroups
        group_data['numberOfSubgroups'] = len(temp_subgroups)
        group_data['elementLabels'] = array1d_to_str([self.label_of(el) for el in self.element_set])
        group_data['caleyTable'] = self.Cayley.export_data_to_dict
        group_data['Orbits'] = [array1d_to_str(x) for x in self.orbits]
        group_data['generators'] = [array1d_to_str(x) for x in self.generators]
        group_data['subgroups'] = [{'subset': sg['subset'],
                                    'subgroup': sg['subgroup'].export_data_to_dict} for sg in temp_subgroups]
        return group_data

    def export_data_to_json_file(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.export_data_to_dict, file, indent=4)
        return

    @property
    def to_html(self) -> str:
        def tag(tag_name: str, inner_text: str, attributes: dict = None, close_tag=True):
            output = '<' + tag_name
            if attributes is not None:
                for att, value in attributes.items():
                    output += ' ' + att + '=' + '"' + value + '"'
            output += '>' + inner_text
            if close_tag:
                output += '</' + tag_name + '>'
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
        content += tag('h1', 'Multiplication table raw data:-')
        content += tag('p', self.Cayley.table.to_html())
        content += tag('h1', 'Element labels:')
        label_list = self.Cayley.labels.to_frame()
        label_list.columns = ['Label']
        content += tag('p', label_list.to_html())
        content += tag('h1', 'Group Name:')
        content += tag('p', self.Cayley.table.to_html())
        html_output = tag('Title', self.name)
        html_output += tag('link', '', attributes={'rel': 'stylesheet', 'href': 'stylesheet.css'}, close_tag=False)
        html_output = tag('head', html_output)
        html_output += tag('body', content)
        return '<!DOCTYPE html>' + tag('html', html_output)

    def export_to_html(self, filename):
        with open(filename, 'w') as html_file:
            html_file.write(self.to_html)
            print('HTML File have been written in ' + filename + '.\n')
