from Permutation import *

Mapping_Type = TypeVar('Mapping_Type', bound='Mapping')


class Mapping(object):
    """
    This Ob
    """

    def __init__(self, phi: pd.DataFrame = pd.DataFrame(columns=['domain', 'co_domain']), strict_mapping: bool = True):
        self.links = phi

    def __str__(self):
        s = 'Mapping properties:-\n'
        s += 'size\t[{}:{}]\n'.format(self.size[0] + 1, self.size[1] + 1)
        s += 'Injective\t[{}]\n'.format(self.is_injective)
        s += 'Surjective\t[{}]\n'.format(self.is_surjective)
        s += 'Bijective\t[{}]\n'.format(self.is_bijective)
        s += '\tDomain\t\tCo-domain\n'
        for row in self.links.iterrows():
            s += '\t' + str(row[1]['domain']) + '\t->\t' + str(row[1]['co_domain']) + '\n'
        return s

    def __getitem__(self, item: int) -> int:
        try:
            return self.function(item)[0]
        except IndexError:
            raise IndexError('The element {} is not mapped.'.format(item))

    def __mul__(self, other: Mapping_Type) -> Mapping_Type:
        unmapped = set(self.co_domain).difference(set(other.domain))
        if len(unmapped) > 0:
            raise ValueError('The first co-domain elements {} are unmapped in second domain')
        total_mapping = Mapping()
        for x in self.domain:
            y = other[self[x]]
            total_mapping.add_link(x, y)
        return total_mapping

    def add_link(self, g: int, h: int):
        new_link = pd.DataFrame(data=[[g, h]], columns=['domain', 'co_domain'])
        self.links = self.links.append(new_link, ignore_index=True)
        self.links = self.links.drop_duplicates()

    @property
    def size(self):
        return len(set(self.domain)), len(set(self.co_domain))

    @property
    def domain(self):
        return list(self.links['domain'])

    @property
    def co_domain(self):
        return list(self.links['co_domain'])

    @property
    def is_injective(self) -> bool:
        """
        The function is injective (one-to-one)
        if each element of the co-domain is mapped to by at most one element of the domain.
        :return: mapping is injective (True/False)
        """
        return not (self.links.duplicated('co_domain').any() or
                    self.links.duplicated('domain').any())

    @property
    def is_surjective(self) -> bool:
        domain = set(self.links['domain'])
        co_domain = set(self.links['co_domain'])
        return len(domain.difference(co_domain)) == 0

    @property
    def is_bijective(self) -> bool:
        return self.is_injective and self.is_surjective

    @classmethod
    def create(cls, domain: list, co_domain: list):
        mapping_obj = Mapping()
        for g, h in zip(domain, co_domain):
            mapping_obj.add_link(g, h)
        return mapping_obj

    @classmethod
    def to_same(cls, set_len: int):
        map_obj = cls()
        for x in range(set_len):
            map_obj.add_link(x, x)
        return map_obj

    @classmethod
    def reversed(cls, set_len: int):
        map_obj = cls()
        for x, y in zip(range(set_len), reversed(range(set_len))):
            map_obj.add_link(x, y)
        return map_obj

    @classmethod
    def by_function(cls, domain_elements: iter, func: callable):
        map_obj = cls()
        for element in domain_elements:
            x = int(element)
            y = int(func(x))
            map_obj.add_link(x, y)
        return map_obj

    @classmethod
    def of_permutation(cls, permutation_obj: Permutation_Type):
        co_domain = permutation_obj.permutated_array
        domain = sorted(co_domain)
        return cls.create(domain, co_domain)

    @property
    def get_permutation(self):
        if not self.is_bijective:
            raise ValueError('Mapping must be bijective to be a permutation.')
        return Permutation.of_input_array(self.co_domain, self.co_domain)

    def function(self, element_from: int) -> list:
        return self.links.loc[self.links['domain'] == element_from]['co_domain'].values.tolist()

    @staticmethod
    def mapping_generator(domain_size: int, co_domain_size: int = None) -> Mapping_Type:
        if co_domain_size is not None:
            if domain_size % co_domain_size != 0 and domain_size > 0 and co_domain_size > 0:
                raise ValueError('domain size must divisible by co-domain size')
            segmentation_mapping = Mapping.by_function([i for i in range(domain_size)],
                                                       lambda x: x // co_domain_size)
        else:
            segmentation_mapping = Mapping.to_same(domain_size)
        for permutation in Permutation.generator(domain_size):
            permutation_mapping = Mapping.of_permutation(permutation)
            total_mapping = permutation_mapping * segmentation_mapping
            if total_mapping[0] == 0:
                yield total_mapping

    def kernel(self):
        return self.links[self.links['co_domain'] == 0]
