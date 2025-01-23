from collections import Counter

from rich import print

from FiniteGroup import FiniteGroup

group = FiniteGroup.alternating(6)
print(group)
print('---')
print(f'Order of group:   {group.order}')
print(f'Is group abelian? {group.is_abelian}')
print(f'Is group simple?  {group.is_simple}')
print(f'Is group solvable?{group.is_solvable}')
print('---')
print(f'Center of group: {group.center}')
print('---')
print('Group permutation elements:')
for i, element in enumerate(group.permutation_elements):
    print(f'Element {i}: {element}')
print('---')
print('Orbits of group:')
for i, orbit in group.orbits.items():
    print(f'Orbit {i}: {orbit} -> is subgroup? {group.is_subgroup_elements(orbit)}')
print('---')
print(f'orbit length counts: {Counter([len(orbit) for orbit in group.orbits.values()])}')
print('---')
print('proper normal subgroups:')
for i, subgroup in enumerate(group.proper_normal_subgroups):
    print(f'Subgroup {i}: {subgroup}')

# print('---')
# a3 = FiniteGroup.alternating(3)
# factor = group.factor_group(a3)
# print(f'G/A3: {factor}')

# print('---')
#
# print(f'Automorphism group:')
# for i, automorphism in enumerate(group.automorphism_group.permutation_elements):
#     print(f'Automorphism {i}: {automorphism}')