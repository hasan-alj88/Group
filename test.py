from Grouplib import *

# g = Group.from_definition(
#     lambda x, y: x * y,
#     [complex(1, 0), complex(-1, 0), complex(0, 1), complex(0, -1)],
#     lambda x: '{}{}{}{}j'.format(
#         '+' if x.real > 0 else '-',
#         abs(x.real),
#         '+' if x.imag > 0 else '-',
#         abs(x.imag)
#      ))

c2 = Group.cyclic(2)
m = Mapping.reversed(2)
g = Group.semidirect_product(c2, c2, m)
print(g)

# with GroupDB() as db:
#     db.cursor.execute('''CREATE TABLE IF NOT EXISTS SubGroups(
#     SubGroupEntry TEXT PRIMARY KEY NOT NULL,
#     GroupRef TEXT NOT NULL,
#     SubGroupRef TEXT NOT NULL
#     );''')
#     db.connection.commit()
#
#
