from Group import *

g = Group.from_definition(
    lambda x, y: x * y,
    [complex(1, 0), complex(-1, 0), complex(0, 1), complex(0, -1)],
    lambda x: '{}{}{}{}j'.format(
        '+' if x.real > 0 else '-',
        abs(x.real),
        '+' if x.imag > 0 else '-',
        abs(x.imag)
    ))

print(g)
g.export_to_html('Test2.html')
# print(g.Cayley.labels.to_frame().to_html())
