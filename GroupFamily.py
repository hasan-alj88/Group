from enum import IntEnum, auto


class GroupFamily(IntEnum):
    # Abelian groups
    Cyclic = auto()
    Abelian = auto()

    # Permutation groups
    Dihedral = auto()
    Symmetric = auto()
    Alternating = auto()
    Quaternion = auto()

    # Matrix groups
    GeneralLinear = auto()
    SpecialLinear = auto()
    Orthogonal = auto()
    Unitary = auto()
    Symplectic = auto()

    # Other groups
    Trivial = auto()
    PGroups = auto()
    Sporadic = auto()
    NilPotent = auto()
    Solvable = auto()
    Special = auto()
