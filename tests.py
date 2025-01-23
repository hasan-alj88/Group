import unittest
from collections import defaultdict

import numpy as np

from FiniteGroup import FiniteGroup
from Permutation import Permutation


class PermutationTest(unittest.TestCase):

    def test_of_cyclic_notation(self):
        cyclic_notations = [
            '(0,2,3)',
            '(1,2)',
            '(1,2)(3,4)',
            '(0,5)(1,2,3)',
        ]

        for cyclic_notation in cyclic_notations:
            p = Permutation.of_cyclic_notation(cyclic_notation)
            self.assertEqual(cyclic_notation, str(p))

    def test_of_cyclic_notation_invalid(self):
        cyclic_notations = [
            '(0,2,3,)',
            '(0,5,,(1,2,3',
        ]

        for cyclic_notation in cyclic_notations:
            print(cyclic_notation)
            with self.assertRaises(ValueError):
                Permutation.of_cyclic_notation(cyclic_notation)


class TestFiniteGroup(unittest.TestCase):
    def setUp(self):
        # Common test groups
        self.trivial_group = FiniteGroup.trivial()
        self.cyclic2 = FiniteGroup.cyclic(2)
        self.cyclic3 = FiniteGroup.cyclic(3)
        self.dihedral4 = FiniteGroup.dihedral(2)  # D4

    def test_cyclic_group(self): #noqa
        g = FiniteGroup.cyclic(4)
        expected_cayley_table = [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2]
        ]
        np.testing.assert_array_equal(g.cayley_table, expected_cayley_table)

    def test_automorphisms(self):
        # Create Klein four-group V₄
        klein_four = FiniteGroup(cayley_table=np.array([
            [0, 1, 2, 3],  # e
            [1, 0, 3, 2],  # a
            [2, 3, 0, 1],  # b
            [3, 2, 1, 0]  # ab
        ]))

        actual_aut = klein_four.automorphism_group

        # Test structural properties
        self.assertEqual(actual_aut.order, 6, "Automorphism group should have order 6")

        # For V₄, all non-identity elements in Aut(V₄) have order 2
        expected_order_counts = {
            1: 1,  # identity
            2: 5  # all other elements
        }

        actual_order_counts = defaultdict(int)
        for element in range(actual_aut.order):
            element_order = len(actual_aut.orbits[element])
            actual_order_counts[element_order] += 1

        self.assertEqual(
            dict(actual_order_counts),
            expected_order_counts,
            "Automorphism group should have correct distribution of element orders"
        )

        # Verify group properties
        self.assertFalse(actual_aut.is_abelian, "Automorphism group should be non-abelian")

    def test_dihedral_group(self):
        g = FiniteGroup.dihedral(4)
        expected_cayley_table = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 0, 5, 6, 7, 4],
            [2, 3, 0, 1, 6, 7, 4, 5],
            [3, 0, 1, 2, 7, 4, 5, 6],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 6, 7, 4, 1, 2, 3, 0],
            [6, 7, 4, 5, 2, 3, 0, 1],
            [7, 4, 5, 6, 3, 0, 1, 2]
        ]
        h =  FiniteGroup(cayley_table=np.array(expected_cayley_table))
        self.assertEqual(g,h)

    def test_symmetric_group(self):
        """Test the symmetric group S₃ construction."""
        g = FiniteGroup.symmetric(3)

        # Correct Cayley table for S₃
        # Elements represent the following permutations:
        # 0: () or [0,1,2]      (identity)
        # 1: (12) or [1,0,2]    (swaps 1 and 2)
        # 2: (23) or [0,2,1]    (swaps 2 and 3)
        # 3: (13) or [2,1,0]    (swaps 1 and 3)
        # 4: (123) or [1,2,0]   (cycles 1->2->3->1)
        # 5: (132) or [2,0,1]   (cycles 1->3->2->1)
        expected_cayley_table = [
            [0, 1, 2, 3, 4, 5],
            [1, 0, 4, 5, 2, 3],
            [2, 5, 0, 4, 3, 1],
            [3, 4, 5, 0, 1, 2],
            [4, 3, 1, 2, 5, 0],
            [5, 2, 3, 1, 0, 4]
        ]
        h = FiniteGroup(cayley_table=np.array(expected_cayley_table))
        self.assertEqual(g, h)

        # Additional verifications
        self.assertEqual(len(g), 6)  # S₃ should have 6 elements
        self.assertFalse(g.is_abelian)  # S₃ is non-abelian
        self.assertEqual(len(g.center), 1)  # Center should only contain identity

    def test_init__empty_table(self):
        """Test initialization with empty Cayley table"""
        with self.assertRaises(ValueError):
            FiniteGroup(cayley_table=np.array([]))

    def test_init__non_square_table(self):
        """Test initialization with non-square Cayley table"""
        with self.assertRaises(ValueError):
            FiniteGroup(cayley_table=np.array([[0, 1], [0, 1], [0, 1]]))

    def test_init__invalid_identity(self):
        """Test initialization with invalid identity element"""
        with self.assertRaises(ValueError):
            FiniteGroup(cayley_table=np.array([[1, 0], [0, 1]]))

    def test_init__invalid_closure(self):
        """Test initialization with invalid closure"""
        with self.assertRaises(ValueError):
            FiniteGroup(cayley_table=np.array([[0, 2], [2, 0]]))

    def test_order__trivial_group(self):
        """Test order of trivial group"""
        self.assertEqual(self.trivial_group.order, 1)

    def test_order__cyclic_group(self):
        """Test order of cyclic group"""
        self.assertEqual(self.cyclic3.order, 3)

    def test_inverse__trivial_group(self):
        """Test inverse in trivial group"""
        self.assertEqual(self.trivial_group.inverse(0), 0)

    def test_inverse__cyclic_group(self):
        """Test inverse in cyclic group"""
        self.assertEqual(self.cyclic2.inverse(1), 1)

    def test_inverse__invalid_element(self):
        """Test inverse of invalid element"""
        with self.assertRaises(IndexError):
            self.cyclic2.inverse(2)

    def test_center__trivial_group(self):
        """Test center of trivial group"""
        self.assertEqual(self.trivial_group.center, {0})
        print(f'Center of trivial group: {self.trivial_group.center}...[success]')

    def test_center__cyclic_group(self):
        """Test center of cyclic group"""
        self.assertEqual(self.cyclic2.center, {0, 1})
        print(f'Center of cyclic group: {self.cyclic2.center}...[success]')

    def test_is_abelian__trivial_group(self):
        """Test if trivial group is abelian"""
        self.assertTrue(self.trivial_group.is_abelian)

    def test_is_abelian__cyclic_group(self):
        """Test if cyclic group is abelian"""
        self.assertTrue(self.cyclic3.is_abelian)

    def test_is_abelian__non_abelian_group(self):
        """Test if dihedral group is non-abelian"""
        self.assertFalse(FiniteGroup.alternating(5).is_abelian)

    def test_conjugate__trivial_group(self):
        """Test conjugation in trivial group"""
        self.assertEqual(self.trivial_group.conjugate(0, 0), 0)

    def test_conjugate__cyclic_group(self):
        """Test conjugation in cyclic group"""
        self.assertEqual(self.cyclic2.conjugate(0, 1), 1)
        self.assertEqual(self.cyclic2.conjugate(1, 1), 1)

    def test_conjugacy_class__trivial_group(self):
        """Test conjugacy class in trivial group"""
        self.assertEqual(self.trivial_group.conjugacy_class(0), {0})

    def test_conjugacy_class__cyclic_group(self):
        """Test conjugacy class in cyclic group"""
        self.assertEqual(self.cyclic2.conjugacy_class(1), {1})

    def test_conjugacy_class__invalid_element(self):
        """Test conjugacy class of invalid element"""
        with self.assertRaises(IndexError):
            self.cyclic2.conjugacy_class(2)

    def test_orbits__trivial_group(self):
        """Test orbits in trivial group"""
        self.assertEqual(self.trivial_group.orbits, {0: {0}})

    def test_orbits__cyclic_group(self):
        """Test orbits in cyclic group"""
        self.assertEqual(self.cyclic2.orbits[1], {0, 1})



if __name__ == '__main__':
    unittest.main()