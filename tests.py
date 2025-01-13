from unittest import TestCase, TestResult
from Permutation import Permutation


class DescriptionTestResult(TestResult):
    def addError(self, test, err):
        print(f"\nTest Description: {test.__doc__}")
        super().addError(test, err)

    def addFailure(self, test, err):
        print(f"\nTest Description: {test.__doc__}")
        super().addFailure(test, err)


class TestPermutation(TestCase):
    def run(self, result=None):
        if result is None:
            result = DescriptionTestResult()
        return super().run(result)

    # Constructor Tests
    def test_init_empty(self):
        """Test initialization with no parameters."""
        p = Permutation()
        print(f"Empty permutation swaps: {p.swaps}")
        self.assertEqual(p.swaps, [])

    def test_init_swaps(self):
        """Test initialization with explicit swaps."""
        swaps = [(0,1), (1,2)]
        p = Permutation(swaps=swaps)
        print(f"Initialized with swaps: {swaps}")
        print(f"Resulting swaps: {p.swaps}")
        self.assertEqual(p.swaps, swaps)

    # of_the_array Tests
    def test_of_the_array_identity(self):
        """Test of_the_array with identity permutation [0,1,2]."""
        arr = [0, 1, 2]
        p = Permutation.of_the_array(arr)
        print(f"Input array: {arr}")
        print(f"Result permutation: {p}")
        self.assertEqual(p.permuted_array, [])
        self.assertEqual(str(p), "()")

    def test_of_the_array_swap(self):
        """Test of_the_array with simple swap [1,0,2]."""
        arr = [1, 0, 2]
        p = Permutation.of_the_array(arr)
        print(f"Input array: {arr}")
        print(f"Result permutation: {p}")
        self.assertEqual(p.permuted_array, [1, 0])
        self.assertEqual(str(p), "(0,1)")

    def test_of_the_array_cycle(self):
        """Test of_the_array with cycle [1,2,0]."""
        arr = [1, 2, 0]
        p = Permutation.of_the_array(arr)
        print(f"Input array: {arr}")
        print(f"Result permutation: {p}")
        self.assertEqual(p.permuted_array, arr)
        self.assertEqual(str(p), "(0,1,2)")

    def test_of_the_array_duplicates(self):
        """Test of_the_array with invalid duplicate elements [1,1,2]."""
        arr = [1, 1, 2]
        print(f"Testing array with duplicates: {arr}")
        with self.assertRaises(ValueError) as context:
            Permutation.of_the_array(arr)
        print(f"Error raised: {str(context.exception)}")

    # of_cyclic_notation Tests
    def test_cyclic_notation_empty(self):
        """Test empty cycle notation '()' creates identity permutation."""
        notation = "()"
        p = Permutation.of_cyclic_notation(notation)
        print(f"Cycle notation: {notation}")
        print(f"Resulting permutation: {p}")
        self.assertEqual(p.permuted_array, [])
        self.assertEqual(str(p), "()")

    def test_cyclic_notation_swap(self):
        """Test simple swap cycle notation '(0,1)' creates correct permutation."""
        notation = "(0,1)"
        p = Permutation.of_cyclic_notation(notation)
        print(f"Cycle notation: {notation}")
        print(f"Resulting permutation: {p}")
        self.assertEqual(p.permuted_array, [1, 0])
        self.assertEqual(str(p), "(0,1)")

    def test_cyclic_notation_invalid_chars(self):
        """Test cycle notation with invalid characters '(a,b)' raises error."""
        notation = "(a,b)"
        print(f"Testing invalid notation: {notation}")
        with self.assertRaises(ValueError) as context:
            Permutation.of_cyclic_notation(notation)
        print(f"Error raised: {str(context.exception)}")

    # permuted_array Tests
    def test_permuted_array_trailing_identity(self):
        """Test permuted_array removes trailing identity mappings [0,2,1,3,4] -> [0,2,1]."""
        arr = [0, 2, 1, 3, 4]
        p = Permutation.of_the_array(arr)
        expected = [0, 2, 1]
        print(f"Input array: {arr}")
        print(f"Expected result: {expected}")
        print(f"Actual result: {p.permuted_array}")
        self.assertEqual(p.permuted_array, expected)

    def test_permuted_array_keep_required(self):
        """Test permuted_array keeps identity mappings required for validation."""
        arr = [2, 1, 0, 3]
        p = Permutation.of_the_array(arr)
        expected = [2, 1, 0]
        print(f"Input array: {arr}")
        print(f"Expected result: {expected}")
        print(f"Actual result: {p.permuted_array}")
        self.assertEqual(p.permuted_array, expected)

    # inverse Tests
    def test_inverse_swap(self):
        """Test inverse of swap permutation."""
        p = Permutation.of_cyclic_notation("(0,1)")
        inv = p.inv
        print(f"Original permutation: {p}")
        print(f"Inverse permutation: {inv}")
        self.assertEqual((p + inv).permuted_array, [])
        self.assertEqual(str(p + inv), "()")

    def test_inverse_cycle(self):
        """Test inverse of cycle permutation."""
        p = Permutation.of_cyclic_notation("(0,1,2)")
        inv = p.inv
        print(f"Original permutation: {p}")
        print(f"Inverse permutation: {inv}")
        self.assertEqual((p + inv).permuted_array, [])
        self.assertEqual(str(p + inv), "()")
    # ---------------------------------------------------
    # - addition (+) operator
    def _addition_result(self, p1, p2, expected):
        result = p1 + p2
        print(f"Permutation 1: {p1}")
        print(f"Permutation 2: {p2}")
        print(f"Result: {result}")
        self.assertEqual(result.permuted_array, expected)

    def test_addition_identity(self):
        """Test addition of identity permutation."""
        p = Permutation.of_the_array([0, 1, 2])
        self._addition_result(p, Permutation(), [])

    def test_addition_simple(self):
        """Test addition of two simple permutations: (0,1) + (1,2).
        This should result in a 3-cycle (0,1,2)."""
        p1 = Permutation.of_cyclic_notation("(0,1)")
        p2 = Permutation.of_cyclic_notation("(1,2)")
        expected = Permutation.of_cyclic_notation("(0,1,2)")

        print(f"p1: {p1}, array: {p1.permuted_array}")
        print(f"p2: {p2}, array: {p2.permuted_array}")
        result = p1 + p2
        print(f"p1 + p2: {result}, array: {result.permuted_array}")
        print(f"expected: {expected}, array: {expected.permuted_array}")

        self.assertEqual(result, expected)

    def test_addition_swap_with_swap(self):
        """Test adding two swap permutations: [1,0] + [0,1].
        Should result in identity permutation []."""
        p1 = Permutation.of_the_array([1, 0])  # (0,1)
        p2 = Permutation.of_the_array([1, 0])  # (0,1)
        self._addition_result(p1, p2, [])

    def test_addition_cycle_with_inverse(self):
        """Test adding a 3-cycle with its inverse: [1,2,0] + [2,0,1].
        Should result in identity permutation []."""
        p1 = Permutation.of_cyclic_notation("(0,1,2)")  # [1,2,0]
        p2 = Permutation.of_cyclic_notation("(0,2,1)")  # [2,0,1]
        result = p1 + p2
        print(f"Permutation 1 array: {p1.permuted_array} cycles: {p1}")
        print(f"Permutation 2 array: {p2.permuted_array} cycles: {p2}")
        print(f"Result array: {result.permuted_array} cycles: {result}")
        self.assertEqual(result.permuted_array, [])
        self.assertEqual(str(result), "()")

    def test_addition_disjoint_cycles(self):
        """Test adding two disjoint cycles: (0,1) + (2,3).
        Should result in both cycles (0,1)(2,3)."""
        p1 = Permutation.of_cyclic_notation("(0,1)")  # [1,0,2,3]
        p2 = Permutation.of_cyclic_notation("(2,3)")  # [0,1,3,2]
        result = p1 + p2
        print(f"Permutation 1 array: {p1.permuted_array} cycles: {p1}")
        print(f"Permutation 2 array: {p2.permuted_array} cycles: {p2}")
        print(f"Result array: {result.permuted_array} cycles: {result}")
        expected = [1, 0, 3, 2]  # Combined effect of both swaps
        self.assertEqual(result.permuted_array, expected)
        self.assertEqual(str(result), "(0,1)(2,3)")

    def test_addition_overlapping_cycles(self):
        """Test adding two overlapping cycles: (0,1,2) + (1,2,3).
        Should result in a permutation that combines their effects."""
        p1 = Permutation.of_cyclic_notation("(0,1,2)")  # [1,2,0,3]
        p2 = Permutation.of_cyclic_notation("(1,2,3)")  # [0,2,3,1]
        result = p1 + p2
        print(f"Permutation 1 array: {p1.permuted_array} cycles: {p1}")
        print(f"Permutation 2 array: {p2.permuted_array} cycles: {p2}")
        print(f"Result array: {result.permuted_array} cycles: {result}")
        expected = Permutation.of_cyclic_notation("(0,1)(2,3)")
        self.assertEqual(result, expected)

    def test_addition_with_identity(self):
        """Test adding any permutation with identity permutation.
        Should result in the original permutation."""
        p1 = Permutation.of_cyclic_notation("(0,1,2)")  # [1,2,0]
        p2 = Permutation()  # Identity permutation
        result = p1 + p2
        print(f"Permutation 1 array: {p1.permuted_array} cycles: {p1}")
        print(f"Permutation 2 array: {p2.permuted_array} cycles: {p2}")
        print(f"Result array: {result.permuted_array} cycles: {result}")
        self.assertEqual(result, p1)

    # - subtraction (-) operator
    def test_subtraction_identity(self):
        """Test subtraction of identity permutation."""
        p = Permutation.of_cyclic_notation("(0,1,2)")
        result = p - Permutation()
        print(f"Original permutation: {p}")
        print(f"Identity permutation: {Permutation()}")
        print(f"Result: {result}")
        self.assertEqual(result, p)

    def test_subtraction_simple(self):
        """Test subtraction of two simple swap permutations."""
        p1 = Permutation.of_cyclic_notation("(0,1)")
        p2 = Permutation.of_cyclic_notation("(1,2)")
        result = p1 - p2
        expected = Permutation.of_cyclic_notation("(0,1,2)")

        print(f"p1: {p1}")
        print(f"p2: {p2}")
        print(f"p2.inv: {p2.inv}")
        print(f"result = p1 - p2: {result}")
        print(f"expected: {expected}")

        self.assertEqual(result, expected)

    def test_subtraction_cycle(self):
        """Test subtraction of cycle permutations: (0,1,2) - (0,2,1).
        p1 = (0,1,2)  maps: 0->1, 1->2, 2->0
        p2 = (0,2,1)  maps: 0->2, 2->1, 1->0
        p2.inv = (0,1,2)
        Therefore p1 - p2 = (0,1,2) + (0,1,2) = (0,2,1)"""
        p1 = Permutation.of_cyclic_notation("(0,1,2)")
        p2 = Permutation.of_cyclic_notation("(0,2,1)")
        expected = Permutation.of_cyclic_notation("(0,2,1)")

        print(f"p1: {p1}, maps: {p1.permuted_array}")
        print(f"p2: {p2}, maps: {p2.permuted_array}")
        print(f"p2.inv: {p2.inv}, maps: {p2.inv.permuted_array}")
        result = p1 - p2
        print(f"p1 - p2: {result}, maps: {result.permuted_array}")
        print(f"expected: {expected}, maps: {expected.permuted_array}")

        self.assertEqual(result, expected)

    def test_subtraction_multi_cycle_permutations(self):
        """Test subtraction of two multi-cycle permutations."""
        p1 = Permutation.of_cyclic_notation("(0,1)(2,3)")
        p2 = Permutation.of_cyclic_notation("(0,2)(1,3)")
        result = p1 - p2
        expected = Permutation.of_cyclic_notation("(0,3)(1,2)")

        print(f"p1: {p1}, maps: {p1.permuted_array}")
        print(f"p2: {p2}, maps: {p2.permuted_array}")
        print(f"p2.inv: {p2.inv}, maps: {p2.inv.permuted_array}")
        print(f"result = p1 - p2: {result}, maps: {result.permuted_array}")
        print(f"expected: {expected}, maps: {expected.permuted_array}")

        self.assertEqual(result, expected)

    # - equality (==) operator
    def test_equality_identity(self):
        """Test equality of identity permutation."""
        p1 = Permutation()
        p2 = Permutation()
        print(f"Permutation 1: {p1}")
        print(f"Permutation 2: {p2}")
        self.assertEqual(p1, p2)

    def test_equality_simple(self):
        """Test equality of two simple swap permutations."""
        p1 = Permutation.of_cyclic_notation("(0,1)")
        p2 = Permutation.of_the_array([1, 0])
        print(f"Permutation 1: {p1}")
        print(f"Permutation 2: {p2}")
        self.assertEqual(p1, p2)

    def test_equality_cycle(self):
        """Test equality of two simple cycle permutations."""
        p1 = Permutation.of_the_array([1, 2, 0])
        p2 = Permutation.of_cyclic_notation("(0,1,2)")
        print(f"Permutation 1: {p1}")
        print(f"Permutation 2: {p2}")
        self.assertEqual(p1, p2)

    def test_equality_multi_cycle_permutations(self):
        """Test equality of two multi-cycle permutations."""
        p1 = Permutation.of_the_array([1, 0, 2, 4, 3])
        p2 = Permutation.of_cyclic_notation("(0,1)(3,4)")

        print(f"Permutation 1: {p1}")
        print(f"Permutation 2: {p2}")
        self.assertEqual(p1, p2)

    # - cycles property
    def test_cycles_identity(self):
        """Test cycles of identity permutation."""
        p = Permutation()
        print(f"Identity permutation: {p}")
        print(f"Cycles: {p.cycles}")
        self.assertEqual(p.cycles, [])

    def test_cycles_simple(self):
        """Test cycles of simple swap permutation."""
        p = Permutation.of_cyclic_notation("(0,1)")
        print(f"Simple swap permutation: {p}")
        print(f"Cycles: {p.cycles}")
        self.assertEqual(p.cycles, [[0, 1]])

    def test_cycles_cycle(self):
        """Test cycles of simple cycle permutation."""
        p = Permutation.of_cyclic_notation("(0,1,2)")
        print(f"Simple cycle permutation: {p}")
        print(f"Cycles: {p.cycles}")
        self.assertEqual(p.cycles, [[0, 1, 2]])

    def test_cycles_multi_cycle_permutations(self):
        """Test cycles of multi-cycle permutation."""
        p = Permutation.of_cyclic_notation("(0,1)(3,4)")
        print(f"Multi-cycle permutation: {p}")
        print(f"Cycles: {p.cycles}")
        self.assertEqual(p.cycles, [[0, 1], [3, 4]])

    # - generator method
    def test_generator(self):
        """Test generator method."""
        n = 3
        permutations = list(Permutation.generator(n))
        print(f"Permutations of size {n}: {permutations}")
        self.assertEqual(len(permutations), 6)

    # - random method
    def test_random(self):
        """Test random method."""
        n = 3
        p = Permutation.random(n)
        print(f"Random permutation of size {n}: {p}")
        self.assertEqual(len(p.permuted_array), n)