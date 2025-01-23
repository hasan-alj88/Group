from collections import Counter
from functools import lru_cache
from itertools import chain
from math import gcd, isqrt
from typing import Set, List, Tuple, Iterator


class NotDivisibleByBasePrime(Exception):
    pass


# Pre-compute prime numbers up to 1000 using sieve of Eratosthenes
def sieve_of_eratosthenes(n: int) -> List[int]:
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(n + 1) if sieve[i]]


# Extend base_primes using sieve
base_primes = sieve_of_eratosthenes(1000)


@lru_cache(maxsize=1024)
def is_prime_number(n: int) -> bool:
    """Optimized primality test using trial division."""
    if n < 2:
        return False
    if n in base_primes:
        return True
    if any(n % p == 0 for p in base_primes):
        return False
    for i in range(base_primes[-1] + 2, isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True


def base_prime_wrapper(find_factor):
    @lru_cache(maxsize=1024)
    def find_base_prime_factor(num: int) -> Tuple[int, int]:
        if num < 0:
            return 1, 1

        # Use binary search for base_primes
        left, right = 0, len(base_primes) - 1
        while left <= right:
            mid = (left + right) // 2
            p = base_primes[mid]
            if num % p == 0:
                return num // p, p
            elif p > num:
                right = mid - 1
            else:
                left = mid + 1
        raise NotDivisibleByBasePrime()

    def find_factor_flow(number: int) -> Tuple[int, int]:
        try:
            return find_base_prime_factor(number)
        except NotDivisibleByBasePrime:
            return find_factor(number)

    return find_factor_flow


@base_prime_wrapper
def find_factor_by_gcd(number: int) -> Tuple[int, int]:
    """Optimized GCD-based factorization."""
    if is_prime_number(number):
        return 1, number

    limit = isqrt(number) + 1
    # Use larger steps for bigger numbers
    step = 2 if number > 1000 else 1

    for x in range(2, limit, step):
        if x > 2 and any(x % p == 0 for p in base_primes[:10]):  # Only check first few primes
            continue
        g = gcd(x, number - x)
        if g > 1:
            return g, number // g
    return 1, number


def factor_by_gcd(number: int) -> List[int]:
    """Optimized prime factorization with memoization."""

    @lru_cache(maxsize=1024)
    def _factor(n: int) -> Tuple[int, ...]:
        if n <= 1:
            return tuple()
        if is_prime_number(n):
            return (n,)

        factor1, factor2 = find_factor_by_gcd(n)
        if factor1 == 1:
            return (n,)
        return tuple(sorted(chain(_factor(factor1), _factor(factor2))))

    return list(_factor(number))


def find_divisors(number: int, factor_function=factor_by_gcd) -> Iterator[int]:
    """Optimized divisor generation using prime factorization."""
    if number <= 0:
        yield 1
        return

    # Get prime factorization and count multiplicities
    prime_factors = Counter(factor_function(number))

    # Generate all divisors using prime factor exponents
    divisors = {1}
    for prime, count in prime_factors.items():
        new_divisors = set()
        for d in divisors:
            for i in range(1, count + 1):
                new_divisors.add(d * (prime ** i))
        divisors.update(new_divisors)

    yield from sorted(divisors)


def consecutive_pairs_up_to(n):
    end = n if n % 2 == 0 else n + 1
    numbers = iter(range(2, end + 1))
    return zip(numbers, numbers)


def is_factorial(n: int) -> bool:
    """
    Determines if a given number is a factorial of any integer.

    This function checks if the provided integer `n` can be expressed as 
    a factorial of some integer value. For `n` to be a factorial, 
    it must satisfy the mathematical property that it equals 
    the product of consecutive integers starting from 1.

    Raises errors when input is not numerical, behaves as expected for positive integers,
    and validates divisibility conditions iteratively across consecutive pairs of integers.

    Parameters:
    n: int
        The integer to check if it's a factorial. Must be greater than or equal to 1.

    Return:
    bool
        Returns True if `n` is a factorial of some integer, otherwise False.
    """
    if n < 1:
        return False
    if n <= 120:
        return n in [1, 2, 6, 24, 120]

    last_pair_dividable = True
    for i, j in consecutive_pairs_up_to(n):
        i_dividable = n % i == 0
        j_dividable = n % j == 0

        if last_pair_dividable and not i_dividable and j_dividable:
            return False
        elif last_pair_dividable and i_dividable and not j_dividable and j > 5:
            return True
        elif not last_pair_dividable and (i_dividable or j_dividable):
            return False

        last_pair_dividable = j_dividable

    return True
        
    
def power_set(sets: List[int]) -> Iterator[List[int]]:
    """
    Generate the power set of a given list.

    This function generates all possible subsets (the power set) of a given list of
    integers. It uses combinations from itertools to create subsets of all possible lengths
    and chains them together into a single iterator. The resulting power set includes the
    empty subset as well as the subset containing all elements of the input list.

    Parameters:
        sets (List[int]): A list of integers for which the power set is to be
                          generated.

    Returns:
        Iterator[List[int]]: An iterator that yields subsets of the input list.
    """
    from itertools import chain, combinations
    return chain.from_iterable(combinations(sets, r) for r in range(len(sets) + 1))
    


def powerset_unions(sets: Set[frozenset[int] | Set[int]]) -> Set[frozenset[int]]:
    """
    Returns the set of all possible unions of subsets of the given sets.

    This function processes a collection of sets, computes all possible subsets 
    of those sets, and returns the set of unions of each subset. It handles both 
    empty input and varied combinations of sets. Each resulting union is represented 
    as a frozenset to maintain immutability within the result set.

    Args:
        sets (Set[frozenset[int] | Set[int]]): A set containing individual sets (or frozensets) 
        consisting of integers. 

    Returns:
        Set[frozenset[int]]: A set of frozensets where each frozenset represents a 
        unique union of subsets from the input sets.

    Raises:
        None
    """
    # Handle empty input
    if not sets:
        return {frozenset()}

    # Convert to list for indexing
    sets_list = list(sets)
    n = len(sets_list)

    # Pre-compute binary representations for efficiency
    result = {frozenset()}
    for i in range(1, 1 << n):
        current_union = frozenset()
        # Use bit operations instead of combinations
        for j in range(n):
            if i & (1 << j):
                current_union |= sets_list[j]
        result.add(current_union)

    return result