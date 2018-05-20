import functools
from math import gcd
from log_configuration import logger, log_decorator
import itertools


class NotDivisibleByBasePrime(Exception):
    pass


base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]


def base_prime_wrapper(find_factor):
    def find_base_prime_factor(num):
        if num < 0:
            return 1
        else:
            for p in base_primes:
                if num % p == 0:
                    return num // p, p
            else:
                raise NotDivisibleByBasePrime()

    def find_factor_flow(number):
        try:
            return find_base_prime_factor(number)
        except NotDivisibleByBasePrime:
            return find_factor(number)

    return find_factor_flow


@base_prime_wrapper
def find_factor_by_gcd(number):
    for x in range(2, int(number ** 0.5) + 1):
        if any(map(lambda p: (max(x, p) % min(x, p)) == 0, base_primes)):
            continue
        g = gcd(x, number - x)
        if g > 1:
            return g, number // g
    else:
        return 1, number


# @log_decorator(logger)
def factor_by_gcd(number):
    # print('Factoring {} ...'.format(number))
    factor1, factor2 = find_factor_by_gcd(number)
    if factor1 == 1:
        # print('{} is a prime number'.format(number))
        return [number]
    else:
        # print('{} factor into {} X {}'.format(number, factor1, factor2))
        factors = list(itertools.chain(factor_by_gcd(factor1), factor_by_gcd(factor2)))
        factors.sort()
        return factors


# @log_decorator(logger)
def all_subsets(iterable):
    """
    This function will return all subsets of an iterable which is 2^n of set of size n
    :param iterable:
    :return: all subsets
    """
    xs = list(iterable)
    # note we return an iterator rather than a list
    return itertools.chain.from_iterable(itertools.combinations(xs, n) for n in range(len(xs) + 1))


# @log_decorator(logger)
def find_divisors(number, factor_function=factor_by_gcd):
    """
    iterate through all divisors of the number
    :param number: The number
    :param factor_function: prime factorization function
    :return: The number divisors
    """
    seen = list()
    for d in all_subsets(factor_function(number)):
        divisor = functools.reduce(int.__mul__, d, 1)
        if divisor in seen:
            continue
        else:
            seen.append(divisor)
            yield divisor


# @log_decorator(logger)
def prime_test(number, factor_function=factor_by_gcd):
    return min(factor_function(number)) == 1


