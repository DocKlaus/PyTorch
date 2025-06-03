from itertools import product


def combo_from_itertools(product_in: list[int], length: int) -> list[list[int]]:
    """ Генерация листа комбинаций с помощью стандартной библиотеки itertools + list comprehensions"""
    return [list(combination) for combination in product(product_in, repeat=length)]


def combo_from_recurs(product_in: list[int], length: int) -> list[list[int]]:
    """ Разворот itertools"""

    base = len(product_in)
    total = base ** length

    combinations = []
    for i in range(total):
        combo = []
        num = i
        for _ in range(length):
            combo.append(product_in[num % base])
            num = num // base
        combinations.append(combo[::-1])
    return combinations

print(combo_from_itertools(product_in = [0, 1], length = 4))
print(combo_from_recurs(product_in = [0, 1], length = 4))
