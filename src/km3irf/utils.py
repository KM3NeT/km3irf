#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .calc import Calculator


def print_multiplication_table(base):
    """Prints the multiplication table for a given base"""
    calculator = Calculator()
    for i in range(1, 11):
        print("{} x {} = {}".format(base, i, calculator.multiply(base, i)))
