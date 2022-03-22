#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest

from km3irf import Calculator


class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = Calculator()

    def test_add(self):
        assert 1 == self.calculator.add(0, 1)
        assert 3 == self.calculator.add(1, 2)

    def test_add_negative_numbers(self):
        assert -1 == self.calculator.add(0, -1)
        assert -1 == self.calculator.add(1, -2)
        assert -3 == self.calculator.add(-1, -2)

    def test_divide(self):
        assert 0 == self.calculator.divide(0, 1)
        self.assertAlmostEqual(0.5, self.calculator.divide(1, 2))

    def test_divide_by_zero_raises(self):
        with self.assertRaises(ValueError):
            self.calculator.divide(1, 0)

    def test_multiply(self):
        assert 2 == self.calculator.multiply(1, 2)
        assert 6 == self.calculator.multiply(2, 3)

    def test_multiply_with_zero(self):
        for b in range(100):
            assert 0 == self.calculator.multiply(0, b)
