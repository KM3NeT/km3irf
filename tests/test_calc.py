#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from os import path, listdir, curdir, remove
import astropy

from km3irf import Calculator
from km3irf.utils import merge_fits, list_data
from km3irf import build_aeff


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


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.test_path = path.join(path.abspath(curdir), "src", "km3irf", "data")

    def test_merge_fits(self):
        merge_fits()
        assert "all_in_one.fits" in listdir(self.test_path)

    def test_list_data(self):
        numb_fits = [i for i in listdir(self.test_path) if ".fits" in i]
        assert len(list_data()) == len(numb_fits)


class TestBuild_IRF(unittest.TestCase):
    pass
