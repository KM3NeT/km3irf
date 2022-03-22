#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Calculator:
    def add(self, a, b):
        return a + b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Undefined division for b=0")
        return a / b

    def multiply(self, a, b):
        return a * b
