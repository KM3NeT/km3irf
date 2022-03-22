"""
How to use the Calculator
=========================

The following example shows how to use the `Calculator` class.
"""
from km3irf import Calculator

#####################################################
# Initialising a Calculator
# -------------------------
# Initialising is a simple as this:

calc = Calculator()

#####################################################
# Adding numbers
# ~~~~~~~~~~~~~~
#
# You can add two numbers by calling the `.add(a, b)` method:

print(calc.add(5, 23))
