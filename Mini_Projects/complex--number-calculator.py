# --------------
import pandas as pd
import numpy as np
import math
import cmath
#Code starts here
class complex_numbers():
    def __init__(self,real,imag):
        self.real = real
        self.imag = imag

    def __repr__(self):
        if self.real == 0.0 and self.imag == 0.0:
            return "0.00"
        if self.real == 0:
            return "%.2fi" % self.imag
        if self.imag == 0:
            return "%.2f" % self.real
        return "%.2f %s %.2fi" % (self.real, "+" if self.imag >= 0 else "-", abs(self.imag))
    
    def __add__(self,other):
        return complex(self.real + other.real,self.imag + other.imag)

    def __sub__(self,other):
        return complex(self.real - other.real,self.imag - other.imag)

    def __mul__(self, other):
        return complex(self.real*other.real - self.imag*other.imag, self.imag*other.real + self.real*other.imag)

    def __truediv__(self, other):
        sr, si, orr, oi = self.real, self.imag, other.real, other.imag 
        r = float(orr**2 + oi**2)
        return complex((sr*orr+si*oi)/r, (si*orr-sr*oi)/r)

    def absolute(self):
        a=(self.real*self.real)+(self.imag*self.imag)
        return (math.sqrt(a))

    def argument(self):
        # return cmath.phase(complex(self.real,self.imag))
        a=cmath.atan(self.imag/self.real)
        return math.degrees(a.real)
        # return (cmath.polar(complex(self.real,self.imag)))
         
    def conjugate(self):
        return (complex(self.real,self.imag)).conjugate()

comp_1=complex_numbers(3,5)
comp_2=complex_numbers(4,4)
comp_sum=comp_1.__add__(comp_2)
comp_diff=comp_1.__sub__(comp_2)
comp_prod=comp_1.__mul__(comp_2)
comp_quot=comp_1.__truediv__(comp_2)
comp_abs=comp_1.absolute()
comp_conj=comp_1.conjugate()
comp_arg=comp_1.argument()
# comp_arg=59.03

print(comp_1,comp_2,comp_sum,comp_diff,comp_prod,comp_quot,comp_abs,comp_conj,comp_arg)






