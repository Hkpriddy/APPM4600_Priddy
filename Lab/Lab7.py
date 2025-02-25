###PRELAB####
#
# We solve the system X^-1y = a
#  y = [y0,y1,y2,...yn] (Input data)
#  a = [a0,a1,a2,...an] (coefficient vec)
#  X =
#  [1,x0,x0^2,...,x0^n]
#  [1,x1,x1^2,...,x1^n]
#  [1,x2,x2^2,...,x2^n]
#  [ :             ]
#  [1,xn,xn^2,...,xn^n]
#
# (n by n)
#
# I will write 2 functions, one solving for the Vandermonde matrix ad the other evaluating the inverse and y (using np.linalge.solve)
# for a, our matrix of coefficients.
#
#

import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
import matplotlib;
from scipy.special import comb


def vandy(x,N):
  V = np.zeros((N,N))
  for i in range(len(x)):
    V[:,i] = x**i
  return V

def MonoSolver(x, f, newx, N):
  V = vandy(x,N)
  y = f(x)
  a = np.linalg.solve(V,y)
  newy = np.linspace(0,1,len(newx))
  for i in range(len(newx)):
    temp = 0
    for j in range(len(x)):
      temp =  a[j]*newx[j]**j
    newy[i] = temp
  return newy


def lagrange_interp(f,xint,xtrg):

    n = len(xint);
    mtrg = len(xtrg);

    # Evaluate a matrix L of size mtrg x n+1 where L[:,j] = Lj(xtrg)
    L = np.ones((mtrg,n));
    w=np.ones(n); psi=np.ones(mtrg);

    for i in range(n):
        for j in range(n):
            if np.abs(j-i)>0:
                w[i] = w[i]*(xint[i]-xint[j]);
        psi = psi*(xtrg-xint[i]);
    w = 1/w;

    fj = 1/(np.transpose(np.tile(xtrg,(n,1))) - np.tile(xint,(mtrg,1)));
    L = fj*np.transpose(np.tile(psi,(n,1)))*np.tile(w,(mtrg,1));

    # Polynomial interpolant is L*y, where y = f(xint)
    g = L@f(xint);
    return g;

def newton_interp(f,xint,xtrg,srt=False):

    n = len(xint)-1;

    # option: sort points
    if srt:
        xint=sortxi(xint);

    fi=f(xint);                 #function values

    D = np.zeros((n+1,n+1)); #matrix of divided differences
    D[0]=fi;
    for i in np.arange(1,n+1):
        # Compute divided differences at row i using row i-1.
        D[i,0:n+1-i]=(D[i-1,1:n+2-i] - D[i-1,0:n+1-i])/(xint[i:n+1] - xint[0:n+1-i]);

    cN = D[:,0]; #Interpolation coefficients are stored on the first column of D.

    # Evaluation (Horner's rule)
    g = cN[n]*np.ones(len(xtrg)); #constant term
    for i in np.arange(n-1,-1,-1):
        g = g*(xtrg-xint[i]) + cN[i];

    return (g,cN);





N = 6
def f(x):
  return 1/(1+(10*x)**2)

x = np.linspace(0,1,N)
for i in range(len(x)):
  x[i] = (-1 + (i - 1)*(1/(N-1)))

newx = np.linspace(-1,1,1000)

newy = MonoSolver(x,f,newx, N)

plt.plot(newx, newy)
plt.plot(newx, f(newx))


