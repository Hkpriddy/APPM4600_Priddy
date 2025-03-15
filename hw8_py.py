# -*- coding: utf-8 -*-
"""HW8.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FB68hYbJ3YeZ9jvmWHr_NmWamLhhA4lS
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
from scipy.interpolate import CubicHermiteSpline

def driver():

    f = lambda x: 1/(1+x**2)
    df = lambda x: -2*x/(1+x**2)**2
    N = 5
    xi = []
    yi = []
    zi = []
    h = 10/(N - 1)
    for i in range(N):
      #xi.append(-5+(i)*h)
      xi.append(-5*np.cos((2*i*np.pi)/(2*N)))
      yi.append(f(xi[i]))
      zi.append(df(xi[i]))

    xi = np.array(xi)
    yi = np.array(yi)
    zi = np.array(zi)

    #print("xi = ", xi)
    newx = np.linspace(-5,5,100)

    Hermy = CubicHermiteSpline(xi,yi, zi)
    newyherm = Hermy(newx)
    errherm = np.abs(newyherm - f(newx))

    (A,B,C) = create_natural_spline(yi, xi, N)
    newynat = eval_cubic_spline(newx,len(newx),xi,N,A,B,C)
    errnat = np.abs(newynat-f(newx))



    (A,B,C) = create_CLAMP_spline(yi, xi, N)

    newyclamp = eval_cubic_spline(newx,len(newx),xi,N,A,B,C)
    errclamp = np.abs(newyclamp-f(newx))

    newyGrang = barycentric_GRUNGE(f, xi, newx, N)
    errGrang = np.abs(newyGrang-f(newx))
    #print("newy = ", newy)

    plt.figure()

    plt.plot(newx, newynat)
    plt.plot(newx, newyclamp)
    plt.plot(newx, newyGrang)
    plt.plot(newx, newyherm)
    plt.plot(newx, f(newx))

    plt.title("Plotting different interpolation methods - N = 5 nodes (Chebyshev)")
    plt.plot(xi,yi,'o')
    plt.legend([ 's(x) Natural', 's(x) Clamped', 'p(x) Lagrange', 'Hermite spline', 'f(x)'])

    plt.figure()
    plt.plot(newx, errnat)
    plt.plot(newx, errclamp)
    plt.plot(newx, errGrang)
    plt.plot(newx, errherm)
    plt.yscale('log')
    plt.title("Error in different interpolation methods - N = 5 nodes (Chebyshev)")
    plt.legend(['s(x) Natural', 's(x) Clamped', 'p(x) Lagrange', 'Hermite spline'])



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


def barycentric_GRUNGE(f,x,newx,N):
  y = f(x)
  p = np.linspace(1,1,len(newx))
  phi = np.linspace(1,1,len(newx))
  for i in range(len(newx)):
    for j in range(N):
      phi[i] = phi[i]*(newx[i] - x[j])

  omega = np.linspace(1,1,len(x))
  for i in range(len(x)):
    for j in range(len(x)):
      if j != i:
        omega[i] = omega[i] * 1/(x[i] - x[j])
  for i in range(len(newx)):
    bigsum = 0
    for j in range(len(x)):
      bigsum += omega[j]/(newx[i] - x[j])*f(x[j])
    p[i] = phi[i]*bigsum

  return(p)

def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N-1);
#  vector values
    h = np.zeros(N);
    h[0] = xint[1]-xint[0]
    for i in range(1,N-1):
       h[i] = xint[i+1] - xint[i]
       b[i] = ((yint[i+1]-yint[i])/h[i] - (yint[i]-yint[i-1])/h[i-1])/(h[i-1]+h[i]);
       #print("b = ", b)
#  create the matrix M so you can solve for the A values
    M = np.zeros((N-1,N-1));
    for i in np.arange(N-1):
        M[i,i] = 4/12;

        if i<(N-2):
            M[i,i+1] = h[i+1]/(6*(h[i]+h[i+1]));

        if i>0:
            M[i,i-1] = h[i]/(6*(h[i]+h[i-1]));
    #print("M =", M)
# Solve system M*A = b to find coefficients (a[1],a[2],...,a[N-1]).
    A = np.zeros(N)
    A[0:N-1] = np.linalg.solve(M,b)
    #print("A = ", A)

    A[N-1] = A[0]
#  Create the linear coefficients
    B = np.zeros(N)
    C = np.zeros(N)
    for j in range(len(B)-1):
       B[j] = yint[j-1] - A[j-1] * h[j]**2/6# find the C coefficients
       C[j] = yint[j] - A[j] * h[j]**2/6# find the D coefficients
    #print("h = ", h)

    return(A,B,C)


def create_CLAMP_spline(yint,xint,N):
#    create the right  hand side for the linear system
    b = np.zeros(N-1);
#  vector values
    h = np.zeros(N);
    h[0] = xint[1]-xint[0]
    for i in range(1,N-1):
       h[i] = xint[i+1] - xint[i]
       b[i] = ((yint[i+1]-yint[i])/h[i] - (yint[i]-yint[i-1])/h[i-1])/(h[i-1]+h[i]);
       #print("b = ", b)
#  create the matrix M so you can solve for the A values
    M = np.zeros((N-1,N-1));
    for i in np.arange(N-1):

        if i == 0 or i == N-1:
            M[i,i] = 2*h[i]/12
        else:
            M[i,i] = 4*h[i]/12;

        if i<(N-2):
            M[i,i+1] = h[i+1]/(6*(h[i]+h[i+1]));

        if i>0:
            M[i,i-1] = h[i]/(6*(h[i]+h[i-1]));
    #print("M =", M)
# Solve system M*A = b to find coefficients (a[1],a[2],...,a[N-1]).
    A = np.zeros(N)
    A[0:N-1] = np.linalg.solve(M,b)
    #print("A = ", A)

    A[N-1] = A[0]
#  Create the linear coefficients
    B = np.zeros(N)
    C = np.zeros(N)
    for j in range(len(B)-1):
       B[j] = yint[j-1] - A[j-1] * h[j]**2/6# find the C coefficients
       C[j] = yint[j] - A[j] * h[j]**2/6# find the D coefficients
    #print("h = ", h)

    return(A,B,C)


def eval_local_spline(xeval,xi,xip,Ai,Aip,B,C):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Aip = A_{i}; Ai = A_{i-1}

    hi = xip-xi;

    yeval = 1/hi * ((-Ai*(xeval-xip)**3)/(6) + Aip * ((xeval-xi)**3)/6 - B*(xeval-xip) + C*(xeval-xi))

    #print("Term 1 = ", Ai*(xip-xeval)**3/(6))

    return yeval;





def  eval_cubic_spline(xeval,Neval,xint,Nint,A,B,C):

    yeval = np.zeros(Neval);

    for j in range(1, Nint-1):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j-1];
        btmp= xint[j];

#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp));
        xloc = xeval[ind];

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,A[j-1],A[j],B[j],C[j])
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)



#def Harmite(f,df, x, newx, N):



driver()