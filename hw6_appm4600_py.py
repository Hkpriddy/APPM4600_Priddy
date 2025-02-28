# -*- coding: utf-8 -*-
"""HW6_Appm4600.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ahYtcuF_oNXhWOWXYLQwkYWsNuyDQBA_
"""

import numpy as np
from numpy import random as rand
import time
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt


##############Q1!################


# Newton method in n dimensions implementation
def newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    Flist = []
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix
        Jn = Jf(xn);
        nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;
        Flist.append(np.linalg.norm(Fn))

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ, Flist);

# Lazy Newton method (chord iteration) in n dimensions implementation
def lazy_newton_method_nd_simple(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;
    Flist = []
    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        if np.linalg.norm(Fn) == np.inf:
          break
        # compute n x n Jacobian matrix only if n==0
        if (n==0):
            Jn = Jf(xn);
            nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step
        pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;
        Flist.append(np.linalg.norm(Fn))
    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Lazy Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Lazy Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ, Flist);
def broyden_method_nd(f,B0,x0,tol,nmax,Bmat='Id',verb=False):

    # Initialize arrays and function value
    d = x0.shape[0];
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1;
    npn=1;

    #####################################################################
    # Create functions to apply B0 or its inverse
    if Bmat=='fwd':
        #B0 is an approximation of Jf(x0)
        # Use pivoted LU factorization to solve systems for B0. Makes lusolve O(n^2)
        lu, piv = lu_factor(B0);
        luT, pivT = lu_factor(B0.T);

        def Bapp(x): return lu_solve((lu, piv), x); #np.linalg.solve(B0,x);
        def BTapp(x): return lu_solve((luT, pivT), x) #np.linalg.solve(B0.T,x);
    elif Bmat=='inv':
        #B0 is an approximation of the inverse of Jf(x0)
        def Bapp(x): return B0 @ x;
        def BTapp(x): return B0.T @ x;
    else:
        Bmat='Id';
        #default is the identity
        def Bapp(x): return x;
        def BTapp(x): return x;
    ####################################################################
    # Define function that applies Bapp(x)+Un*Vn.T*x depending on inputs
    def Inapp(Bapp,Bmat,Un,Vn,x):
        rk=Un.shape[0];

        if Bmat=='Id':
            y=x;
        else:
            y=Bapp(x);

        if rk>0:
            y=y+Un.T@(Vn@x);

        return y;
    #####################################################################

    # Initialize low rank matrices Un and Vn
    Un = np.zeros((0,d)); Vn=Un;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");
    Flist = []
    while npn>tol and n<=nmax:
        if verb:
            print("|--%d--|%1.7f|%1.12f|" % (n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        #Broyden step xn = xn -B_n\Fn
        dn = -Inapp(Bapp,Bmat,Un,Vn,Fn);
        # Update xn
        xn = xn + dn;
        npn=np.linalg.norm(dn);

        ###########################################################
        ###########################################################
        # Update In using only the previous I_n-1
        #(this is equivalent to the explicit update formula)
        Fn1 = f(xn);
        dFn = Fn1-Fn;
        nf+=1;
        I0rn = Inapp(Bapp,Bmat,Un,Vn,dFn); #In^{-1}*(Fn+1 - Fn)
        un = dn - I0rn;                    #un = dn - In^{-1}*dFn
        cn = dn.T @ (I0rn);                # We divide un by dn^T In^{-1}*dFn
        # The end goal is to add the rank 1 u*v' update as the next columns of
        # Vn and Un, as is done in, say, the eigendecomposition
        Vn = np.vstack((Vn,Inapp(BTapp,Bmat,Vn,Un,dn)));
        Un = np.vstack((Un,(1/cn)*un));
        Flist.append(np.linalg.norm(Fn))
        n+=1;
        Fn=Fn1;
        rn = np.vstack((rn,xn));

    r=xn;

    if verb:
        if npn>tol:
            print("Broyden method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Broyden method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return(r,rn,nf, Flist)



def f(x):
  X = x[0]**2 + x[1]**2 - 4

  Y = np.exp(x[0]) + x[1] - 1

  return np.array([X,Y])

def Jf(x):

  return np.array([[2*x[0], 2*x[1]],[np.exp(x[0]), 1]])

'''
x = np.array([1,1])
#x = np.array([1,-1])

#x = np.array([0,0])

(r,rn,nf,nJ, Flist) = newton_method_nd(f,Jf,x,10**(-10),50,True)
plt.figure()
plt.plot(Flist)
plt.yscale("log")
plt.xlabel("n iterations")
plt.ylabel("Absolute Error: norm(F(x))")
plt.title("Comparing the error by iteration of the quasi newton methods")

(r,rn,nf,nJ, List) = lazy_newton_method_nd_simple(f,Jf,x,10**(-10),100,True)
plt.plot(List)


(r,rn,nf, Blist) = broyden_method_nd(f,Jf(x),x,10**(-10),50,'fwd',True)
plt.plot(Blist)

plt.legend(["Newton Method","Lazy Newton Method","Broyden Method"])
'''

##############Q2!################

def line_search(f,Gf,x0,p,type,mxbck,c1,c2):
    alpha=2;
    n=0;
    cond=False; #condition (if True, we accept alpha)
    f0 = np.linalg.norm(f(x0)); # initial function value
    Gdotp = np.linalg.norm(p.T @ Gf(x0));
 #initial directional derivative
    nf=1;ng=1; # number of function and grad evaluations

    # we backtrack until our conditions are met or we've halved alpha too much
    while n<=mxbck and (not cond):
        alpha=0.5*alpha;
        x1 = x0+(-alpha*p);

        # Armijo condition of sufficient descent. We draw a line and only accept
        # a step if our function value is under this line.
        Armijo = np.linalg.norm(f(x1)) <= f0 + c1*alpha*Gdotp;

        nf+=1;
        if type=='wolfe':
            #Wolfe (Armijo sufficient descent and simple curvature conditions)
            # that is, the slope at new point is lower
            Curvature = np.linalg.norm(p.T @ Gf(x1)) >= c2*Gdotp;
            # condition is sufficient descent AND slope reduction
            cond = Armijo and Curvature;
            ng+=1;
            print(Curvature)
        elif type=='swolfe':
            #Symmetric Wolfe (Armijo and symmetric curvature)
            # that is, the slope at new point is lower in absolute value
            Curvature = np.abs(p.T @ Gf(x1)) <= c2*np.abs(Gdotp);
            # condition is sufficient descent AND symmetric slope reduction
            cond = Armijo and Curvature;
            ng+=1;
        else:
            # Default is Armijo only (sufficient descent)
            cond = Armijo;

        n+=1;

    return(x1,alpha,nf,ng);



# Steepest descent algorithm
def steepest_descent(f,Gf,x0,tol,nmax,type='swolfe',verb=True):
    # Set linesearch parameters
    c1=1e-3; c2=0.9; mxbck=3;
    # Initialize alpha, fn and pn
    alpha=1;
    xn = x0; #current iterate
    rn = x0; #list of iterates
    fn = f(xn); nf=1; #function eval
    pn = np.matmul(Gf(xn).T, fn); ng=1; #gradient eval
    # if verb is true, prints table of results
    if verb:
        print("|--n--|-alpha-|----|xn|----|---|f(xn)|---|---|Gf(xn)|---|");

    # while the size of the step is > tol and n less than nmax
    n=0;
    while n<=nmax and np.linalg.norm(pn)>tol:
        if verb:
            print("|--%d--|%1.5f|%1.7f|%1.7f|%1.7f|" %(n,alpha,np.linalg.norm(xn),np.linalg.norm(fn),np.linalg.norm(pn)));

        # Use line_search to determine a good alpha, and new step xn = xn + alpha*pn
        (xn,alpha,nfl,ngl)=line_search(f,Gf,xn,pn,type,mxbck,c1,c2);

        nf=nf+nfl; ng=ng+ngl; #update function and gradient eval counts
        fn = f(xn); #update function evaluation
        pn = np.matmul(Gf(xn).T, fn); # update gradient evaluation
        n+=1;
        rn=np.vstack((rn,xn)); #add xn to list of iterates

    r = xn; # approx root is last iterate

    return (r,rn,nf,ng);


def franken(f,Jf,x0,tol,nmax,verb=False):
  (r,rn,nf,ng) = steepest_descent(f,Jf,x0,10**(-2),nmax,'',True)
  return newton_method_nd(f,Jf,r,tol,nmax,True)


def F(x):
  f = x[0] + np.cos(x[0]*x[1]*x[2])-1
  g = (1-x[0])**.25 + x[1] + 0.05*x[2]**2 - 0.15*x[2] - 1
  h = -x[0]**2 - 0.1*x[1]**2 + 0.01*x[1] + x[2] - 1
  return np.array([f,g,h])

def JF(x):
  row1 = np.array([1-x[1]*x[2]*np.sin(x[0]*x[1]*x[2]), -x[0]*x[2]*np.sin(x[0]*x[1]*x[2]), -x[1]*x[0]*np.sin(x[0]*x[1]*x[2])])
  row2 = np.array([-1/(4*(1-x[0])**.75), 1, 0.1*x[2] - 0.15])
  row3 = np.array([-2*x[0], -0.2*x[1] + 0.01, 1])
  return np.array([row1,row2,row3])
x = np.array([0.2,1,1])

print("Newton Method: \n")
(r,rn,nf,nJ, Flist) = newton_method_nd(F,JF,x,10**(-6),50,True)
print("r = " ,r)
print("Steepest Descent: \n")
(r,rn,nf,ng) = steepest_descent(F,JF,x,10**(-6),100, "", True)
print("r = " ,r)
print("Franken Method: \n")
(r,rn,nf,nJ, Flist) = franken(F,JF,x,10**(-6),50,True)