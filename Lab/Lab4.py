import numpy as np
import matplotlib.pyplot as plt

#
#       log|(p_{n+1} - P_n)/(P_n - P_{n-1})|
# a = ----------------------------------------
#     log|(p_n - P_{n-1})/(P_{n-1} - P_{n-2})|


def g(x):
	return  (10/(x+4))**.5
def dg(x):
	return -0.5 * 10**.5 * (x+4)**(-1.5) 
tol = 10**(-10)


def fixed_point_method(g,dg,x0,a,b,tol,nmax,vrb=False):
     # Fixed point iteration method applied to find the fixed point of g from starting point x0

     # Initial values
     n=0;
     xn = x0;
     # Current guess is stored at rn[n]
     rn=np.array([xn]);
     r=xn;

     if vrb:
         print("\n Fixed point method with nmax=%d and tol=%1.1e\n" % (nmax, tol));
         print("\n|--n--|----xn----|---|g(xn)|---|---|g'(xn)---|");

     while n<=nmax:
         if vrb:
             print("|--%d--|%1.8f|%1.8f|%1.4f|" % (n,xn,np.abs(g(xn)),np.abs(dg(xn))));

         # If the estimate is approximately a root, get out of while loop
         if np.abs(g(xn)-xn)<tol:
             #(break is an instruction that gets out of the while loop)
             break;

         # update iterate xn, increase n.
         n += 1;
         xn = g(xn); #apply g (fixed point step)
         rn = np.append(rn,xn); #add new guess to list of iterates

     # Set root estimate to xn.
     r=xn;

     if vrb:
         ########################################################################
         # Approximate error log-log plot
         logploterr(rn,r);
         plt.title('Fixed Point Iteration: Log error vs n');
         ########################################################################

     return r, rn;

# This auxiliary function plots approximate log error for a list of iterates given
# the list (in array rn) and the exact root r (or our best guess)
def logploterr(rn,r):
    n = rn.size-1;
    e = np.abs(r-rn[0:n]);
    #length of interval
    nn = np.arange(0,n);
    #log plot error vs iteration number
    plt.plot(nn,np.log2(e),'r--');
    plt.xlabel('n'); plt.ylabel('log2(error)');
    return;


r,rn = fixed_point_method(g,dg,1.5,0,3,tol,100, True)
print(rn)

#It takes 11 iterations to converge


#
#
#   	        (p_{n+2}p_n - p * (p_n + p_{n+1}) + p^2 )
# p = p_{n+1}- ----------------------------------------
#	                   (p_{n+1} - p)
#





#Aitken's Method




