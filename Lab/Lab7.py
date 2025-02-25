####PRELAB####
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
