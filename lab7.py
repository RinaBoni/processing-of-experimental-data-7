import numpy as np
import matplotlib.pyplot as plt 

X = np.array([0.35, 0.70, 1.05, 1.40, 1.75, 2.10, 2.45, 2.80, 3.15])
Y = np.array([4.15, 4.30, 4.45, 4.60, 4.75, 4.90, 5.05])
data = np.array([[3.87, 4.17, 8.65, 3.19, 4.65, 4.98, 7.12, 6.65, 3.76],
                [4.75, 3.76, 2.19, 5.34, 4.65, 2.14, 4.54, 3.33, 6.54],
                [5.43, 4.24, 5.43, 4.33, 5.33, 3.54, 5.34, 4.32, 3.43],
                [6.33, 3.33, 2.43, 4.54, 5.34, 5.34, 4.54, 5.43, 4.43],
                [5.43, 4.43, 4.54, 3.43, 5.32, 5.34, 6.54, 4.54, 5.54],
                [4.54, 3.54, 4.76, 3.76, 5.65, 4.54, 5.76, 7.54, 3.54],
                [5.76, 7.54, 5.76, 3.23, 4.34, 3.54, 3.23, 5.76, 3.43]])

def lagranz(x,y,t): 
         z=0 
         for j in range(len(y)): 
             p1=1; p2=1 
             for i in range(len(x)): 
                 if i==j: 
                     p1=p1*1; p2=p2*1    
                 else:  
                     p1=p1*(t-x[i]) 
                     p2=p2*(x[j]-x[i]) 
             z=z+y[j]*p1/p2 
         return z



x=np.array([2,5,-6,7,4,3,8,9,1,-2], dtype=float) 
y=np.array([-1,77,-297,249,33,9,389,573,-3,-21], dtype=float) 

xnew=np.linspace(np.min(x),np.max(x),100) 
ynew=[lagranz(x,y,i) for i in xnew] 
plt.plot(x,y,'o',xnew,ynew) 
plt.grid(True) 
plt.show()