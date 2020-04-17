def summation1(x1,y):
  sum=0
  for i in range(0,len(x1)):
    sum=sum+(w1*x1[i]+b-y[i])*x1[i]
  return sum
def summation2(x1,y):
  sum=0
  for i in range(0,len(x1)):
    sum=sum+(w1*x1[i]+b-y[i])
  return sum
def cost_function(x1,y,w1,b):
  J=0
  for i in range(0,len(x1)):
    J=J+(w1*x1[i]+b-y[i])*(w1*x1[i]+b-y[i])
  return J/(2*len(x1))
  
#1(a1)
import matplotlib.pyplot as plt 
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt1
from mpl_toolkits.mplot3d import Axes3D
x1=[]
y=[]
file=open('ex1data1.txt',"r")
for line in file:
  line = line.rstrip('\n')
  num1,num2=line.split(',')
  x1.append(float(num1))
  y.append(float(num2))
plt.scatter(x1,y, label='Plot', color='k', s=25, marker="o")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
#1(a2)y=w1x1+b
iterations = 1500
alpha = 0.01
w1=0
m=len(x1)#number of dataset points
b=0#initial parameters set to 0
itr=0
list_w1=[]
list_b=[]
list_cost=[]
while itr!=iterations:
  w1_new=w1-alpha/m*(summation1(x1,y))
  b_new=b-alpha/m*(summation2(x1,y))
  itr=itr+1
  w1=w1_new
  b=b_new
  list_w1.append(float(w1))
  list_b.append(float(b))
  list_cost.append(float(cost_function(x1,y,w1,b)))
print('Final parameters values:')
print(w1)
print(b)
print('Final Cost function value')
print(cost_function(x1,y,w1,b))
#1(a3)Surface Plot
Y=list_w1
X=list_b
Z=list_cost
c=np.linspace(-10,10,10)
m=np.linspace(-1,5,10)
cost_surf =np.zeros((len(c),len(m)))
for i in range(len(c)):
    for j in range(len(m)):
        cost_surf[i,j]=1/(2*len(x1))*np.sum((x1[i]*m[j]+c[i]-y[i])**2)    #X: feature vector, y=output
fig = plt1.figure(figsize=(20,20))
fig.suptitle('Cost Function', fontsize=20)
ax = plt1.axes(projection='3d')
surf=ax.plot_surface(c,m,cost_surf,cmap='Spectral')
fig.colorbar(surf, shrink=0.5, aspect=5)
# rotate the axes and update
ax.view_init(30, 10)
plt1.show()


def summation(x1,x2,y,val,w1,w2,b):
  sum=0
  for i in range(0,len(x1)):
    if val==1:
      x=x1[i]
    elif val==2:
      x=x2[i]
    else:
      x=1
    sum=sum+(w1*x1[i]+w2*x2[i]+b-y[i])*x
  return sum
def cost_function(x1,x2,y,w1,w2,b):
  J=0
  for i in range(0,len(x1)):
    J=J+(w1*x1[i]+w2*x2[i]+b-y[i])*(w1*x1[i]+w2*x2[i]+b-y[i])
  return J/(2*len(x1))
  
  
#1(b1)
import matplotlib.pyplot as plt
import statistics
from mpl_toolkits.mplot3d import Axes3D
x1=[]
x2=[]
y=[]
file=open('ex1data2.txt',"r")
for line in file:
  line = line.rstrip('\n')
  num1,num2,num3=line.split(',')
  x1.append(float(num1))
  x2.append(float(num2))
  y.append(float(num3))
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x1, x2, y, c='r', marker='o')
#ax.set_xlabel('X1 Label')
#ax.set_ylabel('X2 Label')
#ax.set_zlabel('Y Label')
#1(b2)feature normalization
std1=statistics.stdev(x1)
mean1=statistics.mean(x1)
std2=statistics.stdev(x2)
mean2=statistics.mean(x2)
for i in range(0,len(x1)):
  x1[i]=(x1[i]-mean1)/std1
  x2[i]=(x2[i]-mean2)/std2
#1(b3)y=w1x1+w2x2+b
iterations = 1500
alpha = 0.01
w1=0
w2=0
list_cost=[]
list_itr=[]
m=len(x1)#number of dataset points
b=0#initial parameters set to 0
itr=0
while itr!=iterations:
  w1_new=w1-alpha/m*(summation(x1,x2,y,1,w1,w2,b))
  w2_new=w2-alpha/m*(summation(x1,x2,y,2,w1,w2,b))
  b_new=b-alpha/m*(summation(x1,x2,y,0,w1,w2,b))
  itr=itr+1
  list_itr.append(itr)
  w1=w1_new
  w2=w2_new                   
  b=b_new
  list_cost.append(float(cost_function(x1,x2,y,w1,w2,b)))
print(w1)
print(w2)
print(b)
#print(cost_function(x1,x2,y,w1,w2,b))
from matplotlib import pyplot as plt1
plt1.plot(list_itr,list_cost)
#1(b5)Create matrix
import numpy
X=numpy.zeros(shape=(m,3))
W=numpy.zeros(shape=(3,1))
Y=numpy.zeros(shape=(m,1))
for i in range(m):
  X[i][0]=1
  X[i][1]=x1[i]
  X[i][2]=x2[i]
  Y[i][0]=y[i] 
W=numpy.linalg.inv((numpy.transpose(X).dot(X))).dot(numpy.transpose(X)).dot(Y)
print(W)
