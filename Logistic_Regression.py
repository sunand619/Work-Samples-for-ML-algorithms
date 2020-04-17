#a.1)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df
data = load_data("ex2data1.txt", None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
admitted = data.loc[y == 1]
not_admitted = data.loc[y == 0]
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
plt.legend()
plt.show()

#a2
import math
def sigmoid(z):
  return 1/(1+math.exp(-1*z))

def summation(x1,x2,y,val,w1,w2,b):
  sum=0
  for i in range(0,len(x1)):
    if val==1:
      x=x1[i]
    elif val==2:
      x=x2[i]
    else:
      x=1
    sum=sum+(sigmoid(w1*x1[i]+w2*x2[i]+b)-y[i])*x
  return sum
def cost_function(x1,x2,y,w1,w2,b):
  J=0
  for i in range(0,len(x1)):
    J=J+(sigmoid(w1*x1[i]+w2*x2[i]+b)-y[i])*(sigmoid(w1*x1[i]+w2*x2[i]+b)-y[i])
  return J/(2*len(x1))
x1=[]
x2=[]
y=[]
file=open('ex2data1.txt',"r")
for line in file:
  line=line.rstrip('\n')
  num1,num2,num3=line.split(',')
  x1.append(float(num1))
  x2.append(float(num2))
  y.append(float(num3))
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
count=0
for i in range(0,m):
  predicted=sigmoid(w1*x1[i]+w2*x2[i]-b)
  if predicted>0.5:
    predicted=1
  else:
    predicted=0
  #print(predicted)
  #print(y[i])
  if int(predicted)==int(y[i]):
    count=count+1
print(count)
print(m)
accuracy=count/m*100
print(accuracy)
  
#b2
import math
def sigmoid(z):
  return 1/(1+math.exp(-1*z))

def summation(x1,x2,y,val,w1,w2,b):
  sum=0
  for i in range(0,len(x1)):
    if val==1:
      x=x1[i]
    elif val==2:
      x=x2[i]
    else:
      x=1
    sum=sum+(sigmoid(w1*x1[i]+w2*x2[i]+b)-y[i])*x
  return sum
def cost_function(x1,x2,y,w1,w2,b):
  J=0
  for i in range(0,len(x1)):
    J=J+(sigmoid(w1*x1[i]+w2*x2[i]+b)-y[i])*(sigmoid(w1*x1[i]+w2*x2[i]+b)-y[i])
  return J/(2*len(x1))
x1=[]
x2=[]
y=[]
file=open('ex2data1.txt',"r")
for line in file:
  line=line.rstrip('\n')
  num1,num2,num3=line.split(',')
  x1.append(float(num1))
  x2.append(float(num2))
  y.append(float(num3))
iterations = 1500
alpha = 0.01
w1=0
w2=0
list_cost=[]
list_itr=[]
m=len(x1)#number of dataset points
b=0#initial parameters set to 0
itr=0
lamda=1
while itr!=iterations:
  w1_new=w1-alpha*(1/m*(summation(x1,x2,y,1,w1,w2,b))+lamda*w1/m)
  w2_new=w2-alpha*(1/m*(summation(x1,x2,y,2,w1,w2,b))+lamda*w2/m)
  b_new=b-alpha*(1/m*(summation(x1,x2,y,0,w1,w2,b))+lamda*b/m)
  itr=itr+1
  list_itr.append(itr)
  w1=w1_new
  w2=w2_new                   
  b=b_new
  list_cost.append(float(cost_function(x1,x2,y,w1,w2,b)))
print(w1)
print(w2)
print(b)
count=0
for i in range(0,m):
  predicted=sigmoid(w1*x1[i]+w2*x2[i]-b)
  if predicted>0.5:
    predicted=1
  else:
    predicted=0
  #print(predicted)
  #print(y[i])
  if int(predicted)==int(y[i]):
    count=count+1
print(count)
print(m)
accuracy=count/m*100
print(accuracy)

#b.1)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df
data = load_data("ex2data2.txt", None)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
admitted = data.loc[y == 1]
not_admitted = data.loc[y == 0]
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
plt.legend()
plt.show()

  

