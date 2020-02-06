import numpy as np
import pandas as pd
import sklearn
import sklearn.naive_bayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
  
    
def write_to_file(lis):
    write=open("submit.dat","w")
    write.writelines("%s\n" % int(i) for i in lis)
    write.close()
# ?write_to_file(y_pred.tolist())
    print("writing all done")   

def decistiontree(x,y,z):
    clf = DecisionTreeClassifier()
    clf = clf.fit(x,y)
    y_pred = clf.predict(z)
    write_to_file(y_pred)
    print(" decstion tree done")


# gets the integer col
def separate(lis):
    arr=[]
    continious_col=[0,7,8,19,21,30,41,46,49,53]
    for i in range(len(continious_col)):
        arr.append(lis[:,continious_col[i]])
    return arr
def neural_net(x,y,z):
    scaler=StandardScaler()
    # x is training data
    scaler.fit(x)
    train=scaler.transform(x)
    # z is the test data 
    test=scaler.transform(z)
    clf=MLPClassifier(hidden_layer_sizes=(74,74,74),max_iter=600)
    # y is the class lablel column
    clf=clf .fit(x, y)
    save= clf.predict(z)
    print("all done") 
dt=np.loadtxt(fname='train.csv', delimiter=',')
test=np.loadtxt(fname='test.csv',delimiter=',')
print("data extracted done")
print(len(dt[0]))
y=dt[:,-1]  # class lable
x=dt[:,0:54]  # traning data without the class label
z=test  # test data
# bays(x,y,z)
arr=[]
continious_col=[0,7,8,19,21,30,41,46,49,53]
for i in range(len(continious_col)):
	arr.append(x[:,continious_col[i]])
con_x=np.transpose(arr)  # this is continious features for the traning set 
arr=[]
continious_col=[0,7,8,19,21,30,41,46,49,53]
for i in range(len(continious_col)):
	arr.append(z[:,continious_col[i]])
con_y=np.transpose(arr)  # this is the contious fetures for the test set
# the whole data set without the class label
# y=dt[:,-1]
# x=dt[:,0:54]
# con_x=np.transpose(seprate(x))  # this is continious features for the traning set 
# cont_y=con_x=np.transpose(seprate(z))  # this is the contious fetures for the test set
x=np.delete(x,0,1)
x=np.delete(x,6,1)
x=np.delete(x,6,1)
x=np.delete(x,16,1)
x=np.delete(x,17,1)
x=np.delete(x,25,1)
x=np.delete(x,35,1)
x=np.delete(x,39,1)
x=np.delete(x,41,1)
x=np.delete(x,44,1)
z=np.delete(z,0,1)
z=np.delete(z,6,1)
z=np.delete(z,6,1)
z=np.delete(z,16,1)
z=np.delete(z,17,1)
z=np.delete(z,25,1)
z=np.delete(z,35,1)
z=np.delete(z,39,1)
z=np.delete(z,41,1)
z=np.delete(z,44,1)
bernNB=BernoulliNB()
gaussianNB=GaussianNB()
# bernNB=MultinomialNB()
# bernNB.fit(x_train,y_train)
b=bernNB.fit(x,y)
c=gaussianNB.fit(con_x,y) #train the gaussan on the contnious data
prob_binary=bernNB.predict_proba(z)
# this will get the prob for the continious data on the traning set
prob_cont=gaussianNB.predict_proba(con_y)  # prdict the contious data


prob=[]
for i in range(len(prob_cont)):
		cur=[]
		for j in range(7):
			val=prob_binary[i][j]*prob_cont[i][j]
			cur.append(val)
		prob.append(cur)
val=[]
for i in range(len(prob)):
	maxe=max(prob[i])
	val.append(prob[i].index(maxe)+1)
write_to_file(val)
print("bays all done")   
y=dt[:,-1]
x=dt[:,0:54]
z=test

# uncomment below line  to run decistion tree
# decistiontree(x,y,z)
# uncomment below line to run neural networl
# neural_net(x,y,z);



    
