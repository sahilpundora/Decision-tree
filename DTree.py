#!/home/sahil/anaconda2/bin/python

import sys
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import math
import optparse

print 'Let\'s do decision tree!!'


#function to create new Table and add the rows as arrays in the table
table=[]
def processText(table,data):
	data = data.strip()
	fields = data.split(',')
	table.append(fields)

for line in sys.stdin.readlines():
        processText(table,line) 

#totalrows in the table
tr= len(table)
print 'Total Rows in the data are-', tr


#Now we have to do the main task for ENCODING
#Encoding is the process of converting Catergorical to Numerical 
#We have a total of 7 cols (excluding the 'ID', 'Age', ) to be converted to Numerical
#STEPS FOR ENCODING-
# 1. SPLIT THE COLUMNS IN INDIVIDUAL ARRAYS 
# 2. ENCODE EACH COLUMN USING LABLE_ENCODER
# 3. COMBINE THE ENCODED LABEL TO FORM THE ORIGINAL TABLE

# STEP 1:SPLIT THE COLUMNS IN INDIVIDUAL ARRAYS
n = 7
lists = [[] for _ in range(n)]
#for 1st column
for l in range(0,tr):
	lists[0].append(table[l][1])
#for rest of the columns
k=3
for i in range(1,7):
	for j in range(0,tr):		
		lists[i].append(table[j][k])
	k=k+1
	if k>10:
		break
lists=np.array(lists)

# STEP 2: ENCODE EACH COLUMN USING LABLE_ENCODER
label_encoder=LabelEncoder()
for i in range(0,7):
	lists[i]=label_encoder.fit_transform(lists[i].ravel()).reshape(*lists[i].shape)
#to check whether the encoder works-
#print lists[0]
#print lists[6]


# STEP 3: COMBINE THE ENCODED LABEL TO FORM THE ORIGINAL TABLE
for i in range(0,tr):
	table[i][1]=lists[0][i]
k=3
for i in range(1,7):
	for j in range(0,tr):
		table[j][k]=lists[i][j]
	k=k+1
	if k>10:
		break
	


#Now split the encoded data in Training and Test sets
#'X' set will contain the features
#'Y' set will contain the targets
trainingX = []
trainingY = []

testX = []
testY = []


#Split the data in 90:10 for Training:Test
#TO NOTE- Also remove the column ID from the data set as it is not a feature, and it useless
#So we start with index 1 and not 0 as it is the ID
for i in range(0, tr):
	newrow=[ table[i][1], table[i][2], table[i][3], table[i][4], table[i][5], table[i][6], table[i][7], table[i][8] ]
	if i%10==0:
		testX.append(newrow)
		testY.append(table[i][9])
	else:
		trainingX.append(newrow)
		trainingY.append(table[i][9])
print 'Total Rows are', tr
print 'Training Set has',len(trainingX),'rows, which is about',round( len(trainingX)*100/ float( (len(trainingX)+len(testX))) ),'% of the total rows'
print 'Test Set has',len(testX),'rows, which is about',float(len(testX)*100/(len(trainingX)+len(testX))),'% of the total rows'



#Replace 'Very Late Adopter' as 1 and the rest as 0
#First for Training
string='Very Late'
for i in range(0,len(trainingY)):
	if str(trainingY[i]) == string:
		trainingY[i]=1
	else:
		trainingY[i]=0
#Next for Test
for i in range(0,len(testY)):
	if str(testY[i]) == string:
		testY[i]=1
	else:
		testY[i]=0




#Convert training and test sets to numpy arrays 
#This is done for efficient processing by scikit
trainingX = np.array(trainingX)
trainingY = np.array(trainingY)
testX = np.array(testX)
testY = np.array(testY)

#Check the training set
#print trainingX
#print trainingY

#NOW we VARY the max_depth
max_leaf_nodes = [2, 4, 8, 16, 32, 64, 128, 256]
max_depth = [None, 2, 4, 8, 16]
for i in range(0, len(max_depth)):
	dicaccuracies = {}
	accuracies = []
	for j in range(0, len(max_leaf_nodes)):
		clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes[j], max_depth = max_depth[i])
		clf.fit(trainingX, trainingY)	
		correct = 0
		incorrect = 0
		#feed the test features through the model, to see how well the model
		#predicts the class from the samples in the test set it has never seen
		predictions = clf.predict(testX)
		for k in range(0, predictions.shape[0]):
			if(predictions[k] == testY[k]):
				correct += 1
			else:
				incorrect += 1
		#print "correct predictions: ", correct, " incorrect predictions: ", incorrect
		#compute accuracy
		accuracy = float(correct) / (correct + incorrect)
		accuracies.append(accuracy)
		dicaccuracies[max_leaf_nodes[i]] = accuracies
	plt.plot(max_leaf_nodes,accuracies)
	plt.xticks(max_leaf_nodes)
	plt.xlabel('Max_Leaf_Nodes')
	plt.ylabel('Accuracy')
	plt.title('Accuracy at Max Depth of '+ str(max_depth[i]))
	plt.show() 
	#print 'For max_depth=', max_depth[i]
	#print 'Accuracies-',dicaccuracies





#PART B BEGINS HERE
#Count the number of Very Late Adopters vs Not very late in the original dataset
countlate=0
for i in range(0,tr):
	if str(table[i][9]) == string:
		countlate+=1
	else:
		countlate+=0
print 'Total Values of Very Late Adopters-',countlate
print 'Total Values of Other Adopters-',tr-countlate

#Create table with 50:50 and shuffle the table in the end
i=0
j=0
table50=[]
while(i<tr and j<=countlate):
	newrow=[ table[i][1], table[i][2], table[i][3], table[i][4], table[i][5], table[i][6], table[i][7], table[i][8], table[i][9]]
	if table[i][9] != string:
		table50.append(newrow)
		j+=1	
	i+=1					
i=0
j=0	
while(i<tr and j<=countlate):
	newrow=[ table[i][1], table[i][2], table[i][3], table[i][4], table[i][5], table[i][6], table[i][7], table[i][8], table[i][9]]
	if table[i][9] == string:
		table50.append(newrow)
		j+=1	
	i+=1	
#SHUFFLE THE TABLE
table50=shuffle(table50, random_state=0)
#print table50


#CHECK WHETHER THE NEW TABLE CREATED IS UPTO STANDARD
print 'Total Rows in the new created table are-', len(table50)
countlate=0
for i in range(0,len(table50)):
	if str(table50[i][8]) == string:
		countlate+=1
	else:
		countlate+=0
print 'IN WHICH'
print 'Very Late Adopters are',countlate,'which is about',float( countlate*100/len(table50) ),'% of the total rows'
countlate=0
for i in range(0,len(table50)):
	if str(table50[i][8]) == string:
		countlate+=1
	else:
		countlate+=0
print 'Others\' are',countlate,'which is about',( countlate*100/len(table50) ),'% of the total rows'


#PART TWO OF THE QUESTION
#Create Test and Training out of the 50:50, with ration 90:10
trainingX = []
trainingY = []
testX = []
testY = []

for i in range(0, len(table50)):
	newrow=[ table50[i][0],table50[i][1], table50[i][2], table50[i][3], table50[i][4], table50[i][5], table50[i][6], table50[i][7]]
	if i%10==0:
		testX.append(newrow)
		testY.append(table50[i][8])
	else:
		trainingX.append(newrow)
		trainingY.append(table50[i][8])
print 'Training Set has',len(trainingX),'rows, which is about', math.ceil( len(trainingX)*100/ float( (len(trainingX)+len(testX))) ),'% of the total rows'
print 'Test Set has',len(testX),'rows, which is about', math.ceil(len(testX)*100/(len(trainingX)+len(testX))),'% of the total rows'

#Replace 'Very Late Adopter' as 1 and the rest as 0
#First for Training
string='Very Late'
for i in range(0,len(trainingY)):
	if str(trainingY[i]) == string:
		trainingY[i]=1
	else:
		trainingY[i]=0
#Next for Test
for i in range(0,len(testY)):
	if str(testY[i]) == string:
		testY[i]=1
	else:
		testY[i]=0


#Convert training and test sets to numpy arrays 
#This is done for efficient processing by scikit
trainingX = np.array(trainingX)
trainingY = np.array(trainingY)
testX = np.array(testX)
testY = np.array(testY)

#Check the training set
#print trainingX
#print trainingY

#NOW we VARY the max_depth
max_leaf_nodes = [2, 4, 8, 16, 32, 64, 128, 256]
max_depth = [None, 2, 4, 8, 16]
for i in range(0, len(max_depth)):
	dicaccuracies = {}
	accuracies = []
	for j in range(0, len(max_leaf_nodes)):
		clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes[j], max_depth = max_depth[i])
		clf.fit(trainingX, trainingY)	
		correct = 0
		incorrect = 0
		#feed the test features through the model, to see how well the model
		#predicts the class from the samples in the test set it has never seen
		predictions = clf.predict(testX)
		for k in range(0, predictions.shape[0]):
			if(predictions[k] == testY[k]):
				correct += 1
			else:
				incorrect += 1
		#print "correct predictions: ", correct, " incorrect predictions: ", incorrect
		#compute accuracy
		accuracy = float(correct) / (correct + incorrect)
		accuracies.append(accuracy)
		dicaccuracies[max_leaf_nodes[i]] = accuracies
	plt.plot(max_leaf_nodes,accuracies)
	plt.xticks(max_leaf_nodes)
	plt.xlabel('Max_Leaf_Nodes')
	plt.ylabel('Accuracy')
	plt.title('Accuracy at Max Depth of '+ str(max_depth[i]))
	plt.show() 
	#print 'For max_depth=', max_depth[i]
	#print 'Accuracies-',dicaccuracies
	


print 'Hope you Enjoyed the graphs!!!!!'
print 'Goodbye World!!!'



