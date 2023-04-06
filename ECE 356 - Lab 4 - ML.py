from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from pandas import read_csv
import matplotlib.pyplot as plt

# Import the data from the .csv files
taskAData = read_csv('./ECE_356_Lab_4_Task_A.csv')
taskBData = read_csv('./ECE_356_Lab_4_Task_B.csv')

# Set features and classifier
features = ['numAwards', 'hitsBatter', 'runsBatter', 'homerunsBatter', 'stolenBases', 'putouts', 'assists', 'shutouts', 'saves', 'hitsPitcher', 'earnedRuns', 'homerunsPitcher', 'strikeouts']
classifier = ['class']

# Separate the data for task A
taskADataX = taskAData[features]
taskADataY = taskAData[classifier]

# Separate the data for task B
taskBDataX = taskBData[features]
taskBDataY = taskBData[classifier]

# Randomly select a training set and testing set for task A
taskADataXTrain, taskADataXTest, taskADataYTrain, taskADataYTest = train_test_split(taskADataX, taskADataY, test_size=0.2, random_state=1)

# Randomly select a training set and testing set for task B
taskBDataXTrain, taskBDataXTest, taskBDataYTrain, taskBDataYTest = train_test_split(taskBDataX, taskBDataY, test_size=0.2, random_state=1)

# Create decision tree
clf = DecisionTreeClassifier(max_depth = 8, min_samples_split=200)

# Train on data for Task A
clf = clf.fit(taskADataXTrain, taskADataYTrain)
plot_tree(clf, fontsize=2, rounded=True, feature_names=features)
plt.savefig("taskADecisionTree.svg")

# Predict on results for Task A
taskADataYPredicted = clf.predict(taskADataXTest)
taskAConfusionMatrix = confusion_matrix(taskADataYTest, taskADataYPredicted)
tn, fp, fn, tp = taskAConfusionMatrix.ravel()
f1ScoreTaskA = f1_score(taskADataYTest, taskADataYPredicted, average='macro')
print((tn+tp)/(tn+tp+fn+fp))
print(taskAConfusionMatrix)
print(f1ScoreTaskA)

# Train on data for Task B
clf = clf.fit(taskBDataXTrain, taskBDataYTrain)
plot_tree(clf, fontsize=2, rounded=True, feature_names=features)
plt.savefig("taskBDecisionTree.svg")

# Predict on results for Task B
taskBDataYPredicted = clf.predict(taskBDataXTest)
taskBConfusionMatrix = confusion_matrix(taskBDataYTest, taskBDataYPredicted)
tn, fp, fn, tp = taskBConfusionMatrix.ravel()
f1ScoreTaskB = f1_score(taskBDataYTest, taskBDataYPredicted, average='macro')
print((tn+tp)/(tn+tp+fn+fp))
print(taskBConfusionMatrix)
print(f1ScoreTaskB)