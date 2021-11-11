from numpy import genfromtxt
from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy

y_pred = test_mlp('./test_X.csv')

test_labels = genfromtxt('./test_y.csv', delimiter=',')
test_accuracy = accuracy(test_labels, y_pred)*100
print(test_accuracy)
