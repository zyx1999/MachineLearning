import matplotlib.pyplot as plt
import numpy as np
import csv

sepal_len = []
sepal_wid = []
petal_len = []
petal_wid = []
species = []
with open('./dataset/iris.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        sepal_len.append(float(row['Sepal.Length']))
        sepal_wid.append(float(row['Sepal.Width']))
        petal_len.append(float(row['Petal.Length']))
        petal_wid.append(float(row['Petal.Width']))
        species.append(row['Species'])

print(petal_len[:50])
print(petal_len[100:150])

# keep = [i for (i, v) in enumerate(species) if v == "setosa"]
plt.scatter(petal_len[:50], petal_wid[:50], marker='*', color='green')
# plt.scatter(petal_wid[50:100], petal_len[50:100], marker='o', color='green')
plt.scatter(petal_len[100:150], petal_wid[100:150], marker='+', color='blue')

w = [0.5,1.7]
b = -3
x1_axis = np.linspace(-1, 10, 100)
x2_axis = -(w[0]*x1_axis+b)/w[1]
plt.plot(x1_axis, x2_axis)

plt.title("Perceptron")
plt.xlabel("Petal Length")
plt.ylabel('Petal Width')
plt.show()



