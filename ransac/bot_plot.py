from sklearn import datasets
import matplotlib.pyplot as plt
#
# Load the boston housing dataset
#
cali = datasets.load_boston()
X = cali.data
y = cali.target
print(y)
#
# Create the box plot
#
fig1, ax1 = plt.subplots()
ax1.set_title('Box plot for Housing Prices')
ax1.boxplot(y, vert=False)
plt.show()