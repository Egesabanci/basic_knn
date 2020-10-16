import sys
sys.path.append("..")

from utils import load_data

# load sample_data.csv from ../data/
path = "../data"
filename = "sample_data.csv"
data = load_data(path, filename)

# get variables from data
xs = data["X"]
ys = data["Y"]
labels = data["Label"]
sample_input = (3.7, 8.8)

# create KNN model
from KNN import KNNClassifier
model = KNNClassifier(xs = xs, ys = ys, labels = labels)

# prediction with sample input
prediction = model.predict(sample_input)
print(f"Prediction: {prediction}")
# Output: 0

# Assume above the discriminator = 0
# and under the discriminator = 1
# According to the sample data,
# the prediction must be a 0.

# plotting data
import matplotlib.pyplot as plt
discriminator = [x for x in range(2, 10)]
plt.title(f"Prediction of the model: {prediction}")
plt.plot(discriminator, discriminator, color = "green")
plt.scatter(x = model.train_x, y = model.train_y, color = "blue")
plt.scatter(x = sample_input[0], y = sample_input[1], color = "red")
plt.show()
