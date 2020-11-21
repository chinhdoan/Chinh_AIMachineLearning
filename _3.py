import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()

class LogisticRegression():
    """Class for training and using a model for logistic regression"""

    def set_values(self, initial_params, alpha=0.01, max_iter=5000, class_of_interest=0):
        """Set the values for initial params, step size, maximum iteration, and class of interest"""
        self.params = initial_params
        self.alpha = alpha
        self.max_iter = max_iter
        self.class_of_interest = class_of_interest

    @staticmethod
    def _sigmoid(x):
        """Sigmoide function"""

        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, x_bar, params):
        """predict the probability of a class"""

        return self._sigmoid(np.dot(params, x_bar))

    def _compute_cost(self, input_var, output_var, params):
        """Compute the log likelihood cost"""

        cost = 0
        for x, y in zip(input_var, output_var):
            x_bar = np.array(np.insert(x, 0, 1))
            y_hat = self.predict(x_bar, params)

            y_binary = 1.0 if y == self.class_of_interest else 0.0
            cost += y_binary * np.log(y_hat) + (1.0 - y_binary) * np.log(1 - y_hat)

        return cost

    def train(self, input_var, label, print_iter=5000):
        """Train the model using batch gradient ascent"""

        iteration = 1
        while iteration < self.max_iter:
            if iteration % print_iter == 0:
                print(f'iteration: {iteration}')
                print(f'cost: {self._compute_cost(input_var, label, self.params)}')
                print('--------------------------------------------')

            for i, xy in enumerate(zip(input_var, label)):
                x_bar = np.array(np.insert(xy[0], 0, 1))
                y_hat = self.predict(x_bar, self.params)

                y_binary = 1.0 if xy[1] == self.class_of_interest else 0.0
                gradient = (y_binary - y_hat) * x_bar
                self.params += self.alpha * gradient

            iteration += 1

        return self.params

    def test(self, input_test, label_test):
        """Test the accuracy of the model using test data"""
        self.total_classifications = 0
        self.correct_classifications = 0

        for x, y in zip(input_test, label_test):
            self.total_classifications += 1
            x_bar = np.array(np.insert(x, 0, 1))
            y_hat = self.predict(x_bar, self.params)
            y_binary = 1.0 if y == self.class_of_interest else 0.0

            if y_hat >= 0.5 and y_binary == 1:
                # correct classification of class_of_interest
                self.correct_classifications += 1

            if y_hat < 0.5 and y_binary != 1:
                # correct classification of an other class
                self.correct_classifications += 1

        self.accuracy = self.correct_classifications / self.total_classifications

        return self.accuracy

digits_train, digits_test, digits_label_train, digits_label_test = train_test_split(digits.data, digits.target, test_size=0.20)

#train0
alpha = 1e-2
params_0 = np.zeros(len(digits.data[0]) + 1)

max_iter = 10000
digits_regression_model_0 = LogisticRegression()
digits_regression_model_0.set_values(params_0, alpha, max_iter, 0)

params = digits_regression_model_0.train(digits_train / 16.0, digits_label_train, 1000)


digits_accuracy = digits_regression_model_0.test(digits_test / 16.0, digits_label_test)
print(f'Accuracy of prediciting a ZERO digit in test set: {digits_accuracy}')
joblib.dump(digits_regression_model_0, "digits_train.pkl", compress=3)
#train1
alpha = 1e-2
params_0 = np.zeros(len(digits.data[0]) + 1)

max_iter = 10000
digits_regression_model_1 = LogisticRegression()
digits_regression_model_1.set_values(params_0, alpha, max_iter, 1)

params = digits_regression_model_1.train(digits_train / 16.0, digits_label_train, 1000)

digits_accuracy = digits_regression_model_1.test(digits_test / 16.0, digits_label_test)
print(f'Accuracy of prediciting a ONE digit in test set: {digits_accuracy}')

joblib.dump(digits_regression_model_1, "digits_train.pkl", compress=3)