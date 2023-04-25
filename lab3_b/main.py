import numpy as np

# Funkcja aktywacji - jednostkowa skokowa funkcja progowa
def step(x):
    return 1 if x >= 0 else 0


# Uczenie perceptronu
def perceptron_learn(X, Y, lr=0.1, epochs=10):
    # Inicjalizacja wag oraz biasu na wartości losowe
    w = np.random.rand(X.shape[1])
    b = np.random.rand()

    # Iteracje uczące
    for epoch in range(epochs):
        for i in range(X.shape[0]):
            x = X[i]
            y = Y[i]
            # Obliczenie wyjścia
            output = step(np.dot(w, x) + b)
            # Aktualizacja wag i biasu
            w += lr * (y - output) * x
            b += lr * (y - output)
    return w, b


# Funkcja logiczna AND
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_and = np.array([0, 0, 0, 1])

w_and, b_and = perceptron_learn(X_and, Y_and)

print("Wagi dla funkcji logicznej AND:", w_and)
print("Bias dla funkcji logicznej AND:", b_and)

# Funkcja logiczna NOT
X_not = np.array([[0], [1]])
Y_not = np.array([1, 0])

w_not, b_not = perceptron_learn(X_not, Y_not)

print("Wagi dla funkcji logicznej NOT:", w_not)
print("Bias dla funkcji logicznej NOT:", b_not)


# Pierwsza warstwa
def layer1(x1, x2):
    # Perceptron 1
    w1, w2 = 1, -1
    b = 0
    output1 = step(w1 * x1 + w2 * x2 + b)
    # Perceptron 2
    w1, w2 = -1, 1
    b = 0
    output2 = step(w1 * x1 + w2 * x2 + b)
    return output1, output2

# Druga warstwa
def layer2(output1, output2):
    # Perceptron
    w1, w2 = 1, 1
    b = -1
    output = step(w1 * output1 + w2 * output2 + b)
    return output

# Sieć perceptronów reprezentująca funkcję XOR
def xor_perceptron(x1, x2):
    output1, output2 = layer1(x1, x2)
    output = layer2(output1, output2)
    return output

# Przykłady użycia
print(xor_perceptron(0, 0)) # Output: 0
print(xor_perceptron(0, 1)) # Output: 1
print(xor_perceptron(1, 0)) # Output: 1
print(xor_perceptron(1, 1)) # Output: 0



# Funkcja aktywacji - sigmoida
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pochodna funkcji aktywacji
def sigmoid_derivative(x):
    return x * (1 - x)

# Implementacja sieci neuronowej
class XORNeuralNetwork:
    def __init__(self):
        # Inicjalizacja wag i biasów
        self.weights1 = np.array([[1, -1], [-1, 1]])
        self.bias1 = np.array([0, 0])
        self.weights2 = np.array([1, 1])
        self.bias2 = -1

    def feedforward(self, x):
        # Warstwa wejściowa
        self.input = x

        # Warstwa ukryta
        self.layer1_output = sigmoid(np.dot(self.input, self.weights1) + self.bias1)

        # Warstwa wyjściowa
        self.output = sigmoid(np.dot(self.layer1_output, self.weights2) + self.bias2)
        return self.output

    def backpropagation(self, y):
        # Obliczenie błędów na wyjściu
        error = y - self.output
        d_output = error * sigmoid_derivative(self.output)

        # Obliczenie błędów w warstwie ukrytej
        error_layer1 = d_output.dot(self.weights2.T)
        d_layer1_output = error_layer1 * sigmoid_derivative(self.layer1_output)

        # Aktualizacja wag i biasów
        self.weights2 += self.layer1_output.T.dot(d_output)
        self.bias2 += np.sum(d_output)
        self.weights1 += self.input.T.dot(d_layer1_output)
        self.bias1 += np.sum(d_layer1_output, axis=0)

    def train(self, X, y, epochs):
        for i in range(epochs):
            for j in range(len(X)):
                output = self.feedforward(X[j])
                self.backpropagation(y[j])

# Przykłady użycia
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = XORNeuralNetwork()