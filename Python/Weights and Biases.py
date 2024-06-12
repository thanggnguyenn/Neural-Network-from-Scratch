# Every neuron has its own bias.
# Every connection between neuron has its own weight.

# one neuron with four inputs
inputs = [1.7, 2.6, 0.5, 2.5]

weights = [4.6, 2, 3.7, -0.7]

bias = 3.0

output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + inputs[3] * weights[3] + bias

print(output)

# Three neurons with four inputs
weights1 = [[4.6, 2, 3.7, -0.7],
            [0.5, -0.95, 4.7, -0.7],
            [2.6, 0.8, 5.7, -1.7]]

biases1 = [3.0, 2.0, 7.0]

output1 = []

for i in range(len(weights1)):
    sum = 0
    for j in range(len(inputs)):
        sum += inputs[j] * weights1[i][j]
    output1.append(sum + biases1[i])

print(output1)
