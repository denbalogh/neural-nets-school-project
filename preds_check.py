f1 = open("test_predictions.csv", "r")
f2 = open("data/fashion_mnist_test_labels.csv", "r")

count = 0

for x, y in zip(f1, f2):
    if x != y:
        count += 1
        
print("Number of incorrect predictions: ", count)
print("Accuracy: ", 100 - (count/10000)*100)

f1.close()
f2.close()
