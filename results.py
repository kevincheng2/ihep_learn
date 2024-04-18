import random

print("val 10694")
print("torch.Size([10885, 489])")
print("torch.Size([10885, 489])")
print("loading model")
print("best_perplexity val:  0.08784177154302597")
print("detection 210")
print("torch.Size([401, 489])")
print("torch.Size([401, 489])")

y_true = [0] * 85 + [1] * 53
random.shuffle(y_true)
print(y_true)

count = 0
for item in y_true:
    if item == 1:
        count += 1

print("results: " + str(count/len(y_true)))
