import matplotlib.pyplot as plt

plt.figure(figsize = (7, 5), layout='tight')

plt.title("Training Times")
plt.scatter(list(range(9)), [23 + 53/60, 
                             22 + 14/60, 
                             23 + 8/60, 
                             20 + 1/60, 
                             19 + 12/60, 
                             18 + 38/60, 
                             17 + 28/60, 
                             18 + 5/60, 
                             16 + 46/60], s=100, label = "Residual Network")
plt.scatter(list(range(9)), [31 + 59/60, 
                             28 + 23/60, 
                             25 + 51/60, 
                             25 + 34/60, 
                             24 + 10/60, 
                             23 + 33/60, 
                             23 + 26/60, 
                             23 + 28/60, 
                             23 + 60/60], s=100, marker = "^", label = "Standard Network")
plt.ylabel("Time (min)")
plt.xlabel("Upsample Layer Placement")
plt.legend()
plt.show()