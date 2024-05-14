import matplotlib.pyplot as plt

plt.figure(figsize = (7, 5), layout='tight')

plt.subplot(2, 1, 1)
plt.title("Training Times for Residual Network")
plt.scatter(list(range(9)), [23 + 53/60, 
                             22 + 14/60, 
                             23 + 8/60, 
                             20 + 1/60, 
                             19 + 12/60, 
                             18 + 38/60, 
                             17 + 28/60, 
                             18 + 5/60, 
                             16 + 46/60])
plt.ylabel("Time (min)")
plt.xlabel("Upsample Layer Placement")

plt.subplot(2, 1, 2)
plt.title("Training Times for Non-Residual Network")
plt.scatter(list(range(9)), [17 + 55/60, 
                             16 + 8/60, 
                             13 + 38/60, 
                             15 + 7/60, 
                             11 + 47/60, 
                             12 + 17/60, 
                             9 + 36/60, 
                             11 + 46/60, 
                             12 + 22/60])
plt.ylabel("Time (min)")
plt.xlabel("Upsample Layer Placement")

plt.show()