# Khan, Sahiba
# 1002_083_293
# 2022_09_07
# Assignment_00_01
from matplotlib import pyplot as plt  # Required import
import numpy as np  # import numpy
x = np.linspace(-20,20,500) # Create 100 linearly spaced numbers from -2 to 2
y = x**2 # y is a function of x (y=x**2)
z = 100*np.sin(x) # z is also a function of x (z=x**3)
plt.plot(x,y) # Plot y vs x
plt.plot(x,z) # Plot z vs x on the same plot
plt.show()  # Show the plot