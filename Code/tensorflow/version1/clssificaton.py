import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

n_data=np.ones((100,2))
x0=np.random.normal(2*n_data,1)
y0=np.zeros(100)
x1=np.random.normal(-2*n_data,1)
y1=np.ones(100)
x=np.vstack((x0,x1))
y=np.hstack((y0,y1))

plt.scatter(x[:,0],x[:,1],c=y,s=100,lw=0,cmap='RdYlGn')
plt.show()
