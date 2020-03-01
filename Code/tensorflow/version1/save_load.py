import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

x=np.linspace(-1,1,100)[:,np.newaxis]
noise=np.random.normal(0,0.1,size=x.shape)
y=np.power(x,2)+noise

def save():
	print('This is save')
	tf_x=tf.placeholder(tf.float32,x.shape)
	tf_y=tf.placeholder(tf.float32,y.shape)
	l=tf.layers.dense(tf_x,10,tf.nn.relu)
	o=tf.layers.dense(l,1)
	loss=tf.losses.mean_squared_error(tf_y,o)
	train_op=tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

	sess=tf.Session()
	sess.run(tf.global_variables_initializer())

	saver=tf.train.Saver()

	for step in range(100):
		sess.run(train_op,{tf_x:x,tf_y:y})

	saver.save(sess,'./params',write_meta_graph=False)

	pred,l=sess.run([o,loss],{tf_x:x,tf_y:y})
	plt.figure(1,figsize=(10,5))
	plt.subplot(121)
	plt.scatter(x,y)
	plt.plot(x,pred,'r-',lw=5)
	plt.text(-1,1.2,'Save Loss=%.4f'%l,fontdict={'size':15,'color':'red'})

def reload():
	print('This is reload')
	tf_x=tf.placeholder(tf.float32,x.shape)
	tf_y=tf.placeholder(tf.float32,y.shape)
	l_=tf.layers.dense(tf_x,10,tf.nn.relu)
	o_=tf.layers.dense(l_,1)
	loss_=tf.losses.mean_squared_error(tf_y,o_)

	sess=tf.Session()
	saver=tf.train.Saver()
	saver.restore(sess,'./params')
	pred,l=sess.run([o_,loss_],{tf_x:x,tf_y:y})
	plt.subplot(122)
	plt.scatter(x,y)
	plt.plot(x,pred,'r-',lw=5)
	plt.text(-1,1.2,'Reload Loss=%.4f'%l,fontdict={'size':15,'color':'red'})
	plt.show()

save()
tf.reset_default_graph()
reload()