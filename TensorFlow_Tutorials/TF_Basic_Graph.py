# Import tensorflow library
# Reference it as tf for ease of calling
import tensorflow as tf

# Let's build our graph nodes
# Let's use inputs values from our example
a = tf.constant(20, name="in_a")
b = tf.constant(10, name="in_b")
c = tf.mul(a,b, name="mul_ab")
d = tf.add(a,b, name="add_ab")
e = tf.add(c,d, name="add_cd")

# Open a TensorFlow Session
sess = tf.Session()

# Execute output node
# Run using active Session object sess
sess.run(e)

# Open a TensorFlow SummaryWriter
# Write the graph to disk
writer = tf.train.SummaryWriter('./graph_dir', sess.graph)

# Close SummaryWriter
writer.close()
# Close Session object
sess.close()

# Visualize graph using TensorBoard after running this file 
# Run the following command, change tensorboard path to your local tensorflow directory:
# $ python /usr/local/lib/python2.7/dist-packages/tensorflow/tensorboard/tensorboard.py --logdir='/mnt/d/TF_ML/TFTutorial/TensorFlow_Tutorials/graph_dir'