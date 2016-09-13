# Import tensorflow library
# Reference it as tf for ease of calling
import tensorflow as tf

# Let's create two matrices, a 3x1 and another 1x3 matrix for multiplication
mat_a = tf.constant([[1., 3., 5.]], name='mat_a')
mat_b = tf.constant([[7.], [11.], [13.]], name='mat_b')

# Let's matrix multiply the two matrices
prod_op = tf.matmul(mat_a, mat_b)

# ---------------------------------------------------------------
# Create a session object to run our matrix multiplication
sess = tf.Session()

# Get the result by calling run on the session
# Returns an numpy ndarray object
mat_mul = sess.run(prod_op)

# Let's view the result 
print(mat_mul)
# [[ 105.]]

# Remember to close the session when done, releases the resources
sess.close()

# ---------------------------------------------------------------
# Easier way to handle session objects is 
# using the familiar 'with' block as follows
with tf.Session() as sess:
	mat_mul = sess.run(prod_op)
	print(mat_mul)
# [[ 105.]]

# ---------------------------------------------------------------
# If we want to run on multiple devices we do as follows
with tf.Session() as sess:
	with tf.device("/cpu:0"):
		mat_mul = sess.run(prod_op)
		print(mat_mul)
