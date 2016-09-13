# Import tensorflow library
# Reference it as tf for ease of calling
# List of math ops can be found here: https://www.tensorflow.org/versions/r0.10/api_docs/python/math_ops.html
import tensorflow as tf

# Example using InteractiveSession 
# Use c.eval() to get evaluate
a = tf.constant(23, name='const_a')
b = tf.constant(11, name='const_b')

# Let's add two prime numbers
c = tf.add(a, b)
# 34

# Let's subtract two prime numbers
c = tf.sub(a, b)
# 12

# Let's multiply two prime numbers
c = tf.mul(a, b)
# 253

# Let's divide two prime numbers
c = tf.div(23, 13)
# 2

a = tf.constant(23., name='const_a')
b = tf.constant(11., name='const_b')

# Now, let's divide two prime numbers to see the difference
c = tf.div(23, 13)
# 2.090909

# Let's get the modulus of two prime numbers
c = tf.mod(a, b)
# 1.0

a = tf.constant(2., name='const_a')
b = tf.constant(10., name='const_b')

# Let's calculate 2^10
c = tf.pow(a, b)
# 1024.0

# Let's try out some control ops
# List of control ops can be found here: https://www.tensorflow.org/versions/r0.10/api_docs/python/control_flow_ops.html
# Check for a < b
c = tf.less(a, b)
# True 

# Check for a <= b
c = tf.less_equal(a, b)
# True

# Check for a > b
c = tf.greater(a, b)
# False

# Check for a >= b
c = tf.greater_equal(a, b)
# False

# Let's define some tensors and select elements from a or b 
# based on condition that c has elements from a only if they are less than 2
a = tf.constant([[1., 2.], [3., 4.]], name ='tensor_a')
b = tf.constant([[5., 6.], [1., -1.]], name ='tensor_b')
condition = tf.less_equal(a, 2)
c = tf.select(condition, a, b, name = 'select_a_b_condition')
# array([[1., 2.],
#        [1., -1.]], dtype=int32)

# d = tf.is_nan(c, name='test_nan')
# array([[False, False],
#        [False, False]], dtype=bool)

# Some conditional check statements
c = tf.logical_and(True, False)
# False

tf.logical_or(True, False)
# True

tf.logical_xor(True, False)
# True