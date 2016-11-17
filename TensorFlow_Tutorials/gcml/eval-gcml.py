import tensorflow as tf


def run_tf_module():
    x = tf.placeholder("float") # Create a placeholder 'x'
    w = tf.Variable(5.0, name="weights")
	b = tf.Variable(1.0, name="bias")
    
	# y = x*w + b
	y = tf.add(tf.mul(w, x), b)

    with tf.Session() as sess:
        # Add the variable initializer Op.
        tf.initialize_all_variables().run()

        print("Evaluating y = x*w + b: %f" % sess.run(y, feed_dict={x: 1.0}))


def main(_):
    run_tf_module()

if __name__ == '__main__':
    tf.app.run()