import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('run_dir', '/tmp/melody_rnn/logdir/run1',
                           'Path to the directory where checkpoints and '
                           'summary events will be saved during training and '
                           'evaluation. Separate subdirectories for training '
                           'events and eval events will be created within '
                           '`run_dir`. Multiple runs can be stored within the '
                           'parent directory of `run_dir`. Point TensorBoard '
                           'to the parent directory of `run_dir` to see all '
                           'your runs.')

if not FLAGS.run_dir:
	tf.logging.fatal('--run_dir required')
	return

run_dir = os.path.expanduser(FLAGS.run_dir)
train_dir = os.path.join(run_dir, 'train')

with tf.Graph().as_default() as g:
  with tf.Session() as sess:
  	# Trick to load variable without loading full graph: use validate_shape=False
  	# However, this also has to be set in the graph
    embedding = tf.Variable(0., tf.float32, name='embedding', validate_shape=False)

    # Load data
    saver = tf.train.Saver()
    checkpoint_path = tf.train.latest_checkpoint(train_dir)
    saver.restore(sess, checkpoint_path)

    # Write numpy file
    embedding_ = sess.run(embedding)
    np.save('...', embedding)