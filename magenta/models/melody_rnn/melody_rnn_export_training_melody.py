import tensorflow as tf
import os
from magenta.models.melody_rnn import melody_rnn_config_flags
import magenta
from magenta.protobuf import music_pb2


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('sequence_example_file', '',
                           'Path to TFRecord file containing '
                           'tf.SequenceExample records for training or '
                           'evaluation. A filepattern may also be provided, '
                           'which will be expanded to all matching files.')
tf.app.flags.DEFINE_integer('record_id', '',
                           'Record id number')
tf.app.flags.DEFINE_float(
    'qpm', 120,
    'The quarters per minute to play generated output at. If a primer MIDI is '
    'given, the qpm from that will override this flag. If qpm is None, qpm '
    'will default to 120.')

def main(unused_argv):
  if not FLAGS.sequence_example_file:
    tf.logging.fatal('--sequence_example_file required')
    return

  if not FLAGS.record_id:
    tf.logging.fatal('--record_id required')
    return

  needed_id = int(FLAGS.record_id)
  sequence_example_file_paths = tf.gfile.Glob(
    os.path.expanduser(FLAGS.sequence_example_file))
  config = melody_rnn_config_flags.config_from_flags()

  labels = find_record(needed_id, config, sequence_example_file_paths)

  seq = music_pb2.NoteSequence()
  seq.tempos.add().qpm = FLAGS.qpm
  config.encoder_decoder.decode_labels(seq, labels) 

  magenta.music.sequence_proto_to_midi_file(seq, 'record-%d.mid' % needed_id)

def find_record(needed_id, config, sequence_example_file_paths):
  with tf.Graph().as_default() as g:
    with tf.Session() as sess:
      _, labels, _, id = magenta.common.get_padded_batch(
        sequence_example_file_paths, 1, config.encoder_decoder.input_size)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      last_id = -1
      while True:
        labels_, id_ = sess.run([labels, id])
        #tf.logging.warn('Saw record %d, looking for %d' % (id_[0], needed_id))

        if needed_id == id_[0]:
          print(labels_)
          return labels_[0]
        # elif needed_id < id_[0]:
        #   tf.logging.warn('Could not find record %d, hit %d', needed_id, id_[0])
        #   return False

        # if id_[0] < last_id:
        #   #tf.logging.warn('Could not find record %d, hit %d', needed_id, id_[0])
        #   return False
        # else:
        #   last_id = id_[0]

def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()