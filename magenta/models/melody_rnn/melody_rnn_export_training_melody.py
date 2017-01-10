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
tf.app.flags.DEFINE_string('record_ids', '',
                           'Record id numbers, separated by comma')
tf.app.flags.DEFINE_float(
    'qpm', 120,
    'The quarters per minute to play generated output at. If a primer MIDI is '
    'given, the qpm from that will override this flag. If qpm is None, qpm '
    'will default to 120.')

def main(unused_argv):
  if not FLAGS.sequence_example_file:
    tf.logging.fatal('--sequence_example_file required')
    return

  if not FLAGS.record_ids:
    tf.logging.fatal('--record_ids required')
    return

  needed_ids = [ int(rid.strip()) for rid in FLAGS.record_ids.split(',') ]
  sequence_example_file_paths = tf.gfile.Glob(
    os.path.expanduser(FLAGS.sequence_example_file))
  config = melody_rnn_config_flags.config_from_flags()

  records = find_records(needed_ids, config, sequence_example_file_paths)

  for id, labels in records.items():
    melody = magenta.music.Melody(events=labels, steps_per_quarter=4)
    config.encoder_decoder.decode_labels(melody, labels) 

    seq = melody.to_sequence(qpm=FLAGS.qpm)
    magenta.music.sequence_proto_to_midi_file(seq, 'record-%d.mid' % id)

def find_records(needed_ids, config, sequence_example_file_paths):
  result = {}

  with tf.Graph().as_default() as g:
    with tf.Session() as sess:
      _, labels, _, id = magenta.common.get_padded_batch(
        sequence_example_file_paths, 1, config.encoder_decoder.input_size)

      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      first_id = -1
      while True:
        labels_, id_ = sess.run([labels, id])

        if first_id == -1:
          first_id = id_[0]
        elif first_id == id_[0]:
          tf.logging.warn('Hit end, could not locate records %s' % needed_ids.join(', '))
          return result

        if id_[0] in needed_ids:
          tf.logging.info('Found %d' % id_[0])
          result[id_[0]] = labels_[0]
          needed_ids.remove(id_[0])

          if len(needed_ids) == 0:
            return result

        missed_ids = [ x for x in needed_ids if id_[0] - x > 10 ]
        if len(missed_ids) > 0:
          print missed_ids
          needed_ids = [ x for x in needed_ids if id_[0] - x <= 10 ]
          if len(needed_ids) == 0:
            return result

def console_entry_point():
  tf.app.run(main)

if __name__ == '__main__':
  console_entry_point()