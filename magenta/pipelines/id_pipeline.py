from magenta.pipelines import pipeline
from magenta.music import events_lib

class IDPipeline(pipeline.Pipeline):
  """A Pipeline that assigns unique IDs to EventSequences and logs the ID mappings in the pipeline stats."""

  def __init__(self, name=None):
    super(IDPipeline, self).__init__(
        input_type=events_lib.EventSequence,
        output_type=events_lib.EventSequence,
        name=name)
    self.counter = 0
    self._set_stats({})

  def transform(self, event_sequence):
  	mappings = self.get_stats()
  	mappings[self.counter] = event_sequence.filename
  	self._set_stats(mappings)

    event_sequence.id = event_sequence
	self.counter++

	return [event_sequence]