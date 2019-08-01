from collections import namedtuple


__all__ = ['FieldOutputRequest', 'HistoryOutputRequest', 'FieldOutput', 'HistoryOutput',
           'FieldOutputFrame', 'HistoryOutputFrame', 'possible_field_outpts', 'possible_history_outpts']

required_descriptions_field = ['domain', 'step', 'time_points']
required_descriptions_history = ['step', 'time_points']
possible_field_outpts = ['displacement', 'load', 'heat', 'temperature', 'plastic_deformation', 'wear']

# in docs nt = named tuple

# nt used to contain requests for field outputs
FieldOutputRequest = namedtuple('FieldOutputRequest', required_descriptions_field + possible_field_outpts,
                                defaults=(None,)*len(possible_field_outpts))

# nt used to contain actual field outputs, frames should relate to each time point (a tuple of field output frames)
FieldOutput = namedtuple('FieldOutput', required_descriptions_field + ['frames'],
                         defaults=(None,))

# nt used to contain the actual numerical data from each frame of a field output
FieldOutputFrame = namedtuple('FieldOutputFrame', possible_field_outpts,
                              defaults=(None,)*len(possible_field_outpts))

# nt used to contain a request for a history output
possible_history_outpts = ['friction', 'elastic_energy', 'heat_energy', 'plastic_energy', 'load']
HistoryOutputRequest = namedtuple('HistorOutputRequest', required_descriptions_history + possible_history_outpts,
                                  defaults=(None,)*len(possible_history_outpts))

# nt used to contain a history output with frames being a tuple of history output frames one for each time point
HistoryOutput = namedtuple('HistoryOutput', required_descriptions_history + ['frames'],
                           defaults=(None,))

# nt used to contain the numerical data for each frame
HistoryOutputFrame = namedtuple('HistoryOutputFrame', possible_history_outpts,
                                defaults=(None,)*len(possible_history_outpts))
