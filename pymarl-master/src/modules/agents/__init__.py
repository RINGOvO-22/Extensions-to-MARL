REGISTRY = {}

from .rnn_agent import RNNAgent
from .dueling_rnn_agent import RNNAgent as dueling_rnn_agent

REGISTRY["rnn"] = RNNAgent
REGISTRY["dueling_rnn"] = dueling_rnn_agent