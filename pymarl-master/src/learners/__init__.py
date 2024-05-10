from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
# altered
from .q_learner_nStep import QLearner_nStep

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner

# altered
REGISTRY["q_learner_nStep"] = QLearner_nStep
