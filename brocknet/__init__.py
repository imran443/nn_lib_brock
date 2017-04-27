from .network_master import NetworkMaster
from .network_data import NetworkData
from .network_tester import NetworkTester
from .network_trainer import NetworkTrainer
from .training_example import TrainingExample


def network():
    return NetworkMaster()