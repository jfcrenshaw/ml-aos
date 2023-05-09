"""Print the number of parameters in the model."""
# flake8: noqa
from ml_aos.utils import count_parameters
from ml_aos.wavenet import WaveNet

model = WaveNet()
print(f"Total parameters: {count_parameters(model, trainable=False):,}")
print(f"Trainable parameters: {count_parameters(model, trainable=True):,}")
