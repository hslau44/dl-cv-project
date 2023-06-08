from .train import objective
from .wce import get_example_configs

config = get_example_configs()
objective(config)
print(config)

print("*****Complete*****")