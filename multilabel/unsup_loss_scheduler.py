import math

class CosineIncreaseScheduler:
    def __init__(self, total_steps, step=0):
        self.total_steps = total_steps
        self.step = step

    def make_step(self):
        self.step += 1

    def get_multiplier(self):
        return 1 - math.cos(math.pi * 0.5 * self.step / self.total_steps)