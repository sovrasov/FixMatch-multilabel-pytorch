import torch


class LossBalancer(torch.nn.Module):
    def __init__(self, num_losses) -> None:
        super().__init__()
        self.loss_weights = torch.nn.Parameter(torch.zeros(num_losses, dtype=torch.float32))
        self.optimizer = torch.optim.SGD([self.loss_weights], lr=0.5)

    def forward(self, x):
        pass

    def balance_losses(self, losses):
        total_loss = 0.
        for i, l in enumerate(losses):
            total_loss += torch.exp(-self.loss_weights[i])*l + \
                0.5*self.loss_weights[i]

        return total_loss

    def init_iteration(self):
        self.optimizer.zero_grad()

    def end_iteration(self):
        self.optimizer.step()