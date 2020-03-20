import torch


class LrScheduler(object):
    def __init__(self, cfg, optimizer, cur_ier=0):
        self.lr = cfg.SOLVER.LR
        self.schedule_type = cfg.SOLVER.LR_SCHEDULE
        self.update_iter = cfg.SOLVER.UPDATE_ITER
        self.update_rate = cfg.SOLVER.UPDATE_RATE
        self.cur_iter = cur_ier
        self.optimizer = optimizer

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def _update_lr(self):
        if self.schedule_type == 'exp':
            new_lr = self.lr * pow(self.update_rate, self.cur_iter / self.update_iter)
        elif self.schedule_type == 'step':
            new_lr = self.lr * pow(self.update_rate, self.cur_iter // self.update_iter)
        else:
            new_lr = self.lr

        self.lr = new_lr
        self.optimizer.lr = self.lr

    def step(self, cur_iter):
        self.cur_iter = cur_iter
        self._update_lr()


class Optimizer(object):
    def __init__(self, model, cfg):
        super(Optimizer, self).__init__()
        self.model = model
        self.solver = cfg.SOLVER
        self.optimizer = cfg.SOLVER.OPTIMIZER
        self.lr = cfg.SOLVER.LR

    def build_optimizer(self):
        params = self.model.parameters()
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, self.lr)

        else:
            # default use sgd
            optimizer = torch.optim.SGD(params, self.lr,
                                        momentum=self.solver.MOMENTUM)
        return optimizer
