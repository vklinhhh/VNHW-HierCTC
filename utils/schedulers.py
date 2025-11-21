# utils/schedulers.py
import math
import logging
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)

class CosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max(1, max_steps)
        self.eta_min = eta_min
        if self.warmup_steps > self.max_steps:
            logger.warning(f"warmup_steps ({self.warmup_steps}) > max_steps ({self.max_steps}). Setting warmup_steps = max_steps.")
            self.warmup_steps = self.max_steps

        super().__init__(optimizer, last_epoch)
        logger.info(f"CosineWarmupScheduler initialized: warmup={warmup_steps}, max_steps={max_steps}, eta_min={eta_min}")


    def get_lr(self):
        current_step = self.last_epoch + 1

        if current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = float(current_step) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            # Clamp progress to [0, 1]
            progress = min(progress, 1.0)
            # Cosine annealing formula: eta_min + 0.5 * (1 - eta_min) * (1 + cos(pi * progress))
            lr_scale = self.eta_min + 0.5 * (1.0 - self.eta_min) * (1.0 + math.cos(math.pi * progress))
        return [base_lr * lr_scale for base_lr in self.base_lrs]

    def state_dict(self):
        state = super().state_dict()
        state['warmup_steps'] = self.warmup_steps
        state['max_steps'] = self.max_steps
        state['eta_min'] = self.eta_min
        return state

    def load_state_dict(self, state_dict):
        self.warmup_steps = state_dict.pop('warmup_steps')
        self.max_steps = state_dict.pop('max_steps')
        self.eta_min = state_dict.pop('eta_min')
        super().load_state_dict(state_dict)


class CosineWarmupWithPlateauScheduler(object):
    def __init__(self, optimizer, warmup_steps, max_steps, plateau_patience=5,
                 plateau_factor=0.5, plateau_min_lr=1e-6, plateau_metric='val_cer',
                 plateau_mode='min', verbose=True):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.plateau_metric = plateau_metric
        self.cosine_scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps, max_steps
        )
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=plateau_mode,
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=plateau_min_lr,
            verbose=verbose
        )
        self.current_step = 0
        self.using_plateau = False
        
        logger.info(f"CosineWarmupWithPlateauScheduler initialized: "
                    f"warmup={warmup_steps}, max_steps={max_steps}, "
                    f"plateau_patience={plateau_patience}, plateau_factor={plateau_factor}, "
                    f"plateau_metric={plateau_metric}")
    
    def step(self, metric=None):
        if self.current_step < self.max_steps:
            self.cosine_scheduler.step()
            self.current_step += 1
            return

        if not self.using_plateau:
            logger.info(f"Switching to ReduceLROnPlateau scheduler after {self.current_step} steps.")
            self.using_plateau = True
        
        if metric is None:
            logger.warning(f"Metric value required for ReduceLROnPlateau but none provided. Skipping step.")
            return

        self.plateau_scheduler.step(metric)
        self.current_step += 1
    
    def state_dict(self):
        return {
            'cosine_state': self.cosine_scheduler.state_dict(),
            'plateau_state': self.plateau_scheduler.state_dict(),
            'current_step': self.current_step,
            'using_plateau': self.using_plateau
        }
    
    def load_state_dict(self, state_dict):
        self.cosine_scheduler.load_state_dict(state_dict['cosine_state'])
        self.plateau_scheduler.load_state_dict(state_dict['plateau_state'])
        self.current_step = state_dict['current_step']
        self.using_plateau = state_dict['using_plateau']