# utils/optimizers.py
import torch.optim as optim
import logging
logger = logging.getLogger(__name__)


def create_optimizer(model, learning_rate, weight_decay=0.01, discriminative_lr=False, encoder_lr_factor=0.1):
    if discriminative_lr:
        compatibility_params = []
        encoder_params = []
        decoder_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'character_diacritic_compatibility.compatibility_matrix' in name:
                compatibility_params.append(param)
            elif name.startswith('vision_encoder.'):
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        optimizer = optim.AdamW([
            {
                'params': compatibility_params, 
                'lr': learning_rate * 0.01,
                'weight_decay': 0.1 
            },
            {'params': encoder_params, 'lr': learning_rate * encoder_lr_factor},
            {'params': decoder_params, 'lr': learning_rate}
        ], weight_decay=weight_decay)
    else:
        compatibility_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'character_diacritic_compatibility.compatibility_matrix' in name:
                compatibility_params.append(param)
            else:
                other_params.append(param)
        
        if compatibility_params:
            optimizer = optim.AdamW([
                {
                    'params': compatibility_params,
                    'lr': learning_rate * 0.01,
                    'weight_decay': 0.1
                },
                {'params': other_params, 'lr': learning_rate}
            ], weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(other_params, lr=learning_rate, weight_decay=weight_decay)

    return optimizer