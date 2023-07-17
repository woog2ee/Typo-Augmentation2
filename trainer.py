import torch
import numpy as np
from tqdm import tqdm



def calc_accuracy(x, y):
    max_vals, max_indices = torch.max(x, 1)
    train_acc = (max_indices == y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


def iteration(device, augment_type, dataset_type, model, criterion, optim_schedule,
              epochs, train_loader, valid_loader, early_stopping_callback):
    # Train & Valid at same time
    max_grad_norm = 5

    for epoch in range(epochs):
        train_acc, test_acc = 0.0, 0.0

        model.train()
        for id, data in enumerate(tqdm(train_loader)):
            optim_schedule.zero_grad()
            #optim.zero_grad()

            token_ids    = data['input_ids'].to(device)
            valid_length = data['valid_length']
            segment_ids  = data['segment_ids'].to(device)
            label        = data['label'].to(device)

            output = model(token_ids, valid_length, segment_ids)
            loss   = criterion(output, label)

            #loss.requires_grad_(True)
            loss.backward()
            optim_schedule.step_and_update_lr()
            #torch.nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)
            #optim.step()
            #optim_schedule.step()

            train_acc += calc_accuracy(output, label)
        print(f'Epoch {epoch+1} Train Accuracy: {train_acc / (id+1)}')

        model.eval()
        for id, data in enumerate(valid_loader):
            token_ids    = data['input_ids'].to(device)
            valid_length = data['valid_length']
            segment_ids  = data['segment_ids'].to(device)
            label        = data['label'].to(device)

            output = model(token_ids, valid_length, segment_ids)

            test_acc += calc_accuracy(output, label)
        print(f'Epoch {epoch+1} Valid Accuracy: {test_acc / (id+1)}')

        if early_stopping_callback(augment_type, dataset_type, model, epoch, test_acc / (id+1)):
            print(f'Early Stopping at Epoch {epoch+1}\n')
            break
    return early_stopping_callback.best_epoch


def predict(device, augment_type, dataset_type, save_path, best_epoch, test_loader):
    # Test
    test_pred, test_label, test_acc = [], [], []

    model = torch.load(save_path+f'{augment_type}_{dataset_type}_{best_epoch}_ckp.pt')
    model.eval()
    for id, data in enumerate(tqdm(test_loader)):
        token_ids    = data['input_ids'].to(device)
        valid_length = data['valid_length']
        segment_ids  = data['segment_ids'].to(device)
        label        = data['label'].to(device)

        with torch.no_grad():
            preds = model(token_ids, valid_length, segment_ids)
        test_acc.append(calc_accuracy(preds, label))

        preds = preds.argmax(dim=-1)
        test_pred.append(preds.cpu().numpy())
        test_label.append(label.cpu().numpy())
    
    return test_pred, test_label, test_acc



class ScheduledOptim():
    def __init__(self, optimizer, d_model, warmup_steps):
        self._optimizer   = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.init_lr      = np.power(d_model, -0.5)
        
        
    def step_and_update_lr(self):
        self._update_lr()
        self._optimizer.step()
        
        
    def zero_grad(self):
        self._optimizer.zero_grad()
        
        
    def _get_lr_scale(self):
        return np.min([
            np.power(self.current_step, -0.5),
            np.power(self.warmup_steps, -1.5) * self.current_step
        ])
    
    
    def _update_lr(self):
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()
        
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr