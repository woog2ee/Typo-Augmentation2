import torch
import torch.nn as nn



class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_dim, num_classes, dropout=0.1):
        super().__init__()
        self.bert       = bert
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)


    def get_attn_mask(self, token_ids, valid_length):
        attn_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attn_mask[i][:v] = 1
        return attn_mask.float()
    
    
    def forward(self, token_ids, valid_length, segment_ids):
        attn_mask = self.get_attn_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids      = token_ids,
                              token_type_ids = segment_ids.long(),
                              attention_mask = attn_mask.float().to(token_ids.device),
                              return_dict    = False)
        
        output     = self.dropout(pooler)
        classified = self.classifier(output)
        return classified
    


class EarlyStopping():
    def __init__(self, patience, save_path):
        self.cnt        = 0
        self.best_acc   = 0
        self.best_epoch = 0
        self.stop       = False
        self.patience   = patience
        self.save_path  = save_path


    def __call__(self, augment_type, dataset_type, model, epoch, acc):
        if acc >= self.best_acc:
            self.cnt        = 0
            self.best_acc   = acc
            self.best_epoch = epoch

            torch.save(model, self.save_path+f'{augment_type}_{dataset_type}_{epoch+1}_ckp.pt')
            self.stop = False
        else:
            self.cnt += 1
            
            if self.cnt == self.patience:
                self.stop = True
        return self.stop