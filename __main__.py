import torch
import torch.nn as nn
from dataset import BERTDataset
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from model import BERTClassifier, EarlyStopping
from torch.optim import Adam
from trainer import ScheduledOptim, iteration, predict
import argparse
import pandas as pd



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    parser.add_argument('--train_dataset_path', type=str)
    parser.add_argument('--valid_dataset_path', type=str)
    parser.add_argument('--test_dataset_path', type=str)
    parser.add_argument('--dataset_type', type=str)
    parser.add_argument('--augment_type', type=str)
    parser.add_argument('--save_path', type=str)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)

    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)

    parser.add_argument('--epochs', type=int, default=20)

    print('Loading All Parse Arguments\n')
    args = parser.parse_args()
    

    print(f'Using Device: {args.device}\n')


    print('Loading Train & Test Dataset\n')
    #model_path = 'monologg/kobert'
    model_path = 'skt/kobert-base-v1'
    train_dataset = BERTDataset(dataset_path = args.train_dataset_path,
                                dataset_type = args.dataset_type,
                                split_type   = 'train',
                                model_path   = model_path,
                                max_length   = args.max_length)
    valid_dataset = BERTDataset(dataset_path = args.valid_dataset_path,
                                dataset_type = args.dataset_type,
                                split_type   = 'valid',
                                model_path   = model_path,
                                max_length   = args.max_length)
    test_dataset  = BERTDataset(dataset_path = args.train_dataset_path,
                                dataset_type = args.dataset_type,
                                split_type   = 'test',
                                model_path   = model_path,
                                max_length   = args.max_length)
    

    print('Loading Train & Test DataLoader\n')
    train_loader = DataLoader(train_dataset,
                              batch_size  = args.batch_size,
                              num_workers = args.num_workers)
    valid_loader = DataLoader(valid_dataset,
                              batch_size  = args.batch_size,
                              num_workers = args.num_workers)
    test_loader  = DataLoader(test_dataset,
                              batch_size  = args.batch_size,
                              num_workers = args.num_workers)
    

    print('Loading Pretrained BERT Model\n')
    num_classes = 2 if args.dataset_type == 'nsmc' else (3 if args.dataset_type == 'hate' else None)
    bert  = BertModel.from_pretrained(model_path)
    model = BERTClassifier(bert,
                           hidden_dim  = 768,
                           num_classes = num_classes).to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    
    
    print('Setting Optimzier\n')
    optim = Adam(params       = model.parameters(),
                 lr           = args.lr,
                 betas        = (0.9, 0.999),
                 weight_decay = 0.01)
    optim_schedule = ScheduledOptim(optimizer    = optim,
                                    d_model      = 768,
                                    warmup_steps = 5000)

    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optim = AdamW(optimizer_grouped_parameters,
    #               lr = args.lr)

    # t_total     = len(train_loader) * args.epochs
    # warmup_step = int(t_total * args.warmup_ratio)
    # schedule    = get_cosine_schedule_with_warmup(optim,
    #                                               num_warmup_steps   = warmup_step,
    #                                               num_training_steps = t_total)
    
    early_stopping_callback = EarlyStopping(patience  = 5,
                                            save_path = args.save_path)
    

    print('Training Start\n')
    best_epoch = iteration(device                  = args.device,
                           augment_type            = args.augment_type,
                           dataset_type            = args.dataset_type,
                           model                   = model,
                           criterion               = loss_fn,
                           optim_schedule          = optim_schedule,
                           #optim                   = optim,
                           #optim_schedule          = schedule,
                           epochs                  = args.epochs,
                           train_loader            = train_loader,
                           valid_loader            = valid_loader,
                           early_stopping_callback = early_stopping_callback)
    

    print('Testing Start\n')
    test_pred, test_label, test_acc = predict(device       = args.device,
                                              augment_type = args.augment_type,
                                              dataset_type = args.dataset_type,
                                              save_path    = args.save_path,
                                              best_epoch   = best_epoch,
                                              test_loader  = test_loader)

    test_result  = pd.DataFrame({'pred': test_pred, 'label': test_label, 'acc': test_acc})
    test_avg_acc = sum(list(test_result['acc'])) / len(test_result)

    print(f'Testing Average Accuracy: {test_avg_acc}\n')
    test_result.to_csv(args.save_path+f'{args.dataset_type}_{best_epoch}.csv', index=False)