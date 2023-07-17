import torch
from torch.utils.data import Dataset
from kobert_tokenizer import KoBERTTokenizer
import pandas as pd
from soynlp.normalizer import repeat_normalize



class BERTDataset(Dataset):
    def __init__(self, dataset_path, dataset_type, split_type, model_path, max_length):
        raw_dataset     = pd.read_csv(dataset_path, encoding='utf-8-sig')
        tokenizer       = KoBERTTokenizer.from_pretrained(model_path)
        self.max_length = max_length

        # Rearrange for augmented train dataset
        if any(augment_type in dataset_path for augment_type in ['eda', 'aeda', 'aug']) and split_type == 'train':
            df1 = raw_dataset.iloc[:int(len(raw_dataset)/2), :]
            df2 = raw_dataset.iloc[int(len(raw_dataset)/2):, :]

            arranged_text, arranged_label = [], []
            for i in range(len(df1)):
                arranged_text.append(df1.iloc[i]['text'])
                arranged_text.append(df2.iloc[i]['text'])
                arranged_label.append(df1.iloc[i]['label'])
                arranged_label.append(df2.iloc[i]['label'])
            raw_dataset = pd.DataFrame({'text': arranged_text, 'label': arranged_label})
            print('Train Dataset Rearranged\n')

        # Preprocess input texts
        raw_dataset['clean_text'] = raw_dataset['text'].map(lambda x:
                                                            self.clean_text(x))
        raw_dataset['tokenized']  = raw_dataset['clean_text'].map(lambda x:
                                                                  tokenizer(x, padding='max_length', truncation=True, max_length=max_length))
        self.tokenized = list(raw_dataset['tokenized'])

        # Attach labels
        classes = {'nsmc': {0: 0, 1: 1},
                   'hate': {'none': 0, 'offensive': 1, 'hate': 2}}
        classes = classes[dataset_type]
        raw_dataset['encoded_label'] = raw_dataset['label'].map(lambda x:
                                                                classes[x])
        self.encoded_label = list(raw_dataset['encoded_label'])


    def clean_text(self, text):
        return repeat_normalize(text, num_repeats=2).strip()


    def __len__(self):
        return len(self.encoded_label)
    

    def __getitem__(self, idx):
        input_ids    = self.tokenized[idx]['input_ids']
        valid_length = self.max_length
        segment_ids  = self.tokenized[idx]['token_type_ids']
        label        = self.encoded_label[idx]

        output = {'input_ids'   : input_ids,
                  'valid_length': valid_length,
                  'segment_ids' : segment_ids,
                  'label'       : label}
        return {k: torch.tensor(v) for k, v in output.items()}