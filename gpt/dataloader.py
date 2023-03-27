from datasets import load_dataset, Dataset
import pandas as pd
 
def load_data(base_path):
    # tsv 파일 로드
    dataset = pd.read_csv(base_path, sep='\t')
    dataset = Dataset.from_pandas(dataset)

    # train set과 validation set으로 분리
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True)

    return dataset
 
