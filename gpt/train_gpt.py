from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from chat_dataset import ChatDataset
from chat_dataloader import ChatDataLoader
import torch
from torch.utils.data import DataLoader
import pandas as pd
import argparse

def gpt_train(args):
    # model  = AutoModelForCausalLM.from_pretrained(args.model_checkpoint)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True, stride=128)
    
    model.config.max_length = args.max_target_length
    tokenizer.model_max_length = args.max_target_length
    return model, tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter for Training gpt-based model for conditional generation')
    parser.add_argument('--model_checkpoint', default='skt/ko-gpt-trinity-1.2B-v0.5', type=str,
                        help='huggingface model name to train')
    parser.add_argument('--max_input_length', default=100, type=int,
                        help='max input length for dialog')
    parser.add_argument('--max_target_length', default=100, type=int,
                    help='max target length for dialog')
    parser.add_argument('--train_batch_size', default=8, type=int,
                                            help='train batch size')
    parser.add_argument('--num_train_epochs', default=1, type=int,
                                            help='train epoch size')
    parser.add_argument('--lr', default=5e-5, type=int,
                                            help='learning rate for training')
    parser.add_argument('--wd', default=0.01, type=int,
                                            help='weight decay for training'),
    parser.add_argument('--model_name', default='skt/ko-gpt-trinity-1.2B-v0.5', type=str,
                                            help='model name for saving')
    parser.add_argument('--base_path', default='../data/data_final.tsv', type=str,
                                                help='dataset path')
    parser.add_argument('--model_path', default='./output', type=str,
                                                help='model path for saving')

    args = parser.parse_args()
    
    #Load dataset
    dataset = pd.read_csv(args.base_path,sep='\t')
    tokenized_dataset = ChatDataset(dataset)
    # batched_dataset = ChatDataLoader(tokenized_dataset, args.train_batch_size)
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True, stride=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False,return_tensors='pt')
    # data_collator = ChatDataLoader.collate_fn
    
    training_args = TrainingArguments(
        output_dir = args.model_path,
        evaluation_strategy='epoch',
        save_total_limit=1,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        save_steps=1000,
        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_steps=3000,
        logging_steps=1000,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False     
    )
    
    trainer = Trainer(
        model = model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    # Training
    print('Start Training...')  
    
    trainer.train()
    
    # Saving model
    print('Saving Model...')
    trainer.save_model()