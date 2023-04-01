import argparse
from transformers import T5TokenizerFast, T5ForConditionalGeneration, T5ForConditionalGeneration


def model_init(args):
        model = T5ForConditionalGeneration.from_pretrained(args.model_path)        
        tokenizer = T5TokenizerFast.from_pretrained(args.model_path)

        model.config.max_length = args.max_target_length
        tokenizer.model_max_length = args.max_target_length
        return model, tokenizer

        
if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Hyperparameter for Chat')
        parser.add_argument('--model_path', default='./experiment1/', type=str,
                                                help='model path for chat') 
        parser.add_argument('--max_input_length', default=64, type=int,
                                                help='user max input length')
        parser.add_argument('--max_target_length', default=64, type=int,
                                                help='max target length')
        parser.add_argument('--prefix', default='qa question: ', type=str,
                                                help='inference input prefix')
        args = parser.parse_args()

        model, tokenizer = model_init(args)
        while True:
                user_inputs = input("user: ")
                if user_inputs == 'false':
                        break
                inputs = [args.prefix + user_inputs]
                inputs = tokenizer(inputs, max_length=args.max_input_length, truncation=True, return_tensors="pt")
                output = model.generate(**inputs, num_beams=2, do_sample=True, min_length=10, max_length=args.max_target_length, no_repeat_ngram_size=2) #repetition_penalty=2.5
                result = tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                result = str(result)
                result = result.replace('[', '').replace(']', '').replace("'", '').replace("'", '')
                print(f'chatbot: {result}')