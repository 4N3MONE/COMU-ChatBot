from src.chatbot import Chatbot
import pandas as pd

def iteratively_generate(chatbot, input_texts):
    output_texts = [chatbot.chat(input_text) for input_text in input_texts]
    return output_texts

def sequancially_chat(chatbot):
    while True:
        query = input('대화를 입력하세요: ').strip()
        if query == 'q':
            break
        answer = chatbot.chat('₩청자₩ ' +query)
        print(f'chatbot: {answer}')
    return

def generate_from_file(chatbot, input_file_path, output_file_path):
    with open(input_file_path, 'r') as f:
        article = f.readlines()
        
    outputs = iteratively_generate(chatbot)
    
    result_df = pd.DataFrame({
        'Query' : article,
        'Answer' : outputs
    })
    
    result_df.to_csv(output_file_path, encoding='utf-8-sig')

if __name__=='__main__':
    chatbot = Chatbot()
    sequancially_chat(chatbot)