from src.chatbot import Chatbot

if __name__=='__main__':
    chatbot = Chatbot()
    while True:
        query = input('대화를 입력하세요: ').strip()
        if query == 'q':
            break
        answer = chatbot.chat(query)
        print(f'chatbot: {answer}')