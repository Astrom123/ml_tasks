import telebot
from humor_classifier import HumorClassifier


clf = HumorClassifier()

with open('token.txt') as f:
    token = f.read()
    bot = telebot.TeleBot(token)


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет, напиши мне шутку и я скажу смешная ли она.')


@bot.message_handler(content_types=['text'])
def send_text(message):
    is_funny = clf.is_humorous(message.text)
    if is_funny:
        bot.send_message(message.chat.id, 'Смешно :D')
    else:
        bot.send_message(message.chat.id, 'Не смешно :/')


bot.polling()
