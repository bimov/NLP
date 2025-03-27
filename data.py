import re

def cleaned_word(word):
    return re.sub(r'[^a-zа-я]', '', word.lower())


def get_words():
    with open('war_and_peace.txt', 'r', encoding='utf-8') as file:
        content = file.read()
        content = content.split()

    for i in range(len(content)):
        content[i] = cleaned_word(content[i])


    content = [content[i] for i in range(len(content)) if content[i] != ""]

    return content[:50000]


