import re

def CleanCorpus(texts):
    texts = [CleanText(post) for post in texts]
    return texts

def CleanText(text_post):

    text_post = text_post.strip() if str(text_post) != 'nan' else ''

    text_post = text_post.replace('Diario Clarín shared a link.', '')
    text_post = text_post.replace('LA NACION shared a link.', '')

    text_post = text_post.replace('...', '')
    text_post = re.sub(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', '', text_post)

    #texto_post = texto_post.replace('\r\n', '')
    text_post = text_post.replace('<3', '')
    text_post = text_post.replace('$', 'pesos ')

    text_post = text_post.replace('.', ' ')
    text_post = text_post.replace(':', ' ')
    text_post = text_post.replace(',', ' ')
    text_post = text_post.replace('¨', ' ')
    text_post = text_post.replace('-', ' ')
    text_post = text_post.replace(';', ' ')
    text_post = text_post.replace('[','')
    text_post = text_post.replace(']','')
    text_post = text_post.replace('(','')
    text_post = text_post.replace(')','')
    text_post = text_post.replace('¿', ' ¿ ')
    text_post = text_post.replace('¿', ' ¿ ')
    
    text_post = text_post.replace('¡', '¡ ')
    #texto_post = texto_post.replace('!',' !')

    text_post = text_post.replace('¿','')
    text_post = text_post.replace('?','')
    text_post = text_post.replace('¡','')
    text_post = text_post.replace('!','')

    text_post = text_post.replace('\'', ' ')
    text_post = text_post.replace('"', ' ')
    text_post = text_post.replace('_', ' ')
    
    text_post = text_post.replace('#','')
    text_post = text_post.replace('|', ' ')
    text_post = text_post.replace('<', ' ')
    text_post = text_post.replace('>', ' ')

    text_post = text_post.replace('“', ' ')
    text_post = text_post.replace('”', ' ')

    text_post = text_post.strip()
    text_post = text_post.lower()

    text_post = text_post.replace('conversacionesln', 'conversaciones ln')
    text_post = text_post.replace('elecciones2015', 'elecciones 2015')

    return text_post