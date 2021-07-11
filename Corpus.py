import pandas as pd
from TextCleaner import CleanText
from LemmaTokenizer import convert_text_to_tokens

class Corpus(object):
    def __init__(self, nombreArchivoEntrada, sheet):
        self.df = pd.read_excel (nombreArchivoEntrada, sheet_name=sheet)
        
        #Limpiar Textos
        self.df['post_message_limpio'] = self.df['post_message'].map(lambda a: CleanText(a))
        self.df['titulo_facebook_limpio'] = self.df['titulo_facebook'].map(lambda a: CleanText(a))

    def get_corpus(self):
        return self.df