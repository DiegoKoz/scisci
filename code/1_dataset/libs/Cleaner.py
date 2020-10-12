from tqdm.notebook import tqdm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import unicodedata2
import string

class Cleaner:
    
    def __init__(self):
        remove_list = ['©',
                       'all rights reserved.',
                       'elsevier ltd.',
                       'Cambridge University Press',
                       '© the author(s)'
                       'all rights reserved',
                       'sage publication',
                       'springer-verlag berlin heidelberg',
                       'springer science+business media dordrecht.',
                       'sage publications.',
                       'university of north carolina press.',
                       'elsevier',
                       'b.v.',
                       'american association for the advancement of science.',
                       'the author(s)',
                       'beech tree publishing',
                       'finnish society for science and technology studies.',
                       'sage publications.',
                       'akadémiai kiadó, budapest, hungary.',
                       'akadémiai kiadó.']
        self.remove_list = remove_list
        self.stemming_dict_ = {}

    
    def strip_accents(self, text):
        text = unicodedata2.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return str(text)
    
    def stemmer(self, filtered_words):
        stemmer = SnowballStemmer('spanish')
        stemmed_words = []
        for i in filtered_words:
            stemmed = stemmer.stem(i)
            stemmed_words.append(stemmed)
            if stemmed in self.stemming_dict_:
                dict_stem = self.stemming_dict_[stemmed]
                if i in dict_stem:
                    dict_stem[i] =  dict_stem[i] + 1
                else:
                    dict_stem[i] = 1
            else:
                 self.stemming_dict_[stemmed] = {i:1}
        doc = (' ').join(stemmed_words)
        return  doc
    
    def de_stemmer(self, data):
        #search the most frequent
        replacement = {}
        for key in tqdm(self.stemming_dict_.keys(), desc="select most representative word of stem"):
            stem_dict = self.stemming_dict_[key]
            replacement[key] = max(stem_dict, key=stem_dict.get)
        self.replacement = replacement
        #replace in docs
        de_stemmed = []
        for doc in tqdm(data, desc = "de-stemming"):
            word_list = doc.split(' ')
            new_list = []
            for word in word_list:
                if word in self.stemming_dict_:
                    new_list.append(replacement.get(word))
                else:
                    new_list.append(word)
            de_stemmed.append(' '.join(new_list))
        return de_stemmed
    
    
    def preprocess(self, doc,stem, stopwords_lang = "english"):
        doc = doc.lower()
        for signature in self.remove_list:
            doc=doc.replace(signature, '')
        remove_words  = stopwords.words(stopwords_lang)
        remove_punctuation_map = dict((ord(char), ' ') for char in string.punctuation)
        doc = self.strip_accents(doc)
        doc = doc.translate(remove_punctuation_map)
        querywords = doc.split()
        #replace numbers with NUM
        querywords = ["NUM" if c.isdigit() else c for c in querywords]
        filtered_words = [palabra for palabra in querywords if palabra not in remove_words]       
        if stem:
            doc = self.stemmer(filtered_words)
        else:
            doc = (' ').join(filtered_words)
        return doc
    
    def data_clean(self,data, stemming):
        data_clean = [self.preprocess(doc, stem = stemming) for doc in tqdm(data, desc = "preprocess")]
        if stemming:
            data_clean = self.de_stemmer(data_clean)
        return data_clean    
