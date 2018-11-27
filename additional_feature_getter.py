
import nltk,re
import enchant
d=enchant.Dict("en_US")
from keras.preprocessing.text import Tokenizer

def feature_getter(text):
	try:
		text=text.decode('utf-8')
	except:
		pass
	text1=re.sub(r'[^\x00-\x7F]+',' ', text)
	text=text1
	features=[]
	tokens=[]
	sentences=nltk.sent_tokenize(text)
	[tokens.extend(nltk.word_tokenize(sentence)) for sentence in sentences]
	features.append(len([t for t in tokens if not d.check(t)])) #spell errors
	features.append(len(tokens)) #num_tokens
	features.append(len(sentences)) #num_sentences
	features.append(len(tokens)/len(sentences))# average sentence length
	features.append(sum([len(wor) for wor in tokens])/len(tokens))# average word length
	features.append(len([wor for wor in tokens if len(wor)>7])) #number of long words
	features.append(sum([len(wor) for wor in tokens])) #num of characters
	return features