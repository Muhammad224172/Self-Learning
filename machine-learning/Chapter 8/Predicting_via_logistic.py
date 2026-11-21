import pandas as pd 
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import re
from nltk.corpus import stopwords 
from tqdm import tqdm
import numpy as np 
stop = stopwords.words('english')




print("="*50)
print("="*50)

path = 'reviews_dataset.csv'

def tokenizer(text):
      text = re.sub('<[^>]*>', '', text)
      text = text.lower()
      tokenized = [w for w in text.split() if w not in stop]

      return tokenized


def stream_docs(path):

    with open(path, 'r' , encoding = 'utf-8') as csv: 
        next(csv) # skips the header 
        for line in csv: 
           
            line = line.split(',')
            text = line[0]
            label = (line[1])
            

        yield text , label 


def mini_batch(doc_stream , size): 

    docs , y = [] , [] 
    
    try: 
        for _ in range(size): 
        
            text , label = next(doc_stream)
            if label.isdigit(): 
                label = str(label)
                docs.append(text)
                y.append(int(label))

    except StopIteration:
        return None , None 
    
    return docs,y

vect = HashingVectorizer(decode_error='ignore' , 
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log' , random_state=1)
doc_stream = stream_docs(path='reviews_dataset.csv')

print("==" * 50)
print("Training...")
total_files = 400000

pbar = tqdm(total=total_files, desc="Loading")

classes = np.array([1, 5])
for _ in range(100):
      X_train, y_train = mini_batch(doc_stream, size=4000)
      if not X_train:
          break
      X_train = vect.transform(X_train)
      clf.partial_fit(X_train, y_train, classes=classes)
      pbar.update()

