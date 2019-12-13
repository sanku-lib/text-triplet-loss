import os
from matplotlib.image import imread
import numpy as np
import pandas as pd
from tensorflow.contrib import learn
import fasttext
import numpy as np
from sklearn.utils import shuffle


class PreProcessing:
    def __init__(self,data_src, max_document_length):
        self.embeddings_model = fasttext.train_unsupervised("./data_repository/text_corpus.txt", model='skipgram', lr=0.05, dim=128,
                                            ws=5, epoch=1)
        self.embeddings_model.save_model("./model_triplet/ft_skipgram_ws5_dim28.bin")
        self.current_index = 0
        similar_pairs = pd.read_csv(data_src)
        similar_pairs = similar_pairs[similar_pairs['is_duplicate'] == 1]
        similar_pairs = similar_pairs[['question1','question2']]
        similar_pairs = shuffle(similar_pairs)
        print('Training Data loaded successfully.')
        similar_pairs.dropna(how='any',inplace=True)
        similar_pairs.columns = ['question1','question2']
        similar_pairs['question1'] = similar_pairs['question1'].str.lower()
        similar_pairs['question2'] = similar_pairs['question2'].str.lower()
        unique_list_of_items = [item.replace('"','').replace("'",'') for item in list(set(similar_pairs['question2']))]
        triplet_list = []
        for item in similar_pairs.values.tolist():
            rand_index = np.random.choice(len(unique_list_of_items))
            rand_item = unique_list_of_items[rand_index]
            item.append(rand_item)
            if item[1] != item[2]:
                triplet_list.append(item)
        print('Triplets created successfully. # of triplets: ',len(triplet_list))
        triplets = triplet_list
        input_X = [item[0] for item in triplets]
        input_Y = [item[1] for item in triplets]
        input_Z = [item[2] for item in triplets]
        wc_list_x = list(len(x.split(' ')) for x in input_X)
        wc_list_y = list(len(x.split(' ')) for x in input_Y)
        wc_list_z = list(len(x.split(' ')) for x in input_Z)
        wc_list = []
        wc_list.extend(wc_list_x)
        wc_list.extend(wc_list_y)
        wc_list.extend(wc_list_z)
        number_of_elements = len(input_X)
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        full_corpus = []
        full_corpus.extend(input_X)
        full_corpus.extend(input_Y)
        full_corpus.extend(input_Z)
        full_data = np.asarray(list(self.vocab_processor.fit_transform(full_corpus)))
        self.embeddings_lookup = []
        for word in list(self.vocab_processor.vocabulary_._mapping):
            try:
                self.embeddings_lookup.append(self.embeddings_model[str(word)])
            except:
                pass
        self.embeddings_lookup = np.asarray(self.embeddings_lookup)
        # np.save("./model_triplet/embeddings_lookup.npz",self.embeddings_lookup)
        self.vocab_processor.save('./model_triplet/vocab')
        self.write_metadata(os.path.join('model_triplet','metadata.tsv'),list(self.vocab_processor.vocabulary_._mapping))
        print('Vocab processor executed and saved successfully.')
        self.X = full_data[0:number_of_elements]
        self.Y = full_data[number_of_elements:2*number_of_elements]
        self.Z = full_data[2*number_of_elements:3*number_of_elements]

    def write_metadata(self, filename, labels):
        with open(filename, 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(labels):
                f.write("{}\t{}\n".format(index, label))
        print('Metadata file saved in {}'.format(filename))

    def get_triplets_batch(self, n):
        last_index = self.current_index
        self.current_index += n
        return self.X[last_index:self.current_index,:], self.Y[last_index:self.current_index,:],self.Z[last_index:self.current_index,:]

