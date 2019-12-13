from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
import sys
import os


class GetItemEmbeddings(object):
    path_to_vocab = "vocab"
    model_path = './model_triplet/'

    def __init__(self):
        print('Loading Deep Learning Model.')
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(self.model_path, self.path_to_vocab))
        checkpoint_file = tf.train.latest_checkpoint(self.model_path)
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)

                # Get the placeholders from the graph by name
                self.anchor_input = graph.get_operation_by_name("anchor_input").outputs[0]

                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("output/scores").outputs[0]
        print('Model loaded successfully.')

    def get_item_embeddings(self, product_description, is_normalize = False):
        query = list(product_description.lower())
        input_queries = np.asarray(list(self.vocab_processor.fit_transform(query)))
        batch_predictions = self.sess.run(self.predictions, {self.anchor_input: [input_queries[0]]})
        if is_normalize:
            normalized_search_vec = batch_predictions / np.linalg.norm(batch_predictions)
            return normalized_search_vec
        else:
            return batch_predictions


if __name__ == '__main__':
    if sys.argv[0] is not None:
        product_description = str(sys.argv[0])
    else:
        product_description = "How to be a Data Scientist?"
    embeddingExtractor = GetItemEmbeddings()
    item_embedding = embeddingExtractor.get_item_embeddings(product_description)
    print('Size of embedding: ',item_embedding.shape[1])
    print(item_embedding)
