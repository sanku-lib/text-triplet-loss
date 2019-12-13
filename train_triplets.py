import tensorflow as tf
from preprocessing import PreProcessing
from model import TripletLoss
from tensorflow.contrib.tensorboard.plugins import projector



# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("output_embedding_size", 128, "size of last hidden layer")
tf.flags.DEFINE_integer('batch_size', 250, 'Batch size.')
tf.flags.DEFINE_integer('train_iter', 100, 'Total training iter')
tf.flags.DEFINE_integer('step', 50, 'Save after ... iteration')
tf.flags.DEFINE_float('learning_rate','0.0001','Learning rate')
tf.flags.DEFINE_integer('max_document_length',15,'maximum number of word in a sentence')
tf.flags.DEFINE_float('momentum','0.99', 'Momentum')
tf.flags.DEFINE_string('model', 'conv_net', 'model to run')
tf.flags.DEFINE_string('data_src', './data_repository/questions.csv', 'source of training dataset')

flags = tf.app.flags
FLAGS = flags.FLAGS

if __name__ == "__main__":
    # Setup Dataset
    dataset = PreProcessing(FLAGS.data_src, FLAGS.max_document_length)
    model = TripletLoss(sequence_length=dataset.X.shape[1],
                vocab_size=len(dataset.vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                output_embedding_size = FLAGS.output_embedding_size,
                dropout_keep_prob = FLAGS.dropout_keep_prob,
                embeddings_lookup= dataset.embeddings_lookup,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
    placeholder_shape = [None] + [dataset.X.shape[1]]
    print("placeholder_shape", placeholder_shape)

    # Setup Network
    next_batch = dataset.get_triplets_batch
    anchor_input = tf.placeholder(tf.int32, placeholder_shape, name='anchor_input')
    positive_input = tf.placeholder(tf.int32, placeholder_shape, name='positive_input')
    negative_input = tf.placeholder(tf.int32, placeholder_shape, name='negative_input')

    margin = 3.5
    anchor_output = model.conv_net(anchor_input, reuse=False)
    positive_output = model.conv_net(positive_input, reuse=True)
    negative_output = model.conv_net(negative_input, reuse=True)
    loss = model.triplet_loss(anchor_output, positive_output, negative_output, margin)

    # Setup Optimizer
    global_step = tf.Variable(0, trainable=False)

    train_step = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum, use_nesterov=True).minimize(loss,
                                                                                                             global_step=global_step)
    # Start Training
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Setup Tensorboard
        tf.summary.scalar('step', global_step)
        tf.summary.scalar('loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('model_triplet', sess.graph)
        # adding embeddings to projector
        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = "embedding/W"
        embed.metadata_path = "metadata.tsv"
        projector.visualize_embeddings(writer, config)
        print('Training...')
        # Train iter
        for i in range(FLAGS.train_iter):
            batch_anchor, batch_positive, batch_negative = next_batch(FLAGS.batch_size)
            _, l, summary_str = sess.run([train_step, loss, merged],
                                         feed_dict={anchor_input: batch_anchor, positive_input: batch_positive, negative_input: batch_negative})

            writer.add_summary(summary_str, i)
            print("\r#%d - Loss" % i, l)

            if (i + 1) % FLAGS.step == 0 or l <= 1e-9:
                print('Saving model. Step: ',i)
                saver.save(sess, "model_triplet/model.ckpt")
        saver.save(sess, "model_triplet/model.ckpt")
    print('Training completed successfully.')