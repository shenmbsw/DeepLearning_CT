import tensorflow as tf
import numpy as np
import data_generators as dg
import small_data_generators as sdg

def cnn_rnn(X):
    pad_heights = 470
    n_hidden_dim = 500
    batch_size = X.shape[0]
    LSTM = tf.contrib.rnn.LSTMCell(num_units = n_hidden_dim)
    initial_state = state = LSTM.zero_state(batch_size,tf.float32)
    xx = tf.transpose(X, perm=[0, 2, 3, 1])
    # To avoid resource depletion, use average_pooling to make the data array smaller.
    # place the height as an input channel of the cnn.
    x_ = tf.layers.average_pooling2d(inputs=xx, pool_size=[2,2],strides=[2,2], padding="valid")
    conv1 = tf.layers.conv2d(inputs=x_, filters=16, strides=(3, 3), kernel_size=[5, 5], padding="same", activation=tf.nn.relu, name='conv1')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2, name='pool1')
    conv2 = tf.layers.conv2d(inputs=pool1, filters=16, strides=(3, 3), kernel_size=[5, 5], padding="same", activation=tf.nn.relu,name='conv2')
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2, name='pool2')
    # the output will be sent to a flat layer and link to fully connection units.
    pool_flat = tf.contrib.layers.flatten(pool2, scope='poolflat')
    flat_dropout = tf.layers.dropout(inputs=pool_flat, rate=0.5)
    dense = tf.layers.dense(inputs=flat_dropout, units=200, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=1)
    return logits

def apply_classification_loss_rnn(model_function):
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):
            trainer = tf.train.AdamOptimizer()
            x_ = tf.placeholder(tf.float32, [1, 470, 512, 512])
            y_ = tf.placeholder(tf.int32, [1, 1])
            y_logits = model_function(x_)
            #1 dim cross_entropy_loss is the same as sigmoid logits comparation
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y_, tf.float32),logits=y_logits)
            train_op = trainer.minimize(loss)
            y_prob = tf.sigmoid(y_logits)
            y_pred = tf.cast(tf.round(y_prob), tf.int32)
            correct_prediction = tf.equal(y_pred, y_)
            accuracy = tf.cast(correct_prediction, tf.float32)
    model_dict = {'graph': g, 'inputs': [x_, y_], 'train_op': train_op,
                  'pred':y_pred,'prob':y_prob,
                  'accuracy': accuracy, 'loss': cross_entropy_loss}
    return model_dict

def train_model(model_dict, dataset_generators, epoch_n, print_every):
    with model_dict['graph'].as_default(), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_record = ()
        for epoch_i in range(epoch_n):
            collect_loss= []
            collect_acc = []
            for iter_i, data_batch in enumerate(dataset_generators['train']):
                train_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                train_to_compute = [model_dict['train_op'],model_dict['loss'],model_dict['accuracy']]
                _, loss, acc = sess.run(train_to_compute, feed_dict=train_feed_dict)
                collect_loss.append(loss)
                collect_acc.append(acc)
            #print train loss and accuracy
            print(np.mean(collect_loss, axis=0),np.mean(collect_acc, axis=0))

            collect_arr = []
            for a, test_batch in enumerate(dataset_generators['test']):
                test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                to_compute = [model_dict['loss'], model_dict['accuracy']]
                collect_arr.append(sess.run(to_compute, test_feed_dict))
            test_averages = np.mean(collect_arr, axis=0)
            test_record += tuple(test_averages)

            fmt = (epoch_i,) + tuple(test_averages)
            #print test loss and accuracy
            print('epoch {:d},  loss: {:.3f}, '
                  'accuracy: {:.3f}'.format(*fmt))

        print('the_test_record_is')
        for i in range(len(test_record)//2):
            print(test_record[2*i], test_record[2*i+1])
        for a, test_batch in enumerate(dataset_generators['test']):
            test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
            print(sess.run(model_dict['pred'], test_feed_dict))

def main():
    train_data_filepath = 'processed_train_data/train_data'
    train_labels_filepath = 'processed_train_data/train_labels'
    test_data_filepath = 'processed_test_data/test_data'
    test_labels_filepath = 'processed_test_data/test_labels'

    train_data_pathlist, train_labels_pathlist = dg.creat_datapath(train_data_filepath, train_labels_filepath, 8)
    test_data_pathlist, test_labels_pathlist = dg.creat_datapath(test_data_filepath, test_labels_filepath, 2)

    dataset_generators = {
            'train': dg.dataset_2d_iterator(train_data_pathlist, train_labels_pathlist),
            'test':  dg.dataset_2d_iterator(test_data_pathlist, test_labels_pathlist),
        }

    model_dict = apply_classification_loss_rnn(cnn_rnn)
    train_model(model_dict, dataset_generators, epoch_n = 30, print_every = 1)

main()
