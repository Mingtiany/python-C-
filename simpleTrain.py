import os
import tensorflow as tf
import numpy as np

LEARNING_RATE_INIT = 0.001
DECAY_STEP = 1000
DECAY_RATE = 0.1
MOMENTUM = 0.9
# weight initialization
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def activate(x, acti_mode, scope=None):
    if acti_mode==0:
        return x
    elif acti_mode==1:
        return tf.nn.relu(x)
    elif acti_mode==2:
        return tf.nn.sigmoid(x)
    elif acti_mode==3:
        return (tf.nn.tanh(x) + x) / 2
    elif acti_mode==4:
        return (tf.nn.sigmoid(x) + x) / 2
    elif acti_mode==5:
        return tf.nn.leaky_relu(x)

def full_connect(x, num_filters_in, num_filters_out, acti_mode, is_train, keep_prob=1, name_w=None, name_b=None):
    w_fc = weight_variable([num_filters_in, num_filters_out], name_w)
    b_fc = bias_variable([num_filters_out], name_b)
    h_fc = tf.matmul(x, w_fc) + b_fc
    h_fc = activate(h_fc, acti_mode)
    h_fc = tf.cond(keep_prob < tf.constant(1.0), lambda: tf.nn.dropout(h_fc, keep_prob), lambda: h_fc)
    print(h_fc)
    return h_fc

def train_net(x, y, global_step, is_train, learning_rate_init, decay_step, decay_rate, momentum):
    acti_mode = 0
    x = tf.reshape(x, [-1, 1])
    y = tf.reshape(y, [-1, 1])
    pre_y = full_connect(x, 1, 1, acti_mode, is_train, keep_prob=1, name_w="fc_w", name_b="fc_b")
    print("train_net...")
    loss = tf.reduce_sum(tf.square(tf.subtract(pre_y, y)))
    opt_vars_all = [v for v in tf.trainable_variables()]
    learning_rate_current = tf.train.exponential_decay(learning_rate_init, global_step, decay_step, decay_rate,
                                                           staircase=True)
    train_step = tf.train.MomentumOptimizer(learning_rate_current, momentum).minimize(loss, var_list=opt_vars_all)

    return pre_y, loss, train_step, learning_rate_current, opt_vars_all

def predict_net(x, y, global_step, is_train, learning_rate_init, decay_step, decay_rate, momentum):
    acti_mode = 0
    x = tf.reshape(x, [-1, 1])
    y = tf.reshape(y, [-1, 1])
    pre_y = full_connect(x, 1, 1, acti_mode, is_train, keep_prob=1, name_w="fc_w", name_b="fc_b")
    print("test_net...")
    opt_vars_all = [v for v in tf.trainable_variables()]
    return pre_y, None ,None,  None ,opt_vars_all

def readData():
    x_list = []
    y_list = []
    batch_size = 2
    rand_list = np.random.rand(batch_size).astype(np.float32)
    for i in range(batch_size):
        x_list.append(rand_list[i])
        y_list.append(2 * rand_list[i] + 1)
    x = np.reshape(np.array(x_list), [-1, 1]).astype(np.float32)
    y = np.reshape(np.array(y_list), [-1, 1]).astype(np.float32)
    return x, y

def train():
    sess = tf.Session()
    x = tf.placeholder("float", [None, 1])
    y = tf.placeholder("float", [None, 1])
    is_train = tf.placeholder("float")
    global_step = tf.placeholder("float")

    pre_y, loss, train_step, learning_rate_current, opt_vars_all = train_net(x, y, global_step, is_train, LEARNING_RATE_INIT, DECAY_STEP, DECAY_RATE, MOMENTUM)

    saver = tf.train.Saver(opt_vars_all, write_version = tf.train.SaverDef.V2)

    sess.run(tf.global_variables_initializer())

    iter_num = 10000
    for i in range(iter_num):
        data_x, data_y = readData()
        loss_temp, learning_rate_temp, _ = sess.run([loss, learning_rate_current, train_step], feed_dict={x:data_x, y:data_y, global_step:(i+1), is_train:1})
        print(f"step{i+1}: loss:{loss_temp} lr:{learning_rate_temp}")
    saver.save(sess, "Mode/mode.data")

def predict():
    sess = tf.Session()
    modelpath = r'H:/vvencDLv2/deepLearning/simpleTest/Mode/mode.data'
    x = tf.placeholder("float", [None, 1])
    y = tf.placeholder("float", [None, 1])
    is_train = tf.placeholder("float")
    global_step = tf.placeholder("float")

    pre_y, loss, train_step, learning_rate_current, opt_vars_all = predict_net(x, y, global_step, is_train, LEARNING_RATE_INIT,
                                                                       DECAY_STEP, DECAY_RATE, MOMENTUM)

    saver = tf.train.Saver(opt_vars_all, write_version=tf.train.SaverDef.V2)
    saver.restore(sess, modelpath)

    data_x = np.reshape(np.array([1,2]),[-1,1]).astype(np.float32)
    data_y = np.reshape(np.array([1,2]),[-1,1]).astype(np.float32)

    pre_y_temp = sess.run([pre_y], feed_dict={x: data_x, y: data_y, global_step: 1, is_train: None})
    print(f"pred value: {pre_y_temp}")


if __name__ == '__main__':
    train()
    #predict()
