import os
import tensorflow as tf
import numpy as np

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

def full_connect(x, num_filters_in, num_filters_out, acti_mode, keep_prob=1, name_w=None, name_b=None):
    w_fc = weight_variable([num_filters_in, num_filters_out], name_w)
    b_fc = bias_variable([num_filters_out], name_b)
    h_fc = tf.matmul(x, w_fc) + b_fc
    h_fc = activate(h_fc, acti_mode)
    h_fc = tf.cond(keep_prob < tf.constant(1.0), lambda: tf.nn.dropout(h_fc, keep_prob), lambda: h_fc)
    print(h_fc)
    return h_fc

def predict_net(x):
    acti_mode = 0
    x = tf.reshape(x, [-1, 1])
    pre_y = full_connect(x, 1, 1, acti_mode, keep_prob=1, name_w="fc_w", name_b="fc_b")
    #print("test_net...")
    return pre_y

sess = tf.Session()
modelpath = r'H:/vvencDLv2/deepLearning/simpleTest/Mode/mode.data'
x = tf.placeholder("float", [None, 1])

pre_y = predict_net(x)

saver = tf.train.Saver()
saver.restore(sess, modelpath)

def predict(data_x):
    data_x = np.reshape(np.array(data_x), [-1, 1]).astype(np.float32)
    pre_y_temp = sess.run([pre_y], feed_dict={x: data_x})
    pre_y_temp = np.squeeze(pre_y_temp)
    #print(f"pred value: {pre_y_temp}")
    ret = pre_y_temp.tolist()
    return ret
'''
if __name__ == '__main__':
    data_x = [10, 20]
    predict(data_x)
'''