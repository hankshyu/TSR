import tensorflow as tf
import cv2
import numpy as np

from stn import spatial_transformer_network as transformer

img=cv2.imread('/Users/xuzihan/Desktop/SeaLionTourwithOptionalSnorkelingExperienceatJurienBay.jpg')
img=np.array(img)
H,W,C=img.shape
img=img[np.newaxis,:]
print(img.shape)
degree=np.deg2rad(45)
theta=np.array([
    [np.cos(degree),-np.sin(degree),0],
    [np.sin(degree),np.cos(degree),0]
])
x = tf.placeh(tf.float32, shape=[None, H, W, C])
with tf.variable_scope('spatial_transformer'):
    theta = theta.astype('float32')
    theta = theta.flatten()

    loc_in = H * W * C  # 输入维度
    loc_out = 6  # 输出维度
    W_loc = tf.Variable(tf.zeros([loc_in, loc_out]), name='W_loc')
    b_loc = tf.Variable(initial_value=theta, name='b_loc')

    # 运算
    fc_loc = tf.matmul(tf.zeros([1, loc_in]), W_loc) + b_loc
    h_trans = transformer(x, fc_loc)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    y = sess.run(h_trans, feed_dict={x: img})
    print(y.shape)

y = np.squeeze(np.array(y, dtype=np.uint8))
print(y.shape)
cv2.imshow('trasformedimg', y)
cv2.waitKey()
cv2.destroyAllWindows()