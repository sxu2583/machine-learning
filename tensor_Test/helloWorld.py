# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 22:21:55 2017

@author: shiha
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))