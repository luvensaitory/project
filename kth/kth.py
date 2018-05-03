import tensorflow as tf
import numpy as np
from PIL import Image

filename_queue = tf.train.string_input_producer([("./p01w1/p1w1_%04d.png" % i ) for i in _ ])
  #  list of files to read

print (filename_queue)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_png(value)
  # use png or jpg decoder based on your files.

init_op = tf.global_variables_initializer()

with tf.Session() as sess:

  sess.run(init_op)

  # Start populating the filename queue.

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1): #length of your filename list
    image = my_img.eval() 
  #here is your image Tensor :) 

  print(image.shape)
  '''
  Image.fromarray(np.asarray(image)).show()

  coord.request_stop()
  coord.join(threads)
  '''
