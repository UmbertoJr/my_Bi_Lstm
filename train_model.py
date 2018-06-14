import os
import time as t
import tensorflow as tf
import my_lib as my
import xml.etree.ElementTree as ET
import re
import pickle
import numpy as np
from importlib import reload
# Load my lib to take sense embeddings
import my_lib as my
row = my.load_obj('row_in_the_file')

import my_model

tf.reset_default_graph()
generator = my.create_batch("TRAIN", row)   ### generator for training data
graph = tf.Graph()
start = t.time()
with graph.as_default():
    model = my_model.modello(100, graph)
    optimizer = my_model.Optimizer(model.loss(), initial_learning_rate=1e-2, num_steps_per_decay=15000,
                          decay_rate=0.1, graph=graph, max_global_norm=1.0)

    sess = tf.Session()
    my_model.train(sess, model, optimizer, generator,graph,  num_optimization_steps=45000, start=start)

