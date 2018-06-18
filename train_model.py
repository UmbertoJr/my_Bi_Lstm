import time as t
import tensorflow as tf
import os
import pickle

import shutil
import my_model as models
import my_lib as my


def train(sess, model, optimizer, generator, graph,num_optimization_steps,start, logdir='./logdir'):
    """ Train.
    
    Args:
        sess: A Session.
        model: A Model.
        optimizer: An Optimizer.
        generator: A generator that yields `(inputs, targets)` tuples, with
            `inputs` and `targets` both having shape `[dynamic_duration, 1]`.
        num_optimization_steps: An integer.
        logdir: A string. The log directory.
    """
    model_path = "./tmp/model.ckpt"
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
        
    with graph.as_default():
        tf.summary.scalar('loss', model.total_lost())
        tf.summary.scalar('loss_senses', model.get_loss_senses())
        tf.summary.scalar('loss_classes', model.get_loss_classes())
        tf.summary.scalar('loss_embeddings', model.get_loss_embeddings())


        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        foo = True
        words = 0
        for step in range(num_optimization_steps):
            while foo:
                words+=1
                inputs, targets, classes, senses = generator.next_sent()
                prec_x,prec_y , prec_c, prec_s = inputs, targets, classes, senses 
                if len(inputs)<1:
                    inputs, targets, classes, senses = prec_x,prec_y , prec_c, prec_s 
                    foo = False
                loss_, summary, _ = sess.run(
                    [model.total_lost(), summary_op, optimizer.optimize_op],
                    {model._inputs: inputs,
                     model.vectors_outputs: targets,
                     model.real_senses: senses,
                     model.classes: classes
                      })
                
                summary_writer.add_summary(summary, global_step=(step+1)*words)
                print('\rStep: %d   Sentences trained: %d. Loss Train: %.6f.' % (step +1, words, loss_), end='')
                if words*(step+1) % 100 ==0:
                    print("\n time exec : ",t.time()-start)
                    
                    # Save model weights to disk
                    save_path = saver.save(sess, model_path)
                    print("Model saved in file: %s" % save_path)
                    
                    
                    
                    
                    

tf.reset_default_graph()
generator = my.create_batch("TRAIN")        ### generator for training data
graph = tf.Graph()
start = t.time()
with graph.as_default():
    model = models.My_Model(hidden_Bi_Lstm=70, attention_hidden=30, graph=graph)
    optimizer = models.Optimizer(model.total_lost(), initial_learning_rate=1e-2, num_steps_per_decay=15000,
                          decay_rate=0.1, graph=graph, max_global_norm=1.0)

    sess = tf.Session()
    train(sess, model, optimizer, generator,graph,  num_optimization_steps=10, start=start)

