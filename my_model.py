import tensorflow as tf
import numpy as np
import time as t

class modello:
    def __init__(self, hidden_Bi_Lstm_dim, graph):
        # Initialization of given values
        self.input_size = 400
        self.hidden_layer_size = hidden_Bi_Lstm_dim
        self.target_size = 400
        self.weights = {}
        self.direction = ""
        self.graph = graph
        with self.graph.as_default():
            # Weights for last output layers
            with tf.name_scope("output_layer") as scope:
                self.Wo = tf.Variable(tf.truncated_normal([self.hidden_layer_size * 2, self.target_size],
                                                          mean=0, stddev=.01))
                self.bo = tf.Variable(tf.truncated_normal([self.target_size], mean=0, stddev=.01))
                

            for direction in ["forward", "backward"]:
                with tf.name_scope("Lstm_new_candidate_gate_layer_"+direction) as scope:
                    # selection weights for update gate layer
                    self.weights[direction + "_Wi"] = tf.Variable(tf.zeros([self.input_size,
                                                                            self.hidden_layer_size]), name="W")
                    self.weights[direction + "_Ui"] = tf.Variable(tf.zeros([self.hidden_layer_size,
                                                                            self.hidden_layer_size]), name="U")
                    self.weights[direction + "_bi"] = tf.Variable(tf.zeros([self.hidden_layer_size]), name="b")


                    # new candidates for Memory gate
                    self.weights[direction + "_Wc"] = tf.Variable(tf.zeros([self.input_size,
                                                                            self.hidden_layer_size]), name="W_c")
                    self.weights[direction + "_Uc"] = tf.Variable(tf.zeros([self.hidden_layer_size,
                                                                            self.hidden_layer_size]), name="U_c")
                    self.weights[direction + "_bc"] = tf.Variable(tf.zeros([self.hidden_layer_size]), name="b_c")


                with tf.name_scope("Lstm_forget_gate_layer_"+direction) as scope:
                    # Forget gate weights
                    self.weights[direction + "_Wf"] = tf.Variable(tf.zeros([self.input_size,
                                                                            self.hidden_layer_size]), name="W_forget")
                    self.weights[direction + "_Uf"] = tf.Variable(tf.zeros([self.hidden_layer_size,
                                                                            self.hidden_layer_size]), name="U_forget")
                    self.weights[direction + "_bf"] = tf.Variable(tf.zeros([self.hidden_layer_size]), name="b_forget")

                with tf.name_scope("Lstm_output_layer_"+direction) as scope:
                    # Output gate weights
                    self.weights[direction + "_Wog"] = tf.Variable(tf.zeros([self.input_size,
                                                                             self.hidden_layer_size]),name="W_output")
                    self.weights[direction + "_Uog"] = tf.Variable(tf.zeros([self.hidden_layer_size,
                                                                             self.hidden_layer_size]),name="u_ouput")
                    self.weights[direction + "_bog"] = tf.Variable(tf.zeros([self.hidden_layer_size]), name="b_output")
            
            
            with tf.name_scope("input") as scope:
                # Placeholder for input vector with shape[batch, seq, embeddings]
                self._inputs = tf.placeholder(tf.float32,
                                              shape=[None, self.input_size],
                                              name='inputs')

                # Reversing the inputs by sequence for backward pass of the LSTM
                self._inputs_rev = tf.reverse(self._inputs, axis= [0], name="reverse")
                
                # Target variable
                self._targets = tf.placeholder(tf.float32,
                                              shape= [None, self.input_size],
                                              name = "targets")
                          
            with tf.name_scope("initial_state") as scope:
                self.initial_hidden = self._inputs[0, :]
                self.initial_hidden = tf.matmul(tf.reshape(self.initial_hidden,[1,400]), tf.zeros([self.input_size,
                                                                                                   self.hidden_layer_size]))

                self.initial_hidden = tf.stack([self.initial_hidden, self.initial_hidden])

            #self.loss = self.loss()
            
            
    
    def my_LSTM(self, previous_hidden_memory_tuple, input_x):
        """
        This function takes previous hidden state
        and memory tuple with input and
        outputs current hidden state.
        """
        with self.graph.as_default():
            direction = self.direction
            with tf.name_scope("LSTM_buillder_"+ direction) as scope:
                previous_hidden_state, c_prev = tf.unstack(previous_hidden_memory_tuple) # divide in due parti l'input 

                # Forget gate Layer
                with tf.name_scope("forget_layer_" + direction) as scope:
                    f = tf.sigmoid(
                        tf.matmul(tf.reshape(input_x,[1,400]), self.weights[direction + "_Wf"], name="1") +
                        tf.matmul(previous_hidden_state, self.weights[direction + "_Uf"], name= "2") +\
                        self.weights[direction + "_bf"], name="forget_values"
                    )

                
                with tf.name_scope("new_candidate_values") as scope:
                    # Input Gate Layer
                    i = tf.sigmoid(
                        tf.matmul(tf.reshape(input_x,[1,400]), self.weights[direction + "_Wi"]) +
                        tf.matmul(previous_hidden_state, self.weights[direction + "_Ui"], name="2") +\
                        self.weights[direction + "_bi"], name="update_values"
                    )
                        
                    # New Memory Cell
                    c_ = tf.nn.tanh(
                        tf.matmul(tf.reshape(input_x,[1,400]), self.weights[direction + "_Wc"]) +
                        tf.matmul(previous_hidden_state, self.weights[direction + "_Uc"]) +\
                        self.weights[direction + "_bc"], name="new_values"
                    )
                with tf.name_scope("Final_Memory_cell") as scope:
                    # Final Memory cell
                    c = f * c_prev + i * c_

                
                with tf.name_scope("output_layer") as scope:
                    # Output Gate
                    o = tf.sigmoid(
                        tf.matmul(tf.reshape(input_x,[1,400]), self.weights[direction + "_Wog"]) +
                        tf.matmul(previous_hidden_state, self.weights[direction + "_Uog"]) +\
                        self.weights[direction + "_bog"], name ="output_before_filter"
                    )
                    # Current Hidden state
                    current_hidden_state = o * tf.nn.tanh(c)
                    
        return tf.stack([current_hidden_state, c])

     # Function to get the hidden and memory cells after forward pass
    def get_states(self):
        """
        Iterates through time/ sequence to get all hidden state
        """
        self.direction = "forward"
        # Getting all hidden state throuh time    scan is like a for loop but take track of gradient
        all_hidden_memory_states = tf.scan(self.my_LSTM,
                                               self._inputs,
                                               initializer=self.initial_hidden,
                                               name='states_forward')
 
        
        all_hidden_states_f = all_hidden_memory_states[:, 0, :, :]
        all_memory_states_f = all_hidden_memory_states[:, 1, :, :]

        # Reversing the hidden and memory state to get the final hidden and
        # memory state
        last_hidden_states = all_hidden_states_f[-1]
        last_memory_states = all_memory_states_f[-1]

        # For backward pass using the last hidden and memory of the forward
        # pass
        initial_hidden_b = tf.stack([last_hidden_states, last_memory_states])

        # Getting all hidden state throuh time
        self.direction = "backward"
        all_hidden_memory_states = tf.scan(self.my_LSTM,
                                           self._inputs_rev,
                                           initializer=initial_hidden_b,
                                           name='states_backward')

        # Now reversing the states to keep those in original order
        all_hidden_states_b = tf.reverse(all_hidden_memory_states[:, 0, :, :], axis=[2])
        all_memory_states_b = tf.reverse(all_hidden_memory_states[:, 1, :, :], axis=[2])

        return all_hidden_states_f, all_memory_states_f, all_hidden_states_b, all_memory_states_b

        
        
        # Function to concat the hiddenstates for backward and forward pass
    def get_concat_hidden(self):

                    # Getting hidden and memory for the forward and backward pass
        all_hidden_states_f, all_memory_states_f, all_hidden_states_b, all_memory_states_b= self.get_states()

        concat_hidden = tf.concat([all_hidden_states_f, all_hidden_states_b],axis = 2)
        return concat_hidden
    
    # Function to get output from a hidden layer
    def get_output(self, hidden_state):
        """
        This function takes hidden state and returns output
        """
        output = tf.nn.sigmoid(tf.matmul(hidden_state, self.Wo) + self.bo)

        return output

    # Function for getting all output layers
    def get_outputs(self):
        """
        Iterating through hidden states to get outputs for all timestamp
        """
        all_hidden_states = self.get_concat_hidden()

        all_outputs = tf.map_fn(self.get_output, all_hidden_states)

        return all_outputs

    def loss(self):
        with self.graph.as_default():
            # Getting all outputs from rnn
            outputs = self.get_outputs()
            outputs = tf.reshape(outputs, [-1,400])
            # Computing the Cross Entropy loss
            cross_entropy = tf.losses.mean_squared_error(self._targets , outputs)
            return cross_entropy


class Optimizer(object):
    
    def __init__(self, loss, initial_learning_rate, num_steps_per_decay,
                 decay_rate,graph, max_global_norm=1.0):
        """ Create a simple optimizer.
        
        This optimizer clips gradients and uses vanilla stochastic gradient
        descent with a learning rate that decays exponentially.
        
        Args:
            loss: A 0-D float32 Tensor.
            initial_learning_rate: A float.
            num_steps_per_decay: An integer.
            decay_rate: A float. The factor applied to the learning rate
                every `num_steps_per_decay` steps.
            max_global_norm: A float. If the global gradient norm is less than
                this, do nothing. Otherwise, rescale all gradients so that
                the global norm because `max_global_norm`.
        """
        with graph.as_default():
            trainables = tf.trainable_variables()
            grads = tf.gradients(loss, trainables)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
            grad_var_pairs = zip(grads, trainables)

            global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
            learning_rate = tf.train.exponential_decay(
                initial_learning_rate, global_step, num_steps_per_decay,
                decay_rate, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self._optimize_op = optimizer.apply_gradients(grad_var_pairs,
                                                          global_step=global_step)

    @property
    def optimize_op(self):
        """ An Operation that takes one optimization step. """
        return self._optimize_op
    
    


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
    model_path = "/tmp/model.ckpt"
    #if os.path.exists(logdir):
        #shutil.rmtree(logdir)
        
    with graph.as_default():
        tf.summary.scalar('loss', model.loss())


        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        foo = True
        words = 0
        for step in range(num_optimization_steps):
            while foo:
                words+=1
                inputs, targets = generator.next_sent()
                prec_x,prec_y = inputs, targets
                if len(inputs)<1:
                    x,y = prec_x,prec_y
                    foo = False
                inputs, targets = np.array(inputs), np.array([ targets[j][0] for j in range(len(inputs))])
                loss_, summary, _ = sess.run(
                    [model.loss(), summary_op, optimizer.optimize_op],
                    {model._inputs: inputs, model._targets: targets})
                summary_writer.add_summary(summary, global_step=step)
                print('\rStep: %d   Sentences trained: %d. Loss Train: %.6f.' % (step +1, words, loss_), end='')
                if words*(step+1) % 100 ==0:
                    print("\n time exec : ",t.time()-start)
                    
                    # Save model weights to disk
                    save_path = saver.save(sess, model_path)
                    print("Model saved in file: %s" % save_path)
