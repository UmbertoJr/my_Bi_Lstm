import tensorflow as tf
import numpy as np

class My_Model:
  
  def __init__(self, hidden_Bi_Lstm, attention_hidden,  graph):
    self.embeddings_dim = 400  # dim sense-embeddings
    self.output_class = 36
    self.output_senses = 25915
    self.hidden_Bi_Lstm = hidden_Bi_Lstm
    self.attention_hidden = attention_hidden

    self.graph = graph
    
    with self.graph.as_default():
      with tf.name_scope("outputs") as scope:
        self.vectors_outputs = tf.placeholder(tf.float32,
                                        shape=[None, self.embeddings_dim],
                                        name='outputs_vec')
        self.classes = tf.placeholder(tf.float16,
                                        shape=[None, self.output_class],
                                        name='classes')
        self.real_senses = tf.placeholder(tf.float16,
                                        shape=[None, self.output_senses],
                                        name='senses')



      
      with tf.name_scope("input") as scope:
          # Placeholder for input vector with shape[batch, seq, embeddings]
          self._inputs = tf.placeholder(tf.float32,
                                        shape=[None, self.embeddings_dim],
                                        name='inputs')

      self.Bi_Lstm = My_Bi_Lstm(inputs= self._inputs,
                                hidden_Bi_Lstm_dim= self.hidden_Bi_Lstm,
                                graph=self.graph)  # Build the instance graph
      self.out = self.Bi_Lstm.get_concat_hidden() # Build a tensor output of the BI-LSTM of dim [T, 2*Hidden]

      self.attent = My_Attention_layer(self.out, attention_size = self.attention_hidden, graph=self.graph)
      self.last_layer = self.attent.build() # This is the last hidden layer and is the concatenation of Bi and attention mecchanism
                                            # dim [T, 4*Hidden]
      
      self.weig = {}
      with tf.name_scope("w_for_loss") as scope:
        self.weig["W_vec"] = tf.Variable(tf.random_normal([4 * self.hidden_Bi_Lstm, self.embeddings_dim], stddev=0.1))
        self.weig["b_vec"] = tf.Variable(tf.random_normal([self.embeddings_dim], stddev=0.1))
        self.weig["W_classes"] = tf.Variable(tf.random_normal([4 * self.hidden_Bi_Lstm, self.output_class], stddev=0.1))
        self.weig["b_classes"] = tf.Variable(tf.random_normal([self.output_class], stddev=0.1))
        self.weig["W_senses"] = tf.Variable(tf.random_normal([4 * self.hidden_Bi_Lstm, self.output_senses], stddev=0.001))
        self.weig["b_senses"] = tf.Variable(tf.random_normal([self.output_senses], stddev=0.001))
      
  def get_loss_embeddings(self):
    vec_pred = tf.tensordot(self.last_layer,self.weig["W_vec"], axes=1) + self.weig["b_vec"]
    loss = tf.losses.mean_squared_error(labels=self.vectors_outputs, predictions=vec_pred)
    return loss
  
  def get_loss_classes(self):
    vec_pred = tf.nn.softmax(tf.tensordot(self.last_layer,self.weig["W_classes"], axes=1) + self.weig["b_classes"])
    loss = tf.losses.softmax_cross_entropy(onehot_labels=self.classes, logits=vec_pred)
    return loss
  
  def get_loss_senses(self):
    vec_pred = tf.tensordot(self.last_layer,self.weig["W_senses"], axes=1) + self.weig["b_senses"]
    loss = tf.losses.mean_squared_error(labels=self.real_senses, predictions=vec_pred)
    return loss
  
  def total_lost(self):
    return self.get_loss_embeddings() + self.get_loss_classes() + 10* self.get_loss_senses()
   
   
   
class My_Attention_layer:
  
  def __init__(self,inputs,attention_size,  graph, session = False, return_alphas=False):
    self.inputs = inputs
    self.graph = graph
    self.return_alphas = return_alphas
    self.session = session
    with self.graph.as_default():
      
      self.hidden_size = inputs.shape[1].value  # D value - hidden size of the RNN layer

      # Trainable parameters
      self.w_omega = tf.Variable(tf.random_normal([self.hidden_size, attention_size], stddev=0.1))
      self.b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
      self.u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    
  def build(self):
    
    with self.graph.as_default():
      
      with tf.name_scope('Attention_mechanism_v'):
          # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
          #  the shape of `v` is (T,D)*(D,A)=(T,A), where A=attention_size
          v = tf.tanh(tf.tensordot(self.inputs, self.w_omega, axes=1) + self.b_omega)

      # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
      vu = tf.tensordot(v, self.u_omega, axes=1, name='vu')  # (T) shape
      self.alphas = tf.nn.softmax(vu, name='alphas')         # (T) shape
      if self.session:
        print("alpha dim : ",self.session.run(tf.shape(self.alphas)))

      # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
      self.output_atten = tf.tensordot(self.alphas, self.inputs, axes=1)
      
      self.final_outputs = self.get_outputs()
      if not self.return_alphas:
        return self.final_outputs
      else:
        return self.output_atten, self.alphas



    # Function to get output from a hidden layer
  def concatenate(self, inputs_1_word):
      final_output = tf.concat([inputs_1_word, self.output_atten], axis = 0)
      return final_output

  # Function for getting all output layers
  def get_outputs(self):
      all_outputs = tf.map_fn(self.concatenate, self.inputs)
      return all_outputs
        

      


class My_Bi_Lstm:
    def __init__(self,inputs, hidden_Bi_Lstm_dim, graph):
        # Initialization of given values
        self.input_size = inputs.shape[1].value # dimention embeddings
        self.hidden_layer_size = hidden_Bi_Lstm_dim
        self.weights = {}
        self.direction = ""
        self.graph = graph
        with self.graph.as_default():
          
          with tf.name_scope("input") as scope:
            # Placeholder for input vector with shape[batch, seq, embeddings]
            self._inputs = inputs

            # Reversing the inputs by sequence for backward pass of the LSTM
            self._inputs_rev = tf.reverse(self._inputs, axis= [0], name="reverse")


            for direction in ["forward", "backward"]:
                with tf.name_scope("Lstm_new_candidate_gate_layer_"+direction) as scope:
                    # selection weights for update gate layer
                    self.weights[direction + "_Wi"] = tf.Variable(tf.truncated_normal([self.input_size,self.hidden_layer_size],
                                                                                      stddev=(1/np.sqrt(self.hidden_layer_size))), name="W")
                    self.weights[direction + "_Ui"] = tf.Variable(tf.truncated_normal([self.hidden_layer_size,self.hidden_layer_size], 
                                                                                      stddev=(1/np.sqrt(self.hidden_layer_size))), name="U")
                    self.weights[direction + "_bi"] = tf.Variable(tf.truncated_normal([self.hidden_layer_size],
                                                                                      stddev=(1/np.sqrt(self.hidden_layer_size))), name="b")


                    # new candidates for Memory gate
                    self.weights[direction + "_Wc"] = tf.Variable(tf.truncated_normal([self.input_size,self.hidden_layer_size],
                                                                                      stddev=(1/np.sqrt(self.hidden_layer_size))), name="W_c")
                    self.weights[direction + "_Uc"] = tf.Variable(tf.truncated_normal([self.hidden_layer_size,self.hidden_layer_size], 
                                                                                      stddev=(1/np.sqrt(self.hidden_layer_size))), name="U_c")
                    self.weights[direction + "_bc"] = tf.Variable(tf.truncated_normal([self.hidden_layer_size], 
                                                                                      stddev=(1/np.sqrt(self.hidden_layer_size))), name="b_c")


                with tf.name_scope("Lstm_forget_gate_layer_"+direction) as scope:
                    # Forget gate weights
                    self.weights[direction + "_Wf"] = tf.Variable(tf.truncated_normal([self.input_size,self.hidden_layer_size], 
                                                                                      stddev=(1/np.sqrt(self.hidden_layer_size))), name="W_forget")
                    self.weights[direction + "_Uf"] = tf.Variable(tf.truncated_normal([self.hidden_layer_size,self.hidden_layer_size], 
                                                                                      stddev=(1/np.sqrt(self.hidden_layer_size))), name="U_forget")
                    self.weights[direction + "_bf"] = tf.Variable(tf.truncated_normal([self.hidden_layer_size], 
                                                                                      stddev=(1/np.sqrt(self.hidden_layer_size))), name="b_forget")

                with tf.name_scope("Lstm_output_layer_"+direction) as scope:
                    # Output gate weights
                    self.weights[direction + "_Wog"] = tf.Variable(tf.truncated_normal([self.input_size,self.hidden_layer_size], 
                                                                                       stddev=(1/np.sqrt(self.hidden_layer_size))),name="W_output")
                    self.weights[direction + "_Uog"] = tf.Variable(tf.truncated_normal([self.hidden_layer_size,self.hidden_layer_size], 
                                                                                       stddev=(1/np.sqrt(self.hidden_layer_size))),name="u_ouput")
                    self.weights[direction + "_bog"] = tf.Variable(tf.truncated_normal([self.hidden_layer_size], 
                                                                                       stddev=(1/np.sqrt(self.hidden_layer_size))), name="b_output")
            
            
         
                          
            with tf.name_scope("initial_state") as scope:
                self.initial_hidden = self._inputs[0, :]
                self.initial_hidden = tf.matmul(tf.reshape(self.initial_hidden,[1,self.input_size]), tf.zeros([self.input_size,
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
                        tf.matmul(tf.reshape(input_x,[1,self.input_size]), self.weights[direction + "_Wf"], name="1") +
                        tf.matmul(previous_hidden_state, self.weights[direction + "_Uf"], name= "2") +\
                        self.weights[direction + "_bf"], name="forget_values"
                    )

                
                with tf.name_scope("new_candidate_values") as scope:
                    # Input Gate Layer
                    i = tf.sigmoid(
                        tf.matmul(tf.reshape(input_x,[1,self.input_size]), self.weights[direction + "_Wi"]) +
                        tf.matmul(previous_hidden_state, self.weights[direction + "_Ui"], name="2") +\
                        self.weights[direction + "_bi"], name="update_values"
                    )
                        
                    # New Memory Cell
                    c_ = tf.nn.tanh(
                        tf.matmul(tf.reshape(input_x,[1,self.input_size]), self.weights[direction + "_Wc"]) +
                        tf.matmul(previous_hidden_state, self.weights[direction + "_Uc"]) +\
                        self.weights[direction + "_bc"], name="new_values"
                    )
                with tf.name_scope("Final_Memory_cell") as scope:
                    # Final Memory cell
                    c = f * c_prev + i * c_

                
                with tf.name_scope("output_layer") as scope:
                    # Output Gate
                    o = tf.sigmoid(
                        tf.matmul(tf.reshape(input_x,[1,self.input_size]), self.weights[direction + "_Wog"]) +
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
        return tf.reshape(concat_hidden, [-1,2* self.hidden_layer_size])
      
    def get_memory_states(self):
        # Getting hidden and memory for the forward and backward pass
        all_hidden_states_f, all_memory_states_f, all_hidden_states_b, all_memory_states_b= self.get_states()

        concat_hidden = tf.concat([all_memory_states_f, all_memory_states_b],axis = 2)
        return tf.reshape(concat_hidden, [-1,2*self.hidden_layer_size])
      
    
    

    

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
    
    



      