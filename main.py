import sys
import os
from collections import namedtuple
# 3th party modules
import numpy as np
import tensorflow as tf
# my modules
import data.dataset as dt
from memory import Memory

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILE = os.path.join(CURRENT_PATH, "data\\dataset.csv")
EMPISLON = 1e-6

class DenseDNC(tf.keras.layers.Layer):
    """
    DNC STATE
    Non trainable variables
        usage vector. initialized to zeros
        write weights. Initialized to EPSILON
        memory matrix. Initialized to zeros
        temporal memory matrix. Initialized to zeros
        precedence vectors:
            backwards weightings. Initialized to zeros
            fordward weightings. Initialized to zeros
        read weights. Initialized to EPSILON
        read vectors. Initialized randomly.
    CONTROLLER
        input:
            X vector from dataset
            R set of read vectors from previous time step
        outpu:
            y vector with prediction
            interface vector of this current time step
    """ 

    state = namedtuple("dnc_state", [
        "memory_state",
        "controller_state",
        "read_vectors",
    ])

    interface = namedtuple("interface", [
        "read_keys",
        "read_strengths",
        "write_key",
        "write_strength",
        "erase_vector",
        "write_vector",
        "free_gates",
        "allocation_gate",
        "write_gate",
        "read_modes",
    ])

    def __init__(self, output_size, num_words=256, word_size=64, num_heads=4):
        super(DenseDNC, self).__init__()

        #define read + write vector size
        self.num_words = num_words #N
        self.word_size = word_size #W        
        #define number of read+write heads
        #we could have multiple, but just 1 for simplicity
        self.num_heads = num_heads #R
        self.output_size = output_size

    def build(self, input_shape):
        self.input_size = input_shape[-1]
        # size of output vector from controller that defines interactions with memory matrix
        self.interface_size = self.num_heads*self.word_size + 3*self.word_size + 5*self.num_heads + 3

        # the actual size of the neural network input after flatenning and
        # concatenating the input vector with the previously read vctors from memory
        self.nn_input_size = self.num_heads * self.word_size + self.input_size

        #size of output
        self.nn_output_size = self.output_size + self.interface_size
        
        #gaussian normal distribution for both outputs
        self.nn_out = tf.random.truncated_normal([1, self.output_size], stddev=0.1)
        self.interface_vec = tf.random.truncated_normal([1, self.interface_size], stddev=0.1)
        
        ### MEMORY 

        self.memory = Memory(self.num_words, self.word_size, self.num_heads)

        ###NETWORK VARIABLES
        self.controller = tf.keras.layers.LSTMCell(units=256)
        self.controller_to_interface = tf.keras.layers.Dense(self.interface_size, name='dense_controller_to_interface')

        self.final_output = tf.keras.layers.Dense(self.nn_output_size, name='dense_final_output')

        #super(DenseDNC, self).build(input_shape)  # Be sure to call this at the end

    def get_initial_state(self, batch_size=None):
        pass


    def parse_dnc_state_vector(self, dnc_state_vec):
        pass

    def parse_interface_vector(self, interface_vec):
        # creates a mask with the indices of each partition
        # example: partition = [[0,0,0,0,0,1,1,1,1,1,1,2,2,2,3,3,3,3,4,4,,5,5,6,6,7,8,9]]
        partition = tf.constant([[0]*(self.num_heads*self.word_size) + [1]*(self.num_heads) + [2]*(self.word_size) + [3] + \
                    [4]*(self.word_size) + [5]*(self.word_size) + \
                    [6]*(self.num_heads) + [7] + [8] + [9]*(self.num_heads*3)], dtype=tf.int32)
        #convert interface vector into a set of read write vectors
        #using tf.dynamic_partitions(Partitions interface_vec into 10 tensors using indices from partition)
        #(read_keys, read_str, write_key, write_str,
        # erase_vec, write_vec, free_gates, alloc_gate, write_gate, read_modes) = \
        #    tf.dynamic_partition(self.interface_vec, partition, 10)
        return DenseDNC.interface(tf.dynamic_partition(interface_vec, partition, 10))

    def call(self, x, prev_dnc_state):
        #reshape input
        
        return 0


    

def main(arguments):
    data = dt.Dataset(dt.DATASET_FILE, dt.VOCAB_FILE)

    input_data = np.asarray([[0,1],[1,0],[0,0],[0,1]])
    input_shape = input_data[0].shape

    print(input_data.shape)
    intut= tf.keras.layers.Input(shape=input_shape)
    
    #dnc = tf.keras.layers.Dense(32, activation="relu")(intut)
    dnc = DenseDNC(2)(intut)
    
    model = tf.keras.models.Model(inputs=intut, outputs=dnc)
    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['mae'])
    model.summary()
    


if __name__=="__main__":
    main(sys.argv[1:])


