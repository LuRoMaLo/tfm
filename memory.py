import os, sys
from collections import namedtuple
# 3th party modules
import numpy as np
import tensorflow as tf
from main import DenseDNC

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILE = os.path.join(CURRENT_PATH, "data\\dataset.csv")
EMPISLON = 1e-6

class Memory:
    """
    This class defines the external memory for DNC
    """
    # data structure to represent the memory state
    state = namedtuple('memory_state',
                ['memory_matrix',
                'usage_vector',
                'link_matrix',
                'precedence_vector',
                'write_weighting',
                'read_weightings'
                ])

    def __init__(self, num_words=256, word_size=64, num_heads=4):
        self._num_words = num_words
        self._word_size = word_size
        self._num_heads = num_heads
        # sizes for each vector. used for parsing flatten input vectors
        self.state_size = Memory.state(
            tf.TensorShape([self._num_words, self._word_size]),         # memory matrix
            tf.TensorShape([self._num_words]),                          # usage vector
            tf.TensorShape([self._num_words, self._word_size]),         # linking matrix
            tf.TensorShape([self._num_words]),                          # precedence vector
            tf.TensorShape([self._num_words, self._num_heads]),         # write weighting
            tf.TensorShape([self._num_words])                           # read weighting
        )
        # initializes the memory State
        self.state = Memory.state(
            np.zeros([self._num_words, self._word_size]),               # memory matrix
            tf.fill([self._num_words, 1], EMPISLON),                    # usage vector
            np.zeros([self._num_words, self._word_size]),               # linking matrix
            tf.zeros([self._num_words, 1]),                             # precedence vector
            tf.fill([self._num_words, self._num_heads], EMPISLON),      # write weighting
            tf.fill([self._num_words, 1], EMPISLON)                     # read weighting
        )

    # properties for access the configuration, but cannot be modified 
    @property
    def num_words(self):
        return self._num_words
    @property
    def word_size(self):
        return self._word_size
    @property
    def num_heads(self):
        return self._num_heads
    
    def parse_memory_state(self, memory_state_vec):
        # partition = tf.constant( [  [0]*(self._num_words*self._word_size) +
        #                             [1]*self._num_words +
        #                             [2]*(self._num_words*self._word_size) +
        #                             [3]*self._num_words +
        #                             [4]*self._num_words*self._num_heads +
        #                             [5]*self._num_words
        #                         ], dtype=tf.int32)
        return tf.nest.pack_sequence_as(self.state_size, memory_state_vec)

    def __call__(self, interface):
        """
        Define the operation for the memory module        
        emites a set of read vectors and the memory state of this current
        time step

        Parameters:
            interface - DNC.interface - interface emitted by the controller
            prev_memory_state - Memory.state - previous memory state
        Returns:
            read_vectors - Tensor(shape=(num_words, num_heads)) - 
        """
        # WRITE HEADS
        usage_vector, write_weighting, memory_matrix, link_matrix, precedence_vector = self.write(interface)
        # READ HEADS
        read_weightings, read_vectors = self.read(interface)

        return read_vectors, Memory.state(
            memory_matrix=memory_matrix,
            usage_vector=usage_vector,
            link_matrix=link_matrix,
            precedence_vector=precedence_vector,
            write_weighting=write_weighting,
            read_weightings=read_weightings
        )

    
    def read(self, interface):
        read_keys = tf.reshape(interface.read_keys,[self._num_heads, self._word_size]) #R*W
        read_str = 1 + tf.nn.softplus(tf.expand_dims(interface.read_strengths, 0)) #1*R
        #the softmax distribution between the three read modes (backward, forward, lookup)
        #The read heads can use gates called read modes to switch between content lookup 
        #using a read key and reading out locations either forwards or backwards 
        #in the order they were written.
        read_modes = tf.nn.softmax(tf.reshape(interface.read_modes, [3, self._num_heads])) #3*R
        
        return None
    
    def write(self, interface, prev_mem_state):
        """
        Updates memory locations.
        Three ways:
            1- Writing (via dynamic memory alloc) into the locations specified on
            the allocation weighting vector.
            2- Writing (via content-based addressing) into de locations specified
            on the write content weighting vector.
            3- not writing at all at this time step

        Scallar allocation gate interpolate between the first two options.
        Scallar write gate determines to what degree the memory is or not writeen
        at this time step.        
        """
        memory_matrix=self.state.memory_matrix
        usage_vector=self.state.usage_vector
        link_matrix=self.state.link_matrix
        precedence_vector=self.state.precedence_vector
        write_weighting=self.state.write_weighting
        read_weightings=self.state.read_weightings
        # UPDATE USAGE VECTOR

        # CALCULATE ALLOCATION WEIGHTS

        # CONTENT-BASED ADDRESSING (lookup_weights)

        # OBTAINS WRITE_WEIGHTS

        # ERASE MEMORY

        # WRITE MATRIX

        # UPDATE TEMPORAL LINK MATRIX

        # UPDATE PRECEDENCE VECTOR

        #return new memory state
        

    @staticmethod
    def content_based_addressing(memory_matrix, lookup_key, key_strenght):
        """
        """
        #The l2 norm of a vector is the square root of the sum of the 
        #absolute values squared        
        norm_mem = tf.nn.l2_normalize(memory_matrix, 1, epsilon=EMPISLON) # num_words * word_size
        norm_lookup_key = tf.nn.l2_normalize(lookup_key, 0, epsilon=EMPISLON) # num_heads * word_size
        similarity = tf.matmul(norm_mem, tf.transpose(norm_lookup_key))
        return tf.nn.softmax(similarity*key_strenght, 0) # num_heads * word_size

    @staticmethod
    def allocation_addressing(num_words, usage_vector):
        """
        The allocation weighting indicates the degree each memory location
        is allocable (write protected) at the time step 't'

        Parameters:
            usage_vector - whether locations can be allocated

        """
        usage = (-1) * tf.transpose(usage_vector)
        usage_sorted, free_list = tf.nn.top_k(usage, k=num_words)
        usage_sorted *= -1
        cumprod = tf.math.cumprod(usage_sorted, axis=1, exclusive=True)
        unorder = (1 - usage_sorted) * cumprod
        alloc_weights = tf.zeros([num_words])
        I = tf.constant(np.identity(num_words, dtype=np.float32))

        for pos, idx in enumerate(tf.unstack(free_list[0])):
            #flatten
            m = tf.squeeze(tf.slice(I, [idx, 0], [1, -1]))
            alloc_weights += m*unorder[0,pos]
        return tf.reshape(alloc_weights, [num_words, 1])


    @staticmethod
    def temporal_linking_addressing():
        pass

def main(arguments):
    
    mem = Memory(num_words=2,word_size=3,num_heads=1)

    mem.state = Memory.state(
            memory_matrix=np.asarray([[-0.5, 0.01, 3.1],[0.2, 0.6, 1.2]]),
            usage_vector=mem.state.usage_vector,
            link_matrix=mem.state.link_matrix,
            precedence_vector=mem.state.precedence_vector,
            write_weighting=mem.state.write_weighting,
            read_weightings=mem.state.read_weightings
    )

    interface_init = DenseDNC.interface(
        read_keys=np.asarray([[-0.5, 0.01, 3.1]]),
        read_strengths=np.asarray([[1.0, 1.0, 1.0]]),
        write_key=np.asarray([[-0.5, 0.01, 3.1]]),
        write_strength=np.asarray([[0.01, 0.01, 0.01]]),
        erase_vector=np.asarray([[0.01, 0.01, 0.01]]),
        write_vector=,
        free_gates=,
        allocation_gate=,
        write_gate=,
        read_modes=)
    

if __name__=="__main__":
    main(sys.argv[1:])
