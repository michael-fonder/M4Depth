#!/usr/bin/env python
# Copyright Michael Fonder 2021. All rights reserved.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

try:
    xrange = xrange
except:
    xrange = range

EPS = 1e-12


class PipelineModel:
    def __init__(self, args = None):
        # *** Mandatory Properties *** #
        self.feature_list = []
        self.save_scopes = []
        self.learning_rate = None
        self.shard_prefix = "serialized_data_shard"
        self.is_training = None  # will be set during pipeline init

    def create_save_collection(self, scope = None, trainable=True):
        """ Collects all the variables in the scope. These variables will be saved in an individual checkpoint file by
            the pipeline.

            Args :
            scope : the scope of variables to collect
            trainable : a flag informing if the variables to collect should be trainable (useful for transfer learning)

            Return :
            /
        """
        if scope is None:
            scope = tf.compat.v1.get_variable_scope().name
        if not trainable:
            variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            for var in variables:
                var.trainable = False
        self.save_scopes.append([scope,trainable])
        return

    def preprocess_sample(self, data, is_training, thread_id = 0, step = None):
        """ Define your preprocessing function here
            
            Args :
            data : a list of tensors corresponding to the features specified in self.feature_list for a single data sample
            
            Return :
            data : a list of processed tensors to pass to the model
            summary_data : a list of processed tensors to pass to the export_results function
        """

        raise NotImplementedError

    def loss_func(self, data_batch, var_scope, step):
        """ Function using self.NN_model to compute losses which will be used to update the weights
        
            Args :
            data_batch  : the list of batched tensors processed by the preprocess_sample function
            var_scope   : a tensorflow var_scope
            
            Return :
            losses_dict : a dictionnary describing the losses and how to perform the update
                key = name of the loss
                item = a list of length 2 with the first item being the losse value and the second a list variables to
                be trained on this loss
        """

        raise NotImplementedError

    def make_lr(self, global_step):
        """ Define your learning rate here
            Args   : the global step number
            Return : a list of learning rate tensors
        """
        # self.learning_rate = [self.learning_rate]

        learning_rate = tf.constant(self.learning_rate, dtype=tf.float32)

        return learning_rate

    def make_opts(self, lr, current_step=0):
        """ Define your optimizers here. They should all implement the methods compute_gradients and apply_gradients.
            Args   : a list of learning rates (the output of self.make_lr)
            Return : a list of N optimizers (one for each loss)
        """

        raise NotImplementedError
        
    def eval_func(self, data_batch, var_scope):
        """ Function using self.NN_model to compute performance scores for a given model
        
            Args :
            data_batch  : a batch of data containing all the features encoded in the record files
            var_scope   : a tensorflow var_scope
            
            Return : 
            perfs_dict          : dictionnary containing different performance scores for the whole batch
                key = name of the performance score
                item = score for this performance for the batch (this value will be average over all the batches)
            nn_outputs          : list of tensors to get out of tensorflow and to pass to the export results function
        """

        raise NotImplementedError

    def export_results(self, nn_outputs, com_features):
        """ Function called when one desires to export results out of the script, e.g. save pictures, write logs,...

            Args :
            nn_outputs      : a list (len == nbre_gpus) of list of the neural network outputs (in numpy arrays)
            com_features    : a list (len == nbre_gpus) of list of comment features (== additionnal_data output from self.preprocess_func)

            Return : None
        """
        raise NotImplementedError
