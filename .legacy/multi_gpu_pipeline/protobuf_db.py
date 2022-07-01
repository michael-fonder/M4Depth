# Protobuffer files handler
#
# Author : Michael Fonder
#
# This file contains the scripts necessary to encode some data samples into
# protobuffer files (ProtoBufSerializer class) and to extract minibatches out
# of a protobuffer database (ProtoBufDeserializer class).
#
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from datetime import datetime
import multiprocessing as mp
import weakref
import queue
import numpy as np

try:
	from tensorflow.contrib import nccl
	have_nccl = True
except ImportError:
	have_nccl = False
	print("WARNING: NCCL support not available")

try:
    xrange = xrange
except:
    xrange = range

def unwrap_self_samples_writer(*arg):
    return ProtoBufSerializer.samples_writer(*arg)


class ProtoBufSerializer:
    """ This class can be used to serialize and save a database into protobuffer record files.
    """
    
    def __init__(self, features_list, data_dir = "serialized_DB", shard_prefix = "serialized_data_shard", samples_per_shard = 512, jpeg_q = 90, nbre_threads=4):
        """ Args:
            features_list       :  list of features; each feature should be a list of string containing in this order : the name of the feature and its type;
                                the different types available can be found in the dictionnaries in the code below
            data_dir            : the directory where the protobuffer records should be written
            shard_prefix        : the prefix for the protobuffer shard records
            samples_per_shard   : the number of samples to be written in each record shard
            jpeg_q              : the desired quality for the jpeg compression when encoding jpeg images
        """
        
        # Available types    
        self.name2var_dict = {
            "int64"          : self._int64_feature,
            "float32"        : self._float32_feature,
            "int64_list"     : self._int64_feature,
            "float32_list"   : self._float32_feature,
            "float16_mat"    : self._float16mat_feature,
            "string"         : self._generic2feature,
            "jpeg"           : self._jpeg2feature,
            "png"            : self._png2feature,
            "png16"          : self._png162feature,
            "bytes"          : self._generic2feature,
            }
        self.sess = None
        self.cnter = 0
        self.data_dir = data_dir
        self.features_list = features_list
        self.samples_per_shard = samples_per_shard
        self.jpeg_q = jpeg_q
        self.shard_prefix = shard_prefix
        
        self.NBRE_THREADS = nbre_threads
        # self.sess = tf.Session()

        self.terminate_event = mp.Event()
        self.data_queue = mp.Queue(self.NBRE_THREADS*128)
        self.shards_queue = mp.Queue(self.NBRE_THREADS*int(128/samples_per_shard+1))
        self.shards_global_cnter = 0

        for i in range(self.NBRE_THREADS):
            process = mp.Process(target=self.samples_writer,
                                 args=( self.data_queue, self.shards_queue, self.terminate_event))
            process.start()

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def __del__(self):
        print("end of test")
        self.terminate_event.set()
            
    def process_sample(self, single_sample):
        """ Processes and saves list of features corresponding to a single sample as TFRecord in 1 thread.
            Args:
            single_example : a list containing the features specified in self.features_list in the correct order
        """
        while not self.shards_queue.full():
            self.shards_queue.put(self.shards_global_cnter)
            self.shards_global_cnter +=1

        self.data_queue.put(single_sample)
        
    def end_serializing(self):
        """ Ensures that the last file opened to write the records is closed properly
            Note : This function MUST be called after all data sample have been processed
           
           Return : void
        """
        self.terminate_event.set()

    def samples_writer(self, data_queue, shards_queue, terminate_event):
        print("new process")

        try:
            # while not terminate_event.is_set():
            #     if data_queue.empty():
            #         i=1
            writer = None
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            samples_cnter = self.samples_per_shard
            shard_id = 0

            while not terminate_event.is_set() or not data_queue.empty():
                try:
                    sample = data_queue.get(block=True, timeout=2)
                except queue.Empty as e:
                    print("data queue empty")
                    continue

                if len(sample) != len(self.features_list):
                    raise Exception("Number of features mismatch")

                if samples_cnter >= self.samples_per_shard:
                    if writer is not None:
                        print("%d samples written into %s-%.5d" % (samples_cnter, self.shard_prefix, shard_id))
                        writer.close()
                    shard_id = shards_queue.get()
                    # print(shard_id)
                    output_filename = '%s-%.5d' % (self.shard_prefix, shard_id)
                    output_file = os.path.join(self.data_dir, output_filename)
                    writer = tf.python_io.TFRecordWriter(output_file)
                    samples_cnter = 0

                samples_cnter += 1

                # write data into file
                sample_dict = {}
                for i, (name, var_type) in enumerate(self.features_list):
                    sample_dict[name] = self.name2var_dict[var_type](sess, sample[i])

                sample_example = tf.train.Example(features=tf.train.Features(feature=sample_dict))
                writer.write(sample_example.SerializeToString())

            print("%d samples written into %s-%.5d" % (samples_cnter, self.shard_prefix, shard_id))
            writer.close()

        except Exception as e:
            print(e)
    
        
    ### "private" functions
    
    def _jpeg2feature(self, sess, image_data):
        """Wrapper for inserting jpeg images into Example proto."""
        if type(image_data) is str:
            bytes_jpg = self._bytes_feature(sess, tf.compat.as_bytes(open(image_data,'rb').read()))
        else:
            raw_data = tf.placeholder(dtype=tf.uint8)
            encoded_jpeg = tf.image.encode_jpeg(raw_data, quality=self.jpeg_q, optimize_size = True)
            encoded_jpeg = sess.run(encoded_jpeg,  feed_dict={raw_data: image_data})
            bytes_jpg = self._bytes_feature(sess, tf.compat.as_bytes(encoded_jpeg))
        return bytes_jpg
        
    def _png2feature(self, sess, image_data):
        """Wrapper for inserting png images into Example proto."""
        if type(image_data) is str:
            bytes_png = self._bytes_feature(sess, tf.compat.as_bytes(open(image_data,'rb').read()))
        else:
            raw_data = tf.placeholder(dtype=tf.uint8)
            encoded_png = tf.image.encode_png(raw_data)
            encoded_png = sess.run(encoded_png,  feed_dict={raw_data: image_data})
            bytes_png = self._bytes_feature(sess, tf.compat.as_bytes(encoded_png))
        return bytes_png

    def _png162feature(self, sess, image_data):
        """Wrapper for inserting png images into Example proto."""
        if type(image_data) is str:
            bytes_png = self._bytes_feature(sess, tf.compat.as_bytes(open(image_data,'rb').read()))
        else:
            raw_data = tf.placeholder(dtype=tf.uint16)
            encoded_png = tf.image.encode_png(raw_data)
            encoded_png = sess.run(encoded_png,  feed_dict={raw_data: image_data})
            bytes_png = self._bytes_feature(sess, tf.compat.as_bytes(encoded_png))
        return bytes_png
        
    def _int64_feature(self, sess, value):
        """Wrapper for inserting int64 features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float16mat_feature(self, sess, matrix):
        """Wrapper for inserting Float32 features into Example proto."""
        if len(matrix.shape)> 3 or matrix.shape[2]>4:
            print("Float16 matrix is of dim %d and has %d channels while max 3 and 4 are allowed" % (len(matrix.shape), matrix.shape[3]))
        matrix = matrix.astype(np.float16)
        matrix.dtype = np.int16
        return self._png162feature(sess, matrix)
        
    def _float32_feature(self, sess, value):
        """Wrapper for inserting Float32 features into Example proto."""
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


    def _bytes_feature(self, sess, value):
        """Wrapper for inserting bytes features into Example proto."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        
    def _generic2feature(self, sess, string):
        return self._bytes_feature(sess, tf.compat.as_bytes(string))
        
        
class ProtoBufDeserializer:
    """ This class can be used to deserialize a database stored as protobuffer record files
        by decoding them to recover tensors containing the encoded information.
    """
            
    def _data_identity(data, thread_id = 0):
        """Dummy identity function which just outputs its input data"""
        return data
    
    def __init__(self, features_list, sess = None, data_dir = "serialized_DB", shard_prefix = "serialized_data_shard", samples_per_shard = 512):
        """ Args:
            features_list       :  list of features; each feature should be a list of string containing in this order : the name of the feature and its type
                                the different types available can be found in the dictionnaries in the code below
            data_dir            : the directory where the protobuffer records should be written
            shard_prefix        : the prefix for the protobuffer shard records
            samples_per_shard   : the number of samples to be written in each record shard
            sess                : the tensorflow session to use. If none is given, a new one will be created locally when needed
        """
        
        self.sess = sess
        
        self.name2deserializer_dict = {
            "int64"         : tf.io.FixedLenFeature([1], tf.int64,  -1),
            "float32"       : tf.io.FixedLenFeature([1], tf.float32,  -1),
            "int64_list"    : tf.io.VarLenFeature(dtype=tf.int64),
            "float32_list"  : tf.io.VarLenFeature(dtype=tf.float32),
            "string"        : tf.io.FixedLenFeature([ ], tf.string, ''),
            "jpeg"          : tf.io.FixedLenFeature([ ], tf.string, ''),
            "png"           : tf.io.FixedLenFeature([ ], tf.string, ''),
            "png16"         : tf.io.FixedLenFeature([ ], tf.string, ''),
            "float16_mat"   : tf.io.FixedLenFeature([ ], tf.string, ''),
            "bytes"         : tf.io.FixedLenFeature([ ], tf.string, ''),
            }        
        self.name2decoder_dict = {
            "int64" : self._data_identity2,
            "float32" : self._data_identity2,
            "int64_list" : self._decode_list,
            "float32_list" : self._decode_list,
            "string" : self._data_identity2,
            "jpeg" : tf.image.decode_jpeg,
            "png" : lambda img: tf.image.decode_png(img, dtype=tf.uint8),
            "png16" : lambda img: tf.image.decode_png(img, dtype=tf.uint16),
            "float16_mat": lambda img: tf.bitcast(tf.image.decode_png(img, dtype=tf.uint16), tf.float16),
            "bytes" : self._data_identity2,
            } 
        
        self.data_dir = data_dir
        self.features_list = features_list
        self.samples_per_shard = samples_per_shard
        self.shard_prefix = shard_prefix
        
        self.nrecord = self.get_num_records(os.path.join(self.data_dir, '%s-*' % self.shard_prefix))
            
        self.feature_map = {}
        for (name, type) in self.features_list:
            self.feature_map[name] = self.name2deserializer_dict[type]
    
    def deserialize_to_minibatch(self, batch_size, is_training, num_queues = 1, prepocessing_f = _data_identity, step = None):
        """"Reads TF records and extract randomly batched data shards which are pushed into a given number of queues
        
            Args:
            batch_size          : integer giving the desired batch size for all queues combined
            num_queues          : integer giving the number of desired queues (often the number of GPUs)
            prepocessing_f      : a function which can be called to preprocess each data shard by default data shards are not preprocessed
            
            Return:
            A list (len==num_queues) of lists (len==number of features for each data shard) of tensors with dim0 == queue_batch_size
        """     
        
        queue_batch_size = int(batch_size/num_queues)
        if queue_batch_size > self.samples_per_shard/2:
            queue_batch_size = int(self.samples_per_shard/2)
            print("Batch size reduced to %s per GPU to avoid potential deadlocks" % str(queue_batch_size))
        n_threads = min(os.cpu_count(), 8*num_queues)
        input_buffer_size     = min(n_threads*64, self.nrecord)
        
        total_batch_size = queue_batch_size*num_queues
        record_input = data_flow_ops.RecordInput(
                file_pattern= os.path.join(self.data_dir, '%s-*' % self.shard_prefix),
                parallelism= min(n_threads, self.nrecord),
                # Note: This causes deadlock during init if larger than dataset
                buffer_size= input_buffer_size,
                batch_size= total_batch_size)
        records = record_input.get_yield_op()
        
        # Split batch into individual images
        records = tf.split(records, total_batch_size, 0)
        records = [tf.reshape(record, []) for record in records]

        # Deserialize and preprocess images into batches for each device
        
        with tf.name_scope('input_pipeline'):
            for i, record in enumerate(records):
                single_example = self.deserialize_record(record)
                for j, (_, var_type) in enumerate(self.features_list):
                    single_example[j] = self.name2decoder_dict[var_type](single_example[j])
                single_training_example, additionnal_data = prepocessing_f(single_example, is_training, thread_id = i, step=step)
                
                # Init storage lists
                if i == 0:
                    nbre_features = len(single_training_example)
                    nbre_add_data = len(additionnal_data)
                    features_data_queues = [ []  for k in range(num_queues)]
                    additionnal_data_queues = [ []  for k in range(num_queues)]
                    for j in xrange(num_queues):
                        features_data_queues[j] = [ []  for k in range(nbre_features)]
                        additionnal_data_queues[j] = [ []  for k in range(nbre_add_data)]

                queue_num = i % num_queues
                for j in xrange(nbre_features):
                    features_data_queues[queue_num][j].append(single_training_example[j])
                for j in xrange(nbre_add_data):
                    additionnal_data_queues[queue_num][j].append(additionnal_data[j])
            
            # Stack images back into a sub-batch for each device
            for queue_num in xrange(num_queues):
                for i in xrange(nbre_features):
                    if features_data_queues[queue_num][i][0].get_shape() == [1]:
                        features_data_queues[queue_num][i] = tf.concat(features_data_queues[queue_num][i], axis=0)
                    else:
                        features_data_queues[queue_num][i] = tf.parallel_stack(features_data_queues[queue_num][i]) #tf.parallel_stack is faster but shape must beknown before execution

                for i in xrange(nbre_add_data):
                    if additionnal_data_queues[queue_num][i][0].get_shape() == [1]:
                        additionnal_data_queues[queue_num][i] = tf.concat(additionnal_data_queues[queue_num][i], axis=0)
                    else:
                        additionnal_data_queues[queue_num][i] = tf.parallel_stack(additionnal_data_queues[queue_num][i]) #tf.parallel_stack is faster but shape must beknown before execution

        return features_data_queues, additionnal_data_queues
        
    def get_num_records(self, tf_record_pattern, shard_count=None):
        """Returns the number of records which batch a given file pattern"""
        def count_records(tf_record_filename):
            count = 0
            for _ in tf.python_io.tf_record_iterator(tf_record_filename):
                count += 1
            return count

        filenames = sorted(tf.gfile.Glob(tf_record_pattern))
        nfile = len(filenames)
        nbre_rec= 0
        first_it = False

        if shard_count is None:
            print("Counting records...")
            shard_count = count_records(filenames[0])
            first_it = True

        if shard_count != count_records(filenames[nfile//2]) or nfile==1:
            for shard in filenames:
                nbre_rec += count_records(shard)
        else:
            nbre_rec = nfile//2*shard_count + self.get_num_records(filenames[nfile//2:], shard_count=shard_count)

        if first_it:
            print("%i records counted in dataset" % nbre_rec, flush=True)

        return nbre_rec
        #return (count_records(filenames[0])*(nfile-1) + count_records(filenames[-1]))
        
    ### "private" functions
        
    def deserialize_record(self, record):
        """ Constructs the dictionnary necessary for a record decoding according to the feature list provided"""
        
        feature_dict = {}
        for var_name, var_type in self.features_list:
            feature_dict[var_name] = self.name2deserializer_dict[var_type]
            
        with tf.name_scope('deserialize_image_record'):
            obj = tf.compat.v1.parse_single_example(record, self.feature_map)
            record_data = []
            for var_name, _ in self.features_list:
                record_data.append(obj[var_name])
        return record_data
        
    def _data_identity2(self, data):
        """Dummy identity function which just outputs its input data"""
        return data
        
    def _decode_list(self, obj_values):
        return tf.stack(obj_values.values)
        