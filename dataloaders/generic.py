import tensorflow as tf
import os
import glob
import json
import pandas as pd
from collections import namedtuple

DataloaderParameters = namedtuple('DataloaderParameters', ('db_path_config', 'records_path', 'db_seq_len', 'seq_len', 'augment' ))

class DataLoaderGeneric():
    """Superclass for other dataset dataloaders
    """
    def __init__(self, dataset_name):
        self.build_functions = {"train" : self._build_train_dataset,
                                "finetune" : self._build_train_dataset,
                                "eval"  : self._build_eval_dataset,
                                "predict": self._build_eval_dataset
                                }
        self.augment = None
        self.settings = None
        self.db_name = dataset_name
        
    def _decode_samples(self, data_sample):
        ''' Creates a sample to be fed to the network from a line of the dataset csv files
            Receives a dict whose keys correspond to the columns of the csv files
            Shal output a dict with the following keys and data:
                - camera: a dict containing the camera intrinsic parameters for the sample {f: [fx, fy], c: [cx,cy]}
                - depth : [optional] the ground truth depth map
                - RGB_im: the color image used to estimate depth
                - rot   : the quaternion expressing the rotation from the previous frame to the current one
                - trans : the translation vector expresing the displacement of the camera from the previous frame to the current one
                - new_traj: a boolean telling if the image is the first of a sequence
        '''
        return NotImplementedError

    def _perform_augmentation(self):
        ''' Process self.out_data to get the desired data augmentation
            shape of image data == [b, seq_len, out_h, out_w, c]
            shape of vector data == [b, seq_len, c] 
        '''
        return NotImplementedError

    def _set_output_size(self, out_size=None):
        ''' Shall set the variable 'self.out_size'. Can be used to adapt other variables that depend on this parameter.
            Is called automatically  when building a dataset.
            out_size = [height, width]
        '''
        return NotImplementedError

    def get_dataset(self, usecase, settings, batch_size=3, out_size=None): # usecase, db_seq_len=None, seq_len=None, batch_size=3):
        ''' Builds a tensorflow dataset using provided parameters
            * usecase : the mode in which the dataset will be used (train, eval, predict,...)
            * db_seq_len: [int] if provided, the input data will be cut in subtrajectories of the given length
            * seq_len : the sequence length to be passed to the network (must be <= db_seq_len)
        '''
        if out_size is None:
            self._set_output_size()
        else:
            self._set_output_size(out_size=out_size)

        self.settings = settings
        self.records_path = settings.records_path
        self.db_path = settings.db_path_config[self.db_name]
        self.db_seq_len = self.settings.db_seq_len
        self.seq_len = self.settings.seq_len
        self.batch_size = batch_size
        self.usecase = usecase
        
        if usecase == "train" and (self.db_seq_len is None or self.seq_len is None):
            raise Exception('db_seq_len and seq_len must be defined in train mode')
        
        if not (self.db_seq_len is None or self.seq_len is None) and self.db_seq_len < self.seq_len:
            raise Exception('db_seq_len must be larger or equal than seq_len')

        try:
            function = self.build_functions[usecase]
        except:
            raise Exception('Usecase "%s" not implemented for this dataloader' % usecase)
        self.dataset = function()
        self.length = self.dataset.cardinality().numpy()

        return self.dataset

    def _get_trajectories(self):
        csv_files = glob.glob(os.path.join(self.records_path, "**/*.csv"), recursive=True)
        trajectories= []
        for file in csv_files:
            pd_dataset = pd.read_csv(file, sep="\t")
            traj_dataset = tf.data.Dataset.from_tensor_slices(dict(pd_dataset)) #\
                # .map(self._decode_samples, num_parallel_calls=tf.data.AUTOTUNE)

            trajectories.append(traj_dataset)
        if trajectories == []:
            raise Exception("No csv files found at the given path: %s" % self.records_path)
        else:
            return trajectories

    def _build_train_dataset(self):
        self.augment = self.settings.augment
        self.new_traj = tf.convert_to_tensor([i == 0 for i in range(self.seq_len)], dtype=tf.bool)
        trajectories = self._get_trajectories()
        dataset = None
        for traj in trajectories:
            traj = traj.batch(self.db_seq_len, drop_remainder=True)\
                        .map(self._cut_sequence, num_parallel_calls=tf.data.AUTOTUNE)
            if dataset is None:
                dataset = traj
            else:
                dataset = dataset.concatenate(traj)

        # We shuffle samples while they all still hold in memory (i.e. before loading pictures) to be able to shuffle
        # the whole dataset. This requires an additional unbatch/batch operation
        dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=True) \
                        .unbatch().map(self._decode_samples, num_parallel_calls=tf.data.AUTOTUNE) \
                        .batch(self.seq_len, drop_remainder=True) \
                        .map(self._build_sequence_samples, num_parallel_calls=tf.data.AUTOTUNE) \
                        .batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        return dataset

    def _build_eval_dataset(self):
        self.augment = False

        if not self.db_seq_len is None:
            print("Evaluating on subsequences of length %i" % self.db_seq_len)
            self.seq_len = self.db_seq_len
        trajectories = self._get_trajectories()
        dataset = None
        for traj in trajectories:
            traj = traj.map(self._decode_samples, num_parallel_calls=tf.data.AUTOTUNE)
            if self.db_seq_len is not None:
                traj = traj.batch(self.db_seq_len, drop_remainder=True)
            if dataset is None:
                dataset = traj
            else:
                dataset = dataset.concatenate(traj)

        if self.db_seq_len is not None:
            self.new_traj = tf.convert_to_tensor([i == 0 for i in range(self.seq_len)], dtype=tf.bool)
            dataset = dataset.map(self._build_sequence_samples, num_parallel_calls=tf.data.AUTOTUNE) \
                            .batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        else:
            dataset = dataset.batch(1, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

        return dataset

    @tf.function
    def _cut_sequence(self, data_sample):
        ''' Cuts a sequence of samples (len==db_se_len) to the desired length (seq_len).
        '''

        out_data = {}
        offset = tf.random.uniform(shape=[], minval=0, maxval=self.db_seq_len - self.seq_len + 1, dtype=tf.int32)
        for key, tensor in data_sample.items():
            out_data[key] = tf.slice(tensor, [offset], [self.seq_len])
        return out_data


    @tf.function
    def _build_sequence_samples(self, data_sample):
        ''' Builds a sequence of samples and performs data augmentation
            on the resulting sequence if self.augment is set to true
        '''

        im_color = data_sample["RGB_im"]
        im_depth = data_sample["depth"]
        rot = data_sample["rot"]
        pos = data_sample["trans"]
        l = rot.get_shape().as_list()[-1]

        camera_data = {
            "f": data_sample["camera"]['f'][0, :],
            "c": data_sample["camera"]['c'][0, :]
        }
        self.out_data = {}
        self.out_data["camera"] = camera_data.copy()
        self.out_data["depth"] = im_depth
        self.out_data["RGB_im"] = im_color
        self.out_data["rot"] = rot
        self.out_data["trans"] = pos
        self.out_data["new_traj"] = self.new_traj
        
        # Perform data augmentation if necessary
        if self.augment:
            self._perform_augmentation()

        return self.out_data.copy()

    def _augmentation_step_color(self, invert_color=True):
        ''' Perform data augmentation on the colors '''

        if self.usecase == "finetune":
            self.out_data["RGB_im"] = tf.image.random_brightness(self.out_data["RGB_im"], 0.2)
            self.out_data["RGB_im"] = tf.image.random_contrast(self.out_data["RGB_im"], 0.8, 1.2)
            self.out_data["RGB_im"] = tf.image.random_saturation(self.out_data["RGB_im"], 0.8, 1.2)
            self.out_data["RGB_im"] = tf.image.random_hue(self.out_data["RGB_im"], 0.2)
        else:
            self.out_data["RGB_im"] = tf.image.random_brightness(self.out_data["RGB_im"], 0.2)
            self.out_data["RGB_im"] = tf.image.random_contrast(self.out_data["RGB_im"], 0.75, 1.25)
            self.out_data["RGB_im"] = tf.image.random_saturation(self.out_data["RGB_im"], 0.75, 1.25)
            self.out_data["RGB_im"] = tf.image.random_hue(self.out_data["RGB_im"], 0.4)

        
        if invert_color:
            def do_nothing():
                return self.out_data["RGB_im"]
            def true_inv_col():
                return 1. - self.out_data["RGB_im"]

            p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            pred = tf.less(p_order, 0.5)
            self.out_data["RGB_im"] = tf.cond(pred, true_inv_col, do_nothing)

    # @tf.function
    def _augmentation_step_flip(self):
        ''' Perform data augmentation on the orientation of the images
            WARNING : works only with quaternion rotations
        '''
        
        im_col = self.out_data["RGB_im"]
        im_depth = self.out_data["depth"]
        rot = self.out_data["rot"]
        trans = self.out_data["trans"]
        c = self.out_data["camera"]["c"]

        h,w = im_col.get_shape().as_list()[1:3]

        def do_nothing():
            return [im_col, im_depth, rot, trans, c]

        def true_flip_v():
            col = tf.reverse(im_col, axis=[1])
            dep = tf.reverse(im_depth, axis=[1])
            r = tf.multiply(rot, [[1., -1., 1., -1.]])
            t = tf.multiply(trans, [[1., -1., 1.]])
            c_ = tf.convert_to_tensor([c[0], h-c[1]])
            return [col, dep, r, t, c_]

        p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        pred = tf.less(p_order, 0.5)
        im_col, im_depth, rot, trans, c = tf.cond(pred, true_flip_v, do_nothing)

        def true_flip_h():
            col = tf.reverse(im_col, axis=[2])
            dep = tf.reverse(im_depth, axis=[2])
            r = tf.multiply(rot, [[1., 1., -1., -1.]])
            t = tf.multiply(trans, [[-1., 1., 1.]])
            c_ = tf.convert_to_tensor([w-c[0], c[1]])
            return [col, dep, r, t, c_]

        p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
        pred = tf.less(p_order, 0.5)
        im_col, im_depth, rot, trans, c = tf.cond(pred, true_flip_h, do_nothing)

        self.out_data["camera"]["c"] = c
        self.out_data["depth"] = im_depth
        self.out_data["RGB_im"] = im_col
        self.out_data["rot"] = rot
        self.out_data["trans"] = trans
