# Script for encoding Mid-Air in Protobuffer files for M4Depth
#
# Author : Michael Fonder 2021
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from protobuf_db import *
import argparse
import h5py
from pyquaternion import Quaternion
import math

parser = argparse.ArgumentParser()
parser.add_argument("--db_path", default="/media/michael/Database/Kiwi/generated_data", help="path to folder containing the databases")
parser.add_argument("--output_dir", help="path to folder to store images")
parser.add_argument("--write", dest="write", action="store_true", help="choose wether to write files or not")
a = parser.parse_args()

if __name__== "__main__":
    SEQ_LEN = 8
    FRAME_SKIP = 4
    SEQ_GAP = FRAME_SKIP #70-(SEQ_LEN*FRAME_SKIP)

    feature_list = []
    for i in range(SEQ_LEN):
        feature_list.append(['image/color_'+str(i).zfill(int(math.log10(SEQ_LEN)+1)),  'jpeg'])
        feature_list.append(['image/depth_'+str(i).zfill(int(math.log10(SEQ_LEN)+1)),  'png16'])
        feature_list.append(['data/omega_'+str(i).zfill(int(math.log10(SEQ_LEN)+1)), 'float32_list'])
        feature_list.append(['data/trans_'+str(i).zfill(int(math.log10(SEQ_LEN)+1)),  'float32_list'])

    train_dir = os.path.join(a.output_dir, "train")
    test_dir = os.path.join(a.output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_serializer = ProtoBufSerializer(feature_list, data_dir = train_dir, samples_per_shard = 64, nbre_threads=2)
    test_serializer = ProtoBufSerializer(feature_list, data_dir = test_dir, samples_per_shard = 64, nbre_threads=2)

    data = ["Kite_training", "PLE_training"]
    sensors = [["color_left", ".JPEG"], ["stereo_disparity", ".PNG"]]

    listOfFile = []
    for env in data:
        listOfFile += [os.listdir(os.path.join(a.db_path,env))]
    print(listOfFile)

    for set in data:
        climates = os.listdir(os.path.join(a.db_path,set))
        for climate in climates:
            print("Processing %s %s" % (set, climate))
            trajectories = os.listdir(os.path.join(*[a.db_path,set,climate, sensors[0][0]]))
            h5_db = h5py.File(os.path.join(*[a.db_path,set, climate, "sensor_records.hdf5"]), 'r')
            for traj_nbre, (traj) in enumerate(trajectories):
                for i in range(0,68):
                    def get_path(sensor, index, ext):
                        im_name = str(index).zfill(6) + "." + ext
                        path = os.path.join(*[a.db_path, set, climate, sensor, traj, im_name])
                        return path

                    db_sample = []
                    base_offset = FRAME_SKIP + i*(SEQ_LEN*FRAME_SKIP)
                    for k in range(SEQ_LEN):
                        ki = base_offset + k*FRAME_SKIP

                        r_a = Quaternion(np.array(h5_db[traj]["groundtruth"]["attitude"][(ki-FRAME_SKIP)*4,:]))
                        r_b = Quaternion(np.array(h5_db[traj]["groundtruth"]["attitude"][(ki)*4,:]))

                        p_a = np.array(h5_db[traj]["groundtruth"]["position"][(ki-FRAME_SKIP) * 4, :])
                        p_b = np.array(h5_db[traj]["groundtruth"]["position"][(ki) * 4, :])

                        trans = r_a.conjugate.rotate(p_b-p_a)
                        rot = (r_a.conjugate*r_b).elements[1:]

                        db_sample.append(get_path("color_left", ki, "JPEG"))
                        db_sample.append(get_path("stereo_disparity", ki, "PNG"))
                        db_sample.append([rot.tolist()[1]*2., rot.tolist()[2]*2., rot.tolist()[0]*2.])
                        db_sample.append([trans.tolist()[1], trans.tolist()[2], trans.tolist()[0]])

                    if a.write and traj_nbre % 3 != 0:
                        train_serializer.process_sample(db_sample)
                    if a.write and traj_nbre % 3 == 0:
                        test_serializer.process_sample(db_sample)

            h5_db.close()
        train_serializer.end_serializing()  # This line is mandatory to ensure a correct ending of the encoding process
        test_serializer.end_serializing()  # This line is mandatory to ensure a correct ending of the encoding process
