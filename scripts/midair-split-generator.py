'''
Note: This script requires two specific libraries:
    * h5py for opening Mid-Air data records
    * pyquaternion for quaternion operations

Both can be installed with pip:
$ pip install pyquaternion h5py
'''

import os
import argparse
import h5py
from pyquaternion import Quaternion
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--db_path", default=os.path.join(*[dir_path,"..", "datasets","MidAir"]), help="path to folder containing the databases")
parser.add_argument("--output_dir", default=os.path.join(*[dir_path,"..", "data", "midair"]), help="path to folder to store csv files")
a = parser.parse_args()

FRAME_SKIP = 4 # Downsample framerate

if __name__== "__main__":

    os.makedirs(a.output_dir, exist_ok=True)

    data = ["Kite_training", "PLE_training"]
    sensors = [["color_left", ".JPEG"], ["stereo_disparity", ".PNG"]]

    listOfFile = []
    for env in data:
        listOfFile += [os.listdir(os.path.join(a.db_path, env))]

    for set in data:
        climates = os.listdir(os.path.join(a.db_path,set))
        for climate in climates:
            print("Processing %s %s" % (set, climate))

            trajectories = os.listdir(os.path.join(*[a.db_path, set, climate, sensors[0][0]]))
            h5_db = h5py.File(os.path.join(*[a.db_path, set, climate, "sensor_records.hdf5"]), 'r')
            for traj_nbre, (traj) in enumerate(trajectories):

                # Assign one-on-three trajectories to the test set
                if traj_nbre % 3 !=0:
                    out_dir = os.path.join(*[a.output_dir, "train_data", set, climate])
                else:
                    out_dir = os.path.join(*[a.output_dir, "test_data", set, climate])

                os.makedirs(out_dir, exist_ok=True)
                file_name = os.path.join(out_dir, "traj_%s.csv" % str(traj_nbre).zfill(4))

                # Create csv file
                with open(file_name, 'w') as file:
                    file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ("id", "camera_l", "disp", "qw", "qx", "qy", "qz", "tx", "ty", "tz"))
                    def get_path(sensor, index, ext):
                        im_name = str(index).zfill(6) + "." + ext
                        path = os.path.join(*[set, climate, sensor, traj, im_name])
                        return path

                    # Reminder: 4 IMU measurements are made between 2 camera frames
                    r_a = np.array(h5_db[traj]["groundtruth"]["attitude"][:-(4*FRAME_SKIP),:])
                    r_b = np.array(h5_db[traj]["groundtruth"]["attitude"][(4*FRAME_SKIP):, :])

                    p_a = np.array(h5_db[traj]["groundtruth"]["position"][:-(4*FRAME_SKIP),:])
                    p_b = np.array(h5_db[traj]["groundtruth"]["position"][(4*FRAME_SKIP):, :])

                    traj_len = r_a.shape[0]//(FRAME_SKIP*4)

                    seq_cam = []
                    seq_disp = []
                    seq_rot = []
                    seq_trans = []
                    seq_start = []

                    # Iterate over sequence samples
                    for index in range(traj_len):

                        # Compute frame-to-frame camera motion
                        i = index*FRAME_SKIP
                        q_r_a = Quaternion(r_a[i*4,:])
                        q_r_b = Quaternion(r_b[i*4,:])
                        trans = q_r_a.conjugate.rotate(p_b[i*4,:] - p_a[i*4,:])
                        rot = (q_r_a.conjugate * q_r_b).elements

                        camera_l = get_path("color_left", i+FRAME_SKIP, "JPEG")
                        stereo_disp = get_path("stereo_disparity", i+FRAME_SKIP, "PNG")

                        # change referential from body to camera
                        rot = rot.tolist()
                        rot = [rot[0], rot[2], rot[3], rot[1]]
                        trans = [trans.tolist()[1], trans.tolist()[2], trans.tolist()[0]]

                        # Write sample to file
                        file.write("%i\t%s\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (index, camera_l, stereo_disp, rot[0], rot[1], rot[2], rot[3], trans[0], trans[1], trans[2]))
