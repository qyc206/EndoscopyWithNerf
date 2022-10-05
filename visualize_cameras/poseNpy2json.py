import os 
import numpy as np 
import json 

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

# this function is borrowed from nerf_pl/datasets/llff.py
def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg

# this function is borrowed from nerf_pl/datasets/llff.py, which is used to re-center the cameras
def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)

# construct intrinsic matrix from image height,width and focal length 
def create_Kmatrix(H, W, focal):
    K = np.array([
            [focal, 0, 0.5*W, 0],
            [0, focal, 0.5*H, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    return K 


def main(recenter_scale=False):
    # the path for camera info file 
    data_file = 'poses_bounds.npy'
    # load it in array
    data = np.load(data_file)
    # check the dimensions
    print(data.shape)


    poses = data[:, :15].reshape(-1, 3, 5) # (N_images, 3, 5) # poses for cameras given in the data file 
    bounds = data[:, -2:] # (N_images, 2) # near and far given in the data file
    print('original bounds are', bounds)
    #bounds[:, 0] = 0.75
    print('now bounds are ', bounds)
    H, W, focal = poses[0, :, -1] # get camera intrinsics, same for all images  
    print('height and width: ', H, W)
    focal = 680

    # construct K matrix for all cameras 
    Kmat = create_Kmatrix(H, W, focal )
    K_list = list(Kmat.flatten())
    K_list = [float(item) for item in K_list]

    #Original poses has rotation in form "down right back", change to "right up back"
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    
    if recenter_scale:
        '''
        Begin the re-center process of the camera poses. (which is used in nerf_pl/datasets/llff.py for preprocessing the data before training)
        '''
        # to see how the camera poses change after re-normalization (the operation in nerf_pl, llff.py)
        poses, pose_avg = center_poses(poses) 
        distances_from_center = np.linalg.norm(poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center) # choose val image as the closest to
                                                    # center image

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = bounds.min() 
        print('near_original is ', near_original)
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                            # the nearest depth is at 1/0.75=1.33
        bounds /= scale_factor
        print('bounds is', bounds)
        poses[..., 3] /= scale_factor

        '''
        end of the re-center process
        '''
    else:
        pass 

    all_camera_info = {} # a container for storing the camera infos, which will be saved into the final json file 
    # get camera2world matrix for each camera 

    for idx in range(data.shape[0]):
        c2w_mat = poses[idx, ...]
        last_row = np.array([0, 0, 0, 1]).reshape((1, 4))

        c2w_mat = np.concatenate([c2w_mat, last_row], axis=0)

        print(c2w_mat.shape)
        w2c_mat = np.linalg.inv(c2w_mat)
        w2c_list = list(w2c_mat.flatten())
        w2c_list = [float(item) for item in w2c_list]

        cur_item = {'K': K_list, 'W2C': w2c_list, 'img_size':[H, W]}
        all_camera_info[f'{idx:03d}'] = cur_item  

    
    if recenter_scale:
        camera_file = 'stomach_camera_recenter.json' 
    else:
        camera_file = 'stomach_camera_original.json'

    with open(camera_file, 'w') as f: 
        json.dump(all_camera_info, f)




if __name__ == '__main__':
    main()   # create a file for original cameras from poses_bounds.npy
    main(True) # create a file for recen-tered cameras from poses_bounds.npy (the re-centering process is used in nerf_pl for data preprocessing)
