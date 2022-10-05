## NOTE: parts of this code are inspired by Yujie's codes

import os
#from scipy.spatial.transform import Rotation as R

import json
import bpy
import math
import numpy as np


'''

SET UP PORTION

'''

CONVERT2LST = True

CAMERA_INFO = []    # if a file is not provided, replace this with camera info

# update camera (i.e. flip upside_down & shift up via z-axis)
UPDATE_CAMERA = False

# FRAMES = 20
FRAMES = 70

resolution_x = 640
resolution_y = 480
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'

Nrows = 5
Ncols = 6

RESULTS_PATH = '/Downloads/frames2_wball_llff'
FRAMES_PATH = '/Downloads/frames2_wball_llff'

# create excel with camera information
CREATE_EXCEL = False
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# create nodes for obtaining the renderings for each componen.
bpy.context.view_layer.cycles.denoising_store_passes = True
render_layers = tree.nodes.new('CompositorNodeRLayers')
image_file_output = tree.nodes.new(type='CompositorNodeOutputFile')
image_file_output.label = "Image_Output"
image_file_output.base_path = RESULTS_PATH

denoise_node = tree.nodes.new(type='CompositorNodeDenoise')
links.new(render_layers.outputs['Noisy Image'], denoise_node.inputs[0])
links.new(render_layers.outputs['Denoising Normal'], denoise_node.inputs[1])
links.new(render_layers.outputs['Denoising Albedo'], denoise_node.inputs[2])

viewer_node = tree.nodes.new('CompositorNodeViewer')   
viewer_node.use_alpha = True
links.new(denoise_node.outputs['Image'], viewer_node.inputs['Image'])
links.new(render_layers.outputs['Depth'], viewer_node.inputs['Alpha'])

links.new(denoise_node.outputs['Image'], image_file_output.inputs['Image'])

# finish creating the nodes


############ Functions ######################################

def parent_obj_to_camera(b_camera, origin):
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

#def readXlsxInfo2Lst():
#    workbook = opx.load_workbook(XLSX_PATH)
#    sheet = workbook.active

#    for row in sheet.iter_rows(min_row=2, max_row=FRAMES+1):
#        # [trans_x, trans_y, trans_z, quot_x, quot_y, quot_z, quot_w]
#        CAMERA_INFO.append([row[3].value, row[4].value, row[5].value, row[6].value, row[7].value, row[8].value, row[9].value])

#    workbook.close()

#if CONVERT2LST:
#    readXlsxInfo2Lst()

def updateCamera(deltaDeg, deltaZ):
    for i in range(len(CAMERA_INFO)):
        cameraInfo = CAMERA_INFO[i]
        rEuler = R.from_quat([cameraInfo[3], cameraInfo[4], cameraInfo[5], cameraInfo[6]]).as_euler('xyz', degrees=True)
        rFlipped = R.from_euler('xyz', [rEuler[0], rEuler[1]+deltaDeg, rEuler[2]], degrees=True).as_quat()

        newCameraInfo = CAMERA_INFO[i][:2]
        newCameraInfo.extend([CAMERA_INFO[i][2] + deltaZ])
        newCameraInfo.extend(rFlipped)
        CAMERA_INFO[i] = newCameraInfo

if UPDATE_CAMERA:
    updateCamera(180, 37)


def saveCameraInfo2Excel():
    workbook = opx.Workbook()
    sheet = workbook.active

    sheet.append(['', 'ImageFrame', 'Pose_Index', 'trans_x', 'trans_y', 'trans_z', 'quot_x', 'quot_y', 'quot_z', 'quot_w'])

    for i in range(len(CAMERA_INFO)):
        row = [i, 0, 0]
        row.extend(CAMERA_INFO[i])
        sheet.append(row)

    workbook.save(RESULTS_PATH+"/custom_frames_camera_info.xlsx")

if CREATE_EXCEL:
    saveCameraInfo2Excel()
  
    


'''

RENDERING PORTION

'''

# set up camera
cam = scene = bpy.context.scene.objects['Camera']
#cam = bpy.data.objects['Camera']
#camera.scale = (10.0, 10.0, 10.0)
#camera.data.lens_unit = 'FOV'
#camera.data.angle = math.radians(26.20)
#camera.rotation_mode = 'QUATERNION'

#cam = scene.objects['Camera']
#cam = scene.objects['Camera']
cam.location = (-0.5, -0.5, 12.0)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
#b_empty = parent_obj_to_camera(cam, origin=(-0.34, 0.08, 7.67))
b_empty = parent_obj_to_camera(cam, origin=(0.0, 0.0, 0.0))
cam_constraint.target = b_empty
cam.rotation_euler = (0,0,0)



# data to store in JSON file
out_data = {
    #'camera_scale': camera.scale,
    'camera_fov_radians': bpy.data.objects['Camera'].data.angle_x,
    'camera_rotation_mode': bpy.data.objects['Camera'].rotation_mode,
}

# render optimizations
bpy.context.scene.render.use_persistent_data = True

# run the function used for creating input/output nodes for grayscale image and depth image
#create_nodes() 

# background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# set up scene
scene = bpy.context.scene
scene.render.resolution_x = resolution_x    # width
scene.render.resolution_y = resolution_y    # height
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = FORMAT



out_data['frames'] = []

# set up path to store rendered frames
fp = bpy.path.abspath(f"//{FRAMES_PATH}")
if not os.path.exists(fp):
    os.makedirs(fp)

x_arr = np.linspace(-0.3, 0.3, Nrows)
y_arr = np.linspace(-0.3, 0.3, Ncols)
z_arr = np.linspace(8.2, 8.4, Ncols)


for row_idx in range(Nrows):
    for col_idx in range(Ncols):
        
        i = row_idx * Ncols + col_idx # calculate the frame_idx
        
        cur_y = y_arr[col_idx]
        if row_idx %2 == 0:
            cur_x = x_arr[row_idx]
            cur_z = z_arr[col_idx]
        else:
            cur_x = x_arr[row_idx] - col_idx * (x_arr[1]- x_arr[0])
            cur_z = z_arr[col_idx]
        
        
        cam.location = (cur_x, cur_y, cur_z)
    #    cam.rotation_quaternion = (CAMERA_INFO[i][6], CAMERA_INFO[i][3], CAMERA_INFO[i][4], CAMERA_INFO[i][5]) # (w,x,y,z)

        print("Rendering Frame {}...".format(i))
        #print("\tRotation in quaternion (x,y,z,w):\n\t{}\n".format(CAMERA_INFO[i][3:]))
        scene.render.filepath = fp + '/r_{0:03d}'.format(i)
        bpy.ops.render.render(write_still=True)  # render still
        
        
        #render original depth values to npy file.
        
        pixels = np.array(bpy.data.images['Viewer Node'].pixels)
        width = bpy.context.scene.render.resolution_x 
        height = bpy.context.scene.render.resolution_y
        
        #reshaping into image array 4 channel (rgbz)
        image = pixels.reshape(height,width,4)
        
        #depth analysis...
        depth_array = np.asarray(image[:,:,3])
        print(np.max(depth_array), np.min(depth_array))
        
        print(depth_array.shape)
        #np_depth_path = dir_cur + dir_dataset + str(s_iter).zfill(6)+"_"+ str(f_iter).zfill(2) + 'Depth.npy'
        np_depth_path = scene.render.filepath + '_depth92.npz'
        depth_array = np.array(depth_array[::-1, ...])
        np.savez_compressed(np_depth_path, depth=depth_array)
        
        
        #saving the camera parameters
        quat = cam.rotation_quaternion
        print(quat)

        frame_data = {
            'file_path': scene.render.filepath,
            'transform_matrix': listify_matrix(cam.matrix_world)
        }
    

        out_data['frames'].append(frame_data)

with open(fp + '/' + 'transforms.json', 'w') as out_file:
         json.dump(out_data, out_file, indent=4)
