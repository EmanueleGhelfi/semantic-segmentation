import numpy as np
from unrealcv import *
import json
from commands import *
import sys, os, io
import time
from image_helper import *
conf = open("conf.json").read()
conf = json.loads(conf)

y_min = -200
y_max = 400
x_min = -100
x_max = 100
z_min = -50
z_max = 200

DATASET_SIZE = 5000
PIXEL_THRESHOLD = 100
DEPTH_PATH = os.getcwd()+"/dataset/depth/"
ANNOTATION_PATH = os.getcwd()+"/dataset/label/"
NORMAL_PATH = os.getcwd()+"/dataset/rgb/"

port = 9014
host ='127.0.0.1'

def setup_category_color(client):
    # get all ids from server
    res = client.request(REQUEST_OBJECTS)
    ids = res.split(" ")
    for id in ids:
        r = 0
        g = 0
        b = 0

        # if it is an object to detect then set its color
        if id in conf['objects_cat'].keys():
            r = conf['cat_to_col'][conf['objects_cat'][id]]['r']*255
            g = conf['cat_to_col'][conf['objects_cat'][id]]['g']*255
            b = conf['cat_to_col'][conf['objects_cat'][id]]['b']*255

        client.request(SET_OBJECT+f"{id}/color {r} {g} {b}")

        # check result
        res = client.request(f'vget /object/{id}/color')
        print(f"Set color of object {id} to {res}")


def main():
    # connect to server
    client = Client((host, port))
    client.connect()
    if not client.isconnected():
        print('UnrealCV server is not running. Run the game downloaded from http://unrealcv.github.io first.')
        sys.exit(-1)
    print("Client connected")
    time.sleep(1)

    setup_category_color(client)

    init_pos = client.request("vget /camera/0/location").split(" ")
    z = init_pos[2]
    init_rot = client.request("vget /camera/0/rotation").split(" ")
    pitch = init_rot[0]
    roll = init_rot[2]
    # save image
    for i in range(DATASET_SIZE):
        object_found = False
        while not object_found:
            x = np.random.rand()*(x_max-x_min) + x_min
            y = np.random.rand()*(y_max-y_min) + y_min
            yaw = np.random.rand()*359
            print(f"camera position {x},{y},{z},{roll},{pitch},{yaw}")
            # change camera position
            res = client.request(f"vset /camera/0/location {x} {y} {z}")

            # change camera rotation
            res = client.request(f"vset /camera/0/rotation {pitch} {yaw} {roll}")

            # get camera location
            loc = client.request("vget /camera/0/location")
            rot = client.request("vget /camera/0/rotation")
            print("Camera location ", loc)
            print("Camera rotation ", rot)
            
            obj = client.request(f'vget /camera/0/object_mask png', timeout=60)
            obj_np = read_png(obj)

            # an object is found if there are at least 100 pixel different from 0
            object_found = (obj_np[:,:,:-1] != np.reshape(np.array([0,0,0]),(1,1,3))).sum() > PIXEL_THRESHOLD

            print("Object found: ", object_found)
            if object_found:

                img = client.request(f'vget /camera/0/lit png', timeout=60)
                depth = client.request(f'vget /camera/0/depth npy', timeout=60)
                depth = read_npy(depth)

                # save to file
                with open(NORMAL_PATH+f'image{i}.png','wb') as f:
                    f.write(img)

                with open(ANNOTATION_PATH+f'image{i}.png','wb') as f:
                    f.write(obj)

                np.save(DEPTH_PATH+f'image{i}.npy', depth)         
            
    pass



if __name__=="__main__":
    print(conf)
    main()