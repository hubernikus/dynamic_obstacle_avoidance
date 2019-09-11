# /usr/bin/python2
# This script is in python2

# import numpy as np
# import matplotlib.pyplot as plt

import rosbag
import json

def main():
    print("Start it")
    bag = rosbag.Bag('../data/laser_lidar_recording.bag')

    print("Topics names are") 
    print(bag.get_type_and_topic_info()[1].keys())
    
    data = []
    ii_max = 2

    ii = 0
    
    for topic, msg, t in bag.read_messages(topics=['/front/scan', '/rear/scan']):
        data.append({})
        data[ii]["name"] = topic
        data[ii]["angle_min"] = msg.angle_min
        data[ii]["angle_max"] = msg.angle_max
        data[ii]["increment"] = msg.angle_increment
        data[ii]["ranges"] = msg.ranges

        print("Got scan ", topic)
        ii += 1
        
        if ii >= ii_max:
            break

    with open('../data/laser_lidar_recording.txt', 'w') as outfile:  
        json.dump(data, outfile)

    bag.close()

    print("Done it")

# if __name__==("__main__"):
if True:
    main()
