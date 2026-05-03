#!/bin/bash

roslaunch franka_example_controllers position_force_example.launch robot_ip:=172.16.0.3
# roslaunch orbbec_camera gemini2_force.launch