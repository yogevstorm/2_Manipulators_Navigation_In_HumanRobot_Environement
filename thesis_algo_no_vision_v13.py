#!/usr/bin/env python3.6
# license removed for brevity
import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud
from std_msgs.msg import String
from std_msgs.msg import Int8
from std_msgs.msg import Float64
from manipulator_h_base_module_msgs.msg import JointPose
from sensor_msgs.msg import Joy
import heapq
from visualization_msgs.msg import MarkerArray
import time
from geometry_msgs.msg import Point
from yolo.msg import array_of_arrays
import random
import math
import copy
import numpy as np
#import cupy as np
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from std_srvs.srv import Empty
from std_msgs.msg import Float32MultiArray
from robot_markers.msg import path_msg
from robotis_controller_msgs.msg import SyncWriteItem
from std_msgs.msg import MultiArrayDimension
from scipy.spatial import distance
from threading import Thread, enumerate
from queue import Queue


#np.random.seed(1)

############ main algorithm class ########################

class Algorithm:


    def __init__(self):

        ############## Publishers ######################

        self.pub_visualization = rospy.Publisher('Algorithm_visualization',MarkerArray, queue_size=1)

        self.pub_set_mode = rospy.Publisher('/robotis/base/set_mode_msg', String, queue_size=10)

        self.pub_set_joints_arm1 = rospy.Publisher('/robotis/base/joint_pose_msg',JointPose, queue_size=10)

        self.pub_set_joints_arm2 = rospy.Publisher('/robotis/base/joint_pose_msg_arm2',JointPose, queue_size=10)

        #/rh_p12_rn_a_h1/rh_p12_rn_a_position/command

        self.pub_set_gripper_arm1 = rospy.Publisher('/rh_p12_rn_a_h1/rh_p12_rn_a_position/command',Float64, queue_size=10)

        self.pub_set_gripper_arm2 = rospy.Publisher('/rh_p12_rn_a_h2/rh_p12_rn_a_position/command',Float64, queue_size=10)

        self.pub_trajectory_state = rospy.Publisher('trajectory_state',path_msg, queue_size=1)

        self.pub_pointcloud_after_filter = rospy.Publisher('pointcloud_after_filter',PointCloud, queue_size=1)

        ############## Subscribers ######################

        self.joints_state_arm1_sub = rospy.Subscriber("/robotis/present_joint_states", JointState, self.joint_state_callback)

        self.joints_state_arm2_sub = rospy.Subscriber("/robotis/present_joint_states_arm2", JointState, self.joint_state_arm2_callback)

        self.occupied_cells_sub = rospy.Subscriber("/occupied_cells_vis_array", MarkerArray, self.occupied_cells_callback)

        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)

        self.objects_locations_sub = rospy.Subscriber("objects_locations",array_of_arrays,self.objects_locations_callback)

        self.pointcloud_sub = rospy.Subscriber("velodyne_pcl",PointCloud,self.velodyne_points_callback)

        self.pointcloud_sub_after_filter = rospy.Subscriber("pcl_after_filter",PointCloud,self.pcl_after_filter_callback)

        ############## Variables definitions ######################

        self.joint_state_data , self.joint_state_data_arm2 = JointState() , JointState()

        self.joint_state_data.position , self.joint_state_data_arm2.position = np.zeros(6) ,np.zeros(6)

        self.occupied_cells=MarkerArray()

        self.X_button,self.start_button,self.back_button = 0 ,0 ,0

        self.objects_locations=array_of_arrays()

        self.set_joints_msg=JointPose()

        self.set_joints_msg.name=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

        self.min_dis_obstacle_from_manipulator=0.13

        self.min_dis_obstacle_from_objects=0.0

        self.calibrate_lidar=False

        if(self.calibrate_lidar):

            self.min_dis_obstacle_from_manipulator=0.0

            self.min_dis_obstacle_from_objects=0.0



        self.kinematics=Kinematics()

        #self.laser_init_pose ,self.laser_init_ori = [-0.25, 1.2, 0.6] , [0,0,-1.57]

        self.laser_init_pose ,self.laser_init_ori = [0, -1.4, 0.4] , [0,0,0]

        self.current_joints_points_arm1 , self.current_joints_points_arm2 = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]] , [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

        self.objects_location=array_of_arrays()

        self.pointclouds=PointCloud()

        self.pointclouds_after_filter=PointCloud()

        self.is_grab_object=True

        self.obj_id=2000

        self.gripper_steps = 700

        self.arm1_gripper , self.arm2_gripper = 0,0

        self.pose_accuracy=0.2

        self.num_points_on_link=6

        self.is_lidar_on=True

        self.operation_mode=True

        self.move_t=1


        ############# cubes position #####################
        #0.58

        self.cubes_poses_arm1=[[0.15,0.3,0.58],[0.15,0.25,0.58],[0.25,0.3,0.58],[0.25,0.25,0.58]]

        self.cubes_poses_arm2=[[-0.15,0.3,0.58],[-0.15,0.25,0.58],[-0.25,0.3,0.58],[-0.25,0.25,0.58]]

        #self.cubes_poses_arm1=[[0.25,-0.3,0.47]]

       # self.cubes_poses_arm2=[[-0.25,-0.3,0.47]]

        self.destination_pose=[0.4,-0.4,0.6]

        self.destination_pose2=[-0.4,-0.4,0.6]


    ############## Subscribers Functions ######################

    def joint_state_callback(self,data):

        self.joint_state_data=data


    def joint_state_arm2_callback(self,data):

        self.joint_state_data_arm2=data


    def occupied_cells_callback(self,data):

        self.occupied_cells=data

    def velodyne_points_callback(self,data):

        self.pointclouds=data


    def pcl_after_filter_callback(self,data):

        self.pointclouds_after_filter=data


    def joy_callback(self,data):

        self.X_button=data.buttons[0]

        self.O_button=data.buttons[1]

        self.start_button=data.buttons[7]

        self.back_button=data.buttons[6]


    def objects_locations_callback(self,data):

        self.objects_locations=data
        



    ############ Controll arms Function ##########


    def constrain(self,pose):

        if(abs(pose[0])>3.14):

                pose[0]=3.14*np.sign(pose[0])

        if(abs(pose[1])>1.55):

                pose[1]=1.55*np.sign(pose[1])

        if(abs(pose[2])>1.55):

                pose[2]=1.55*np.sign(pose[2])

        if(abs(pose[3])>1.55):

                pose[3]=1.55*np.sign(pose[3])

        if(abs(pose[4])>1.55):

                pose[4]=1.55*np.sign(pose[4])

        if(abs(pose[5])>1.55):

                pose[5]=1.55*np.sign(pose[5])
        
        if(abs(pose[6])>1.55):

                pose[6]=1.55*np.sign(pose[6])

        if(abs(pose[7])>3.14):

                pose[7]=3.14*np.sign(pose[7])

        if(abs(pose[8])>1.55):

                pose[8]=1.55*np.sign(pose[8])

        if(abs(pose[9])>1.55):

                pose[9]=1.55*np.sign(pose[9])

        if(abs(pose[10])>1.55):

                pose[10]=1.55*np.sign(pose[10])
        
        if(abs(pose[11])>1.55):

                pose[11]=1.55*np.sign(pose[11])

        if(abs(pose[12])>1.55):

                pose[12]=1.55*np.sign(pose[12])

        if(abs(pose[13])>1):

                pose[13]=1*np.sign(pose[13])

        return pose






    def set_joints_state(self,joints_state_array_arm1,joints_state_array_arm2):

        self.set_joints_msg.value=joints_state_array_arm1

        self.pub_set_joints_arm1.publish(self.set_joints_msg)

        self.pub_set_gripper_arm1.publish(joints_state_array_arm1[6])

        self.set_joints_msg.value=joints_state_array_arm2

        self.pub_set_joints_arm2.publish(self.set_joints_msg)

        self.pub_set_gripper_arm2.publish(joints_state_array_arm2[6])



    ####### Transformation Matrix ########

    def Transformation_Matrix(self,pos,ori,init_point):

        point_return=Point()

        R_x = np.array([[1 , 0 , 0] , [0 , np.cos(ori[0]) ,-np.sin(ori[0]) ] , [0 , np.sin(ori[0]) , np.cos(ori[0])]])

        R_y = np.array([[np.cos(ori[1]) , 0 , np.sin(ori[1])] , [0 , 1, 0 ] , [-np.sin(ori[1]) , 0 , np.cos(ori[1])]])

        R_z = np.array([[np.cos(ori[2]) , -np.sin(ori[2]) , 0] , [np.sin(ori[2]) , np.cos(ori[2]) , 0] , [0 , 0 , 1]])

        R=np.linalg.multi_dot([R_z,R_y,R_x])

        p_new=np.dot(R,pos)

        point_new=(p_new[0]+init_point[0],p_new[1]+init_point[1],p_new[2]+init_point[2])

        point_return.x , point_return.y , point_return.z = point_new[0] , point_new[1] , point_new[2]

        return point_return



    def points_linspace_algo(self,joints):

        points_array=[]

        for j in range(len(joints)-1):

            points_x_array=np.linspace(joints[j][0],joints[j+1][0],num = self.num_points_on_link,endpoint = True,retstep = False,dtype = None)

            points_y_array=np.linspace(joints[j][1],joints[j+1][1],num = self.num_points_on_link,endpoint = True,retstep = False,dtype = None)

            points_z_array=np.linspace(joints[j][2],joints[j+1][2],num = self.num_points_on_link,endpoint = True,retstep = False,dtype = None)

            for i in range(self.num_points_on_link):

                points_array.append((points_x_array[i],points_y_array[i],points_z_array[i]))

        return points_array


    

    def table_pcl(self):

        pcls=[[0.5,0.3,0.35],[-0.5,0.3,0.35]]

        points_array=[]

        for p in range(len(pcls)-1):

            points_x_array=np.linspace(pcls[p][0],pcls[p+1][0],num = 10,endpoint = True,retstep = False,dtype = None)

            points_y_array=np.linspace(pcls[p][1],pcls[p+1][1],num = 10,endpoint = True,retstep = False,dtype = None)

            for i in range(10):

                points_array.append((points_x_array[i],points_y_array[i],pcls[0][2]))


        return points_array



  
    def forward_kinematics_on_manipulator(self,arm_id,pose):


        joint_points_array_origin=self.kinematics.forward_kinematics(arm_id,[pose[0],pose[1],pose[2],pose[3],pose[4],pose[5]])

        joint_points_array=self.points_linspace_algo([joint_points_array_origin[0],joint_points_array_origin[1],joint_points_array_origin[2],joint_points_array_origin[3],
                joint_points_array_origin[4],joint_points_array_origin[5],joint_points_array_origin[6]])

        from_joint4_to_joint6=self.points_linspace_algo([joint_points_array_origin[5],joint_points_array_origin[7]])

        joint_points_array.insert(0,joint_points_array_origin[8])

        joint_points_array.insert(0,joint_points_array_origin[9])

        joint_points_array.extend(from_joint4_to_joint6)

        #joint_points_array.insert(len(joint_points_array),joint6_point)

        return joint_points_array


    def distance_between_2_points(self,a, b):

        return np.linalg.norm(np.array(a) - np.array(b))

    def nearest_object_to_manipulator(self,objects_list,arm_id):

        if(arm_id==2):

            claw_point=self.current_joints_points_arm2[0]

        else:

            claw_point=self.current_joints_points_arm1[4]

        return objects_list[int(np.argmin([np.linalg.norm(np.array(obj) - np.array(claw_point))
                                        for obj in objects_list]))]

 

    def euler_to_quaternion(self,yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

    ################################################################
    ######## Create obstacles array and visualize it ###############
    ################################################################

   

    def remove_obstacles_from_manipulators(self,obstacles_visualization=True):

        id=0

        pcl=self.pointclouds.points

        #print(pcl)

        point_after_transformation=Point()

        obstacles_pcl=PointCloud()

        joint_points_array=[]

        obstacles_pcl.header.stamp=rospy.Time.now()

        obstacles_pcl.header.frame_id='world'

        joints_points_arm1=self.current_joints_points_arm1

        joints_points_arm2=self.current_joints_points_arm2

        joint_points_array.extend(joints_points_arm1)

        joint_points_array.extend(joints_points_arm2)

    

        for point in pcl:

            
            point_tuple=(point.x,point.y,point.z)

            point_after_transformation=self.Transformation_Matrix(point_tuple,self.laser_init_ori,self.laser_init_pose)

            point_tuple=(point_after_transformation.x,point_after_transformation.y,point_after_transformation.z)

            skip=False

            #print(point_tuple)  0.7
            
            if(point_tuple[2]>0.0 and self.d2_distance(joints_points_arm1[0],point_tuple)<0.7): #and distance_between_2_points(joint0_point,point_tuple)<3):

                for joint_point in joint_points_array:

                    dis_obstacle_from_manipulator=self.distance_between_2_points(joint_point,point_tuple)
                    
                    if(dis_obstacle_from_manipulator<self.min_dis_obstacle_from_manipulator):

                        skip=True

                        break
                   
                if(skip):
                    continue

                #pointcloud=Point()

                #pointcloud.x , pointcloud.y ,pointcloud.z  = point_after_transformation[0] ,point_after_transformation[1] ,point_after_transformation[2]

                obstacles_pcl.points.append(point_after_transformation)
      
                #id+=1

        if(obstacles_visualization):

            for m in range(0,5):

                self.pub_pointcloud_after_filter.publish(obstacles_pcl)
        
 

   




    def create_obstacles_array(self,obstacles_visualization=True):

        self.remove_obstacles_from_manipulators()

        pcl_ = rospy.wait_for_message("pcl_after_filter",PointCloud)

        obstacles_array=[]

        pcl=self.pointclouds_after_filter.points

        for point in pcl:

            point_tuple=(point.x,point.y,point.z)

            obstacles_array.append(point_tuple)

        obstacles_array.extend(self.table_pcl())

        return obstacles_array









        

    #############################################################
    ########## Create Algorithm's Markers Functions #############
    ############################################################


    def d2_distance(self,pointA,pointB):

        return np.sqrt((pointA[0]-pointB[0])**2 + (pointA[1]-pointB[1])**2)




    def create_joint_marker(self,position,i):

        joint=Marker()

        joint.header.stamp=rospy.Time.now()

        joint.header.frame_id='world'

        joint.ns='joints'

        joint.id=i

        joint.type=1

        joint.lifetime=rospy.Duration(3)

        joint.pose.position.x ,joint.pose.position.y,joint.pose.position.z = position[0],position[1],position[2]

        joint.scale.x , joint.scale.y , joint.scale.z = 0.05 ,0.05 ,0.05
    
        joint.color.b , joint.color.a = 1,0.9

        return joint


    


    def create_object_marker(self,position,i):

        objectt=Marker()

        objectt.header.stamp=rospy.Time.now()

        objectt.header.frame_id='world'

        objectt.ns='objects'

        objectt.id=i

        objectt.type=1

        objectt.lifetime=rospy.Duration(10)

        objectt.pose.position.x ,objectt.pose.position.y,objectt.pose.position.z =position[0],position[1],position[2]

        objectt.scale.x , objectt.scale.y , objectt.scale.z = 0.05 ,0.05 ,0.05

        objectt.color.b ,objectt.color.g , objectt.color.a = 1,1,0.9

        return objectt


    #############################################################
    ########## Visualize Algorithm's Markers Functions ##########
    #############################################################





    def visualize_path(self,path,visualize=True):

        n=0

        joint_state=path_msg()

        q1 = self.euler_to_quaternion(self.kinematics.arm1_init_orientation[0], self.kinematics.arm1_init_orientation[1], self.kinematics.arm1_init_orientation[2])

        q2 = self.euler_to_quaternion(self.kinematics.arm2_init_orientation[0], self.kinematics.arm2_init_orientation[1], self.kinematics.arm2_init_orientation[2])

        path_len=len(path)

        for pose in path:

            joint_state.waypoint.append(Float32MultiArray())

            joint_state.waypoint[n].layout.dim.append(MultiArrayDimension())

            joint_state.waypoint[n].layout.dim[0].label=str(n)
	
            joint_state.waypoint[n].data=[pose[0],pose[1],pose[2],pose[3],pose[4],pose[5],self.kinematics.arm1_init_point[0],self.kinematics.arm1_init_point[1],self.kinematics.arm1_init_point[2] ,
                            q1[0] ,q1[1] ,q1[2],q1[3] , pose[6],pose[7],pose[8],pose[9],pose[10],pose[11],self.kinematics.arm2_init_point[0] ,self.kinematics.arm2_init_point[1] ,
                            self.kinematics.arm2_init_point[2] , q2[0],q2[1],q2[2],q2[3],self.arm1_gripper,self.arm2_gripper,path_len]

            n+=1

        self.pub_trajectory_state.publish(joint_state)







    def visualize_objects(self,visualize=True):

        objects=MarkerArray()

        objects_list=[]

        for ob in range(0,len(self.cubes_poses_arm1)):

            obj_pose=self.cubes_poses_arm1[ob]

            obj=self.create_object_marker(obj_pose,self.obj_id)

            objects.markers.append(obj)

            self.obj_id+=1

            objects_list.append(obj_pose)



        for ob in range(0,len(self.cubes_poses_arm2)):

            obj_pose=self.cubes_poses_arm1[ob]

            obj=self.create_object_marker(obj_pose,self.obj_id+20)

            objects.markers.append(obj)

            self.obj_id+=1

            objects_list.append(obj_pose)


        if(visualize):

            self.pub_visualization.publish(objects)

        return objects_list



    def visualize_joints(self,visualize=True):

        id=3000

        joints=MarkerArray()

        joint_points_array=[]

        joints_points_arm1=self.current_joints_points_arm1

        joints_points_arm2=self.current_joints_points_arm2

        joint_points_array.extend(joints_points_arm1)

        joint_points_array.extend(joints_points_arm2)

            

        
        for joint_point in joint_points_array:

            joint=self.create_joint_marker(joint_point,id)

            joints.markers.append(joint)

            id+=1

        if(visualize):

            self.pub_visualization.publish(joints)

        

    ##################################################
    ####### The Navigation Function ##################
    ##################################################


    def navigation(self,goal_pose_arm1,goal_pose_arm2):

        print (" planning path ")

        self.current_joints_points_arm1 , self.current_joints_points_arm2 = self.forward_kinematics_on_manipulator(1,self.joint_state_data.position) , self.forward_kinematics_on_manipulator(2,self.joint_state_data_arm2.position)

        start_pose=(self.joint_state_data.position[0],self.joint_state_data.position[1],self.joint_state_data.position[2],self.joint_state_data.position[3],self.joint_state_data.position[4],self.joint_state_data.position[5],
                self.joint_state_data_arm2.position[0],self.joint_state_data_arm2.position[1],self.joint_state_data_arm2.position[2],self.joint_state_data_arm2.position[3],self.joint_state_data_arm2.position[4],self.joint_state_data_arm2.position[5])

        goal_pose=(goal_pose_arm1[0],goal_pose_arm1[1],goal_pose_arm1[2],goal_pose_arm1[3],goal_pose_arm1[4],goal_pose_arm1[5],
                goal_pose_arm2[0],goal_pose_arm2[1],goal_pose_arm2[2],goal_pose_arm2[3],goal_pose_arm2[4],goal_pose_arm2[5])

        self.arm1_gripper , self.arm2_gripper = goal_pose_arm1[6] , goal_pose_arm2[6]

        obstacles_array=self.create_obstacles_array()
        
        drrt = DynamicRrt(start_pose, goal_pose, 0.3, 0.1, 0.1, 1000,obstacles_array)

        drrt.obstacles_array=obstacles_array

        t_start=time.time()

        path_orig=drrt.planning()

        print("path optimization")

        path=drrt.path_optimization(path_orig)

        print("finished optimizing path")

        path_to_cheack_for_replanning=path.copy()

        self.visualize_path(path)

        path_len=len(path)

        t_start=time.time()

        i=0

        

        while i<path_len:

            self.operation_mode=True

            if(self.operation_mode):

                pose=path[i]

                joints_state_array_arm1=[pose[0],pose[1],pose[2],pose[3],pose[4],pose[5],goal_pose_arm1[6]]

                joints_state_array_arm2=[pose[6],pose[7],pose[8],pose[9],pose[10],pose[11],goal_pose_arm2[6]]

                self.set_joints_state(joints_state_array_arm1,joints_state_array_arm2)

                time.sleep(self.move_t)

                self.current_joints_points_arm1 , self.current_joints_points_arm2 = self.forward_kinematics_on_manipulator(1,self.joint_state_data.position) , self.forward_kinematics_on_manipulator(2,self.joint_state_data_arm2.position)

                if(self.is_lidar_on):

                    velodyne_pcl = rospy.wait_for_message("velodyne_pcl",PointCloud)

                    drrt.obstacles_array=self.create_obstacles_array()

                path_to_cheack_for_replanning.remove(path_to_cheack_for_replanning[0])
          
                if (drrt.is_obstacle_in_path(path_to_cheack_for_replanning)):

                    drrt.InvalidateNodes()

                    start_time_replanning=time.time()
       
                    print("Path is Replanning ...")

                    path_orig = drrt.replanning(drrt.waypoints[len(path_orig)-i-1])

                    path=path_orig.copy()


                    path.insert(len(path),pose)


                    path = drrt.path_optimization(path)

                    self.visualize_path(path)

                    path_to_cheack_for_replanning=path.copy()


                    path_len=len(path)

                    i=0

                i+=1

                

                


            

  
           



########################################################
############### DRRT Algorithm Classes #################
########################################################

class Node(object):

    def __init__(self, n):

        self.pos = n

        self.parent = None

        self.flag = "VALID"


class Edge:

    def __init__(self, n_p, n_c):

        self.parent = n_p

        self.child = n_c

        self.flag = "VALID"


class DynamicRrt(object):

    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, waypoint_sample_rate, iter_max,obstacles_array):

        self.s_start = Node(s_start)

        self.s_goal = Node(s_goal)

        self.step_len = step_len

        self.goal_sample_rate = goal_sample_rate

        self.waypoint_sample_rate = waypoint_sample_rate

        self.iter_max = iter_max

        self.vertex = [self.s_start]

        self.vertex_old = []

        self.vertex_new = []

        self.edges = []

        self.kinematics=Kinematics()

        self.range_1 = (-3.14,3.14)

        self.range_2 = (-1.55,1.55)

        self.range_3 = (-1.55,1.55)

        self.range_4 = (-1.55,1.55)

        self.range_5 = (-1.55,1.55)

        self.range_6 = (-1.55,1.55)

        self.path = []

        self.waypoints = []

        self.collision_dist=0.13
        #self.collision_dist=0.1 

        self.obstacles_array=obstacles_array

        self.sensitivity_to_obstacles_after_cheacking_if_obstacle_in_path=0.11

        self.delta_generate_random_node=0.2

        self.num_of_cheack_points = 40

        self.step_size=0.3

        self.num_points_on_link=6



    def distance_between_2_points(self,a, b):

        return np.linalg.norm(np.array(a) - np.array(b))



    def is_collision_in_cheack_point(self,current_pose_index,desired_pose_index,path):

        i=1

        dis=self.distance_between_2_points(path[current_pose_index],path[desired_pose_index])

        dis_between_points=float(dis/self.num_of_cheack_points)

        direction_vec=np.array(path[desired_pose_index]) - np.array(path[current_pose_index])

        direction_vec=direction_vec/np.linalg.norm(direction_vec)

        while i<self.num_of_cheack_points:

            cheack_point = path[current_pose_index] + i*dis_between_points*direction_vec


            if(self.is_collision(cheack_point,self.obstacles_array,self.kinematics,False,False)):

                return True

            i+=1

        return False


    def is_obstacle_in_path(self,path):

        for j in range(0,len(path)-1):

            if(self.is_collision(path[j],self.obstacles_array,self.kinematics,True,False)):

                return True

        return False




    def increase_path_points(self,path):

        increased_path=[]



        for j in range(0,len(path)-1):

            direction_vec=np.array(path[j+1]) - np.array(path[j])

            direction_vec=direction_vec/np.linalg.norm(direction_vec)

            dis=self.distance_between_2_points(path[j+1],path[j])

      

            num_waypoints=int(dis/self.step_size)

         

            if(num_waypoints==0):

                num_waypoints=1

            for k in range(0,num_waypoints):

                waypoint = path[j] + k*self.step_size*direction_vec

   

                increased_path.append(waypoint)

        increased_path.append(path[len(path)-1])

        return increased_path













    def path_optimization(self,path):

        current_pose_index=len(path)-1

        optimazed_path=[path[current_pose_index]]

        desired_pose_index=0

        optimazed_path_index=[len(path)-1]


        n=1
        

        while True:

            if(self.is_collision_in_cheack_point(current_pose_index,desired_pose_index,path) and abs(desired_pose_index-current_pose_index)>1):

                desired_pose_index=current_pose_index-int(current_pose_index/(n*2))

                n+=1


            else:

                optimazed_path.append(path[desired_pose_index])

                optimazed_path_index.append(desired_pose_index)

                if(desired_pose_index==0):

                    optimazed_path=self.increase_path_points(optimazed_path)


                    return optimazed_path

                current_pose_index=desired_pose_index

                desired_pose_index=0

                n=1







    def is_node_clean_of_sight_to_goal(self):


        while self.stop_thread == False:

            if(len(self.vertex)>2):

                i=1

                skip_to_next_node = False

                current_node=self.q.get()

                #current_node=self.node_new

                dis=self.distance_between_2_points(current_node.pos,self.s_goal.pos)

                dis_between_points=float(dis/self.num_of_cheack_points)

                direction_vec=np.array(self.s_goal.pos) - np.array(current_node.pos)

                direction_vec=direction_vec/np.linalg.norm(direction_vec)

                while i<self.num_of_cheack_points:

                    cheack_point = current_node.pos + i*dis_between_points*direction_vec


                    if(self.is_collision(cheack_point,self.obstacles_array,self.kinematics,False,False)):

                        skip_to_next_node = True

                        break

                    i+=1

                if(skip_to_next_node == False):

                    print(" found clean line of sight ")

                    self.is_node_clean_of_sight = True

                    self.node_who_have_clean_line_of_sight_with_goal = current_node




            



    def planning(self):

        count_times_of_non_collisions , count_times_of_collisions = 1 , 0

        collisions_weight=0.5

        if(self.is_collision(self.s_goal,self.obstacles_array,self.kinematics)):

            print('goal pose close to an obstacle')
            
        i=0

        algo=Algorithm()

        self.stop_thread = False

        self.is_node_clean_of_sight = False

        self.q = Queue(maxsize = 200)

        Thread(target=self.is_node_clean_of_sight_to_goal, args=()).start()

        

        while i<self.iter_max:
    
            node_rand = self.generate_random_node(self.goal_sample_rate)

            node_near = self.nearest_neighbor(self.vertex, node_rand)

            node_new = self.new_state(node_near, node_rand)

            if(self.is_collision(self.s_goal,self.obstacles_array,self.kinematics)):

                algo.current_joints_points_arm1 , algo.current_joints_points_arm2 = algo.forward_kinematics_on_manipulator(1,algo.joint_state_data.position) , algo.forward_kinematics_on_manipulator(2,algo.joint_state_data_arm2.position)

                velodyne_pcl = rospy.wait_for_message("velodyne_pcl",PointCloud)

                self.obstacles_array=algo.create_obstacles_array()

                print('goal pose close to an obstacle')
            
                i=0


            if (self.is_node_clean_of_sight):

                self.new_state(self.node_who_have_clean_line_of_sight_with_goal, self.s_goal)

                path,path_nodes = self.extract_path(self.node_who_have_clean_line_of_sight_with_goal)

                self.path = path

                self.waypoints = self.extract_waypoint(self.node_who_have_clean_line_of_sight_with_goal)

                print('found path from clean of sight  ',i)

                self.stop_thread = True

                return path


            


            if node_new and not self.is_collision(node_new,self.obstacles_array,self.kinematics):

                self.q.put(node_new)

                count_times_of_non_collisions+=1

                count_times_of_collisions-=1

                self.vertex.append(node_new)

                self.edges.append(Edge(node_near, node_new))

                dist, _ = self.get_distance_and_direction(node_new, self.s_goal)

                if dist <= self.step_len:

                    self.new_state(node_new, self.s_goal)

                    path,path_nodes = self.extract_path(node_new)

                    self.path = path

                    self.waypoints = self.extract_waypoint(node_new)

                    print('found path  ',i)

                    self.stop_thread = True

                    return path

                i+=1

            else:

                count_times_of_collisions+=1

                count_times_of_non_collisions-=1

            if(count_times_of_non_collisions<0):

                count_times_of_non_collisions=0

            if(count_times_of_collisions<0):

                count_times_of_collisions=0

            self.goal_sample_rate = count_times_of_non_collisions/(count_times_of_non_collisions + collisions_weight*count_times_of_collisions)



            if(i==self.iter_max-2):

                algo=Algorithm()

                self.q = Queue(maxsize = 200)

                count_times_of_non_collisions , count_times_of_collisions = 1 , 0

                joint_state_data = rospy.wait_for_message("/robotis/present_joint_states", JointState)

                joint_state_data_arm2 = rospy.wait_for_message("/robotis/present_joint_states_arm2", JointState)

                algo.current_joints_points_arm1 , algo.current_joints_points_arm2 = algo.forward_kinematics_on_manipulator(1,algo.joint_state_data.position) , algo.forward_kinematics_on_manipulator(2,algo.joint_state_data_arm2.position)

                velodyne_pcl = rospy.wait_for_message("velodyne_pcl",PointCloud)

                self.obstacles_array=algo.create_obstacles_array()

                print("didnt find available path")

                self.vertex = [self.s_start]

                self.edges=[]

                if(self.is_collision(self.s_goal,self.obstacles_array,self.kinematics)):

                    print('goal pose close to an obstacle')
        
                i=0

        return None




   


    def InvalidateNodes(self):

        for edge in self.edges:

            if self.is_collision(edge.parent, self.obstacles_array,self.kinematics):

                edge.child.flag = "INVALID"


    def points_linspace(self,joints):

        points_array=[]

        for j in range(len(joints)-1):

            points_x_array=np.linspace(joints[j][0],joints[j+1][0],num = self.num_points_on_link,endpoint = True,retstep = False,dtype = None)

            points_y_array=np.linspace(joints[j][1],joints[j+1][1],num = self.num_points_on_link,endpoint = True,retstep = False,dtype = None)

            points_z_array=np.linspace(joints[j][2],joints[j+1][2],num = self.num_points_on_link,endpoint = True,retstep = False,dtype = None)

            for i in range(self.num_points_on_link):

                points_array.append((points_x_array[i],points_y_array[i],points_z_array[i]))

        return points_array

    
    def is_collision(self,node,obstacles_array,kinematics,is_checked_obstacle_in_path=False,is_Node=True):

        collision_dist=self.collision_dist
        
        if(is_checked_obstacle_in_path):

            collision_dist=self.collision_dist-self.sensitivity_to_obstacles_after_cheacking_if_obstacle_in_path

        if(is_Node):

            pose=node.pos

        else:

            pose=node

   

        if(self.is_arm1_or_arm2_near_obstacle_or_arm1_is_near_arm2(pose,obstacles_array,kinematics,collision_dist)):

            return True

        return False

    def is_arm1_or_arm2_near_obstacle_or_arm1_is_near_arm2(self,pose,obstacles_array,kinematics,collision_dist):

        #t1=time.time()

        joint_points_array_arm1_origin=kinematics.forward_kinematics(1,[pose[0],pose[1],pose[2],pose[3],pose[4],pose[5]])

        joint_points_array_arm2_origin=kinematics.forward_kinematics(2,[pose[6],pose[7],pose[8],pose[9],pose[10],pose[11]])

        joint_points_array_arm1_arm2=[]


        joint_points_array_arm1=self.points_linspace([joint_points_array_arm1_origin[0],joint_points_array_arm1_origin[1],joint_points_array_arm1_origin[2],joint_points_array_arm1_origin[3],
                joint_points_array_arm1_origin[4],joint_points_array_arm1_origin[5],joint_points_array_arm1_origin[6]])

        joint_points_array_arm1.insert(0,joint_points_array_arm1_origin[8])

        joint_points_array_arm1.insert(0,joint_points_array_arm1_origin[9])

        from_joint4_to_joint6=self.points_linspace([joint_points_array_arm1_origin[5],joint_points_array_arm1_origin[7]])

        joint_points_array_arm1.extend(from_joint4_to_joint6)



        joint_points_array_arm2=self.points_linspace([joint_points_array_arm2_origin[0],joint_points_array_arm2_origin[1],joint_points_array_arm2_origin[2],joint_points_array_arm2_origin[3],
                joint_points_array_arm2_origin[4],joint_points_array_arm2_origin[5],joint_points_array_arm2_origin[6]])

        joint_points_array_arm2.insert(0,joint_points_array_arm2_origin[8])

        joint_points_array_arm2.insert(0,joint_points_array_arm2_origin[9])

        from_joint4_to_joint6=self.points_linspace([joint_points_array_arm2_origin[5],joint_points_array_arm2_origin[7]])

        joint_points_array_arm2.extend(from_joint4_to_joint6)

        



        joint_points_array_arm1_arm2.extend(joint_points_array_arm1)

        joint_points_array_arm1_arm2.extend(joint_points_array_arm2)

        #print(time.time()-t1)

        #print(joint_points_array_arm1_arm2)

        if(len(obstacles_array)!=0):

            distances=distance.cdist(joint_points_array_arm1_arm2, obstacles_array, 'euclidean')

            min_dis=np.amin(distances)

            if(min_dis<collision_dist):

                return True

        distances=distance.cdist(joint_points_array_arm1, joint_points_array_arm2, 'euclidean')

        min_dis=np.amin(distances)

        if(min_dis<collision_dist):

            return True

        if(joint_points_array_arm1_origin[6][2]>=kinematics.arm1_init_point[2]-0.06 or joint_points_array_arm2_origin[6][2]>=kinematics.arm2_init_point[2]-0.06):

            return True

        return False


        


   

   


    def replanning(self,start_node):

        self.TrimRRT()

        orig_vertex=self.vertex

        orig_edges=self.edges

        i=0

        algo=Algorithm()

        count_times_of_non_collisions , count_times_of_collisions = 1 , 0

        collisions_weight=1

        self.iter_max=self.iter_max-len(orig_vertex)

        self.stop_thread = False

        self.is_node_clean_of_sight = False

        self.q = Queue(maxsize = 300)

        Thread(target=self.is_node_clean_of_sight_to_goal, args=()).start()

        while i<self.iter_max:

            node_rand = self.generate_random_node_replanning(self.goal_sample_rate, self.waypoint_sample_rate)
   
            node_near = self.nearest_neighbor(self.vertex, node_rand)

            node_new = self.new_state(node_near, node_rand)

            if(self.is_collision(self.s_goal,self.obstacles_array,self.kinematics)):

                algo.current_joints_points_arm1 , algo.current_joints_points_arm2 = algo.forward_kinematics_on_manipulator(1,algo.joint_state_data.position) , algo.forward_kinematics_on_manipulator(2,algo.joint_state_data_arm2.position)

                velodyne_pcl = rospy.wait_for_message("velodyne_pcl",PointCloud)

                self.obstacles_array=algo.create_obstacles_array()

                print('goal pose close to an obstacle')
            
                i=0


            if (self.is_node_clean_of_sight):

                self.new_state(self.node_who_have_clean_line_of_sight_with_goal, self.s_goal)

                path,path_nodes = self.extract_path(self.node_who_have_clean_line_of_sight_with_goal)

                self.path = path

                self.waypoints = self.extract_waypoint(self.node_who_have_clean_line_of_sight_with_goal)

                print('found path from clean of sight  ',i)

                self.stop_thread = True

                #self.q = Queue(maxsize = 2000)

                return path

            

            if node_new and not self.is_collision(node_new,self.obstacles_array,self.kinematics):

                self.q.put(node_new)

                count_times_of_non_collisions+=1

                count_times_of_collisions-=1

                self.vertex.append(node_new)

                self.vertex_new.append(node_new)

                self.edges.append(Edge(node_near, node_new))

                dist, _ = self.get_distance_and_direction(node_new, self.s_goal)

                if dist <= self.step_len:

                    self.new_state(node_new, self.s_goal)

                    path = self.extract_path_replanning(node_new,start_node)

                    print("found_path: length ", len(path))

                    self.stop_thread = True
                   
                    return path

                i+=1

            else:

                count_times_of_collisions+=1

                count_times_of_non_collisions-=1


            if(count_times_of_non_collisions<0):

                count_times_of_non_collisions=0

            if(count_times_of_collisions<0):

                count_times_of_collisions=0


            self.goal_sample_rate = count_times_of_non_collisions/(count_times_of_non_collisions + collisions_weight*count_times_of_collisions)

            if(i==self.iter_max-2):

                self.q = Queue(maxsize = 300)

                count_times_of_non_collisions , count_times_of_collisions = 1 , 0

                joint_state_data = rospy.wait_for_message("/robotis/present_joint_states", JointState)

                joint_state_data_arm2 = rospy.wait_for_message("/robotis/present_joint_states_arm2", JointState)

                algo.current_joints_points_arm1 , algo.current_joints_points_arm2 = algo.forward_kinematics_on_manipulator(1,algo.joint_state_data.position) , algo.forward_kinematics_on_manipulator(2,algo.joint_state_data_arm2.position)

                velodyne_pcl = rospy.wait_for_message("velodyne_pcl",PointCloud)

                self.obstacles_array=algo.create_obstacles_array()

                print("didnt find available path trying again")

                self.vertex = orig_vertex

                self.edges=orig_edges

                if(self.is_collision(self.s_goal,self.obstacles_array,self.kinematics)):

                    print('goal pose close to an obstacle')
        
                i=0

        return None


    def TrimRRT(self):

        for i in range(1, len(self.vertex)):

            node = self.vertex[i]

            node_p = node.parent

            if node_p.flag == "INVALID":

                node.flag = "INVALID"

        self.vertex = [node for node in self.vertex if node.flag == "VALID"]

        self.vertex_old = copy.deepcopy(self.vertex)

        self.edges = [Edge(node.parent, node) for node in self.vertex[1:len(self.vertex)]]


    def generate_random_node(self, goal_sample_rate):

        delta=self.delta_generate_random_node

        if np.random.random() > goal_sample_rate:

            return Node((np.random.uniform(self.range_1[0] + delta, self.range_1[1] - delta),
                         np.random.uniform(self.range_2[0] + delta, self.range_2[1] - delta),
                         np.random.uniform(self.range_3[0] + delta, self.range_3[1] - delta),
                         np.random.uniform(self.range_4[0] + delta, self.range_4[1] - delta),
                         np.random.uniform(self.range_5[0] + delta, self.range_5[1] - delta),
                         np.random.uniform(self.range_6[0] + delta, self.range_6[1] - delta),
                         np.random.uniform(self.range_1[0] + delta, self.range_1[1] - delta),
                         np.random.uniform(self.range_2[0] + delta, self.range_2[1] - delta),
                         np.random.uniform(self.range_3[0] + delta, self.range_3[1] - delta),
                         np.random.uniform(self.range_4[0] + delta, self.range_4[1] - delta),
                         np.random.uniform(self.range_5[0] + delta, self.range_5[1] - delta),
                         np.random.uniform(self.range_6[0] + delta, self.range_6[1] - delta)))

        return self.s_goal



    def generate_random_node_replanning(self, goal_sample_rate, waypoint_sample_rate):
        #delta = self.utils.delta
        delta=0.1
        p = np.random.random()
        if p < goal_sample_rate:
            return self.s_goal
        elif goal_sample_rate < p < goal_sample_rate + waypoint_sample_rate:
            return self.waypoints[np.random.randint(0, len(self.waypoints) - 1)]
        else:
            return Node((np.random.uniform(self.range_1[0] + delta, self.range_1[1] - delta),
                         np.random.uniform(self.range_2[0] + delta, self.range_2[1] - delta),
                         np.random.uniform(self.range_3[0] + delta, self.range_3[1] - delta),
                         np.random.uniform(self.range_4[0] + delta, self.range_4[1] - delta),
                         np.random.uniform(self.range_5[0] + delta, self.range_5[1] - delta),
                         np.random.uniform(self.range_6[0] + delta, self.range_6[1] - delta),
                         np.random.uniform(self.range_1[0] + delta, self.range_1[1] - delta),
                         np.random.uniform(self.range_2[0] + delta, self.range_2[1] - delta),
                         np.random.uniform(self.range_3[0] + delta, self.range_3[1] - delta),
                         np.random.uniform(self.range_4[0] + delta, self.range_4[1] - delta),
                         np.random.uniform(self.range_5[0] + delta, self.range_5[1] - delta),
                         np.random.uniform(self.range_6[0] + delta, self.range_6[1] - delta)))


    def nearest_neighbor(self,node_list, n):

        return node_list[int(np.argmin([np.linalg.norm(np.array(nd.pos) - np.array(n.pos))
                                        for nd in node_list]))]


    def new_state(self, node_start, node_end):

        dist, direction = self.get_distance_and_direction(node_start, node_end)

        dist = min(self.step_len, dist)

        node_new = Node(node_start.pos+dist*direction)

        node_new.parent = node_start

        return node_new


    def extract_path(self, node_end):
        path = [(self.s_goal.pos[0], self.s_goal.pos[1] , self.s_goal.pos[2],self.s_goal.pos[3],self.s_goal.pos[4],self.s_goal.pos[5],self.s_goal.pos[6],
                self.s_goal.pos[7],self.s_goal.pos[8],self.s_goal.pos[9],self.s_goal.pos[10],self.s_goal.pos[11])]

        node_now = node_end

        path_nodes=[self.s_goal]

        while node_now.parent is not None:

            node_now = node_now.parent

            path_nodes.append(node_now)

            path.append((node_now.pos[0], node_now.pos[1] , node_now.pos[2],node_now.pos[3],node_now.pos[4],node_now.pos[5],node_now.pos[6],
                node_now.pos[7],node_now.pos[8],node_now.pos[9],node_now.pos[10],node_now.pos[11]))

        return path,path_nodes



    def extract_path_replanning(self, node_end,node_start):

        path = [(self.s_goal.pos[0], self.s_goal.pos[1] , self.s_goal.pos[2],self.s_goal.pos[3],self.s_goal.pos[4],self.s_goal.pos[5],self.s_goal.pos[6],
            self.s_goal.pos[7],self.s_goal.pos[8],self.s_goal.pos[9],self.s_goal.pos[10],self.s_goal.pos[11])]

        node_cur_of_node_start_blood_line=node_start

        node_start_ancestors=[node_start]

        waypoints=[self.s_goal]

        ###### lets take the start Node and get a list of his ancestors ##########

        while node_cur_of_node_start_blood_line.parent is not None:

            node_cur_of_node_start_blood_line = node_cur_of_node_start_blood_line.parent

            node_start_ancestors.append(node_cur_of_node_start_blood_line)

        node_cur=node_end


        ##### lets take a look on the node_end ancestors ####

        while node_cur.parent is not None:

            node_cur = node_cur.parent

            #### if node_start is node_end's ancestor so we have a path #######

            if(node_cur==node_start):

                path.append((node_cur.pos[0], node_cur.pos[1] , node_cur.pos[2],node_cur.pos[3],node_cur.pos[4],node_cur.pos[5],node_cur.pos[6],
                    node_cur.pos[7],node_cur.pos[8],node_cur.pos[9],node_cur.pos[10],node_cur.pos[11]))

                waypoints.append(node_cur)

                self.waypoints=waypoints

                self.path=path

                return path

            ###### if  node_cur isnt ancestor of node_start then add the node to the path and keep going to the next ancestor ####

            if((node_cur in node_start_ancestors)==False):
    
                path.append((node_cur.pos[0], node_cur.pos[1] , node_cur.pos[2],node_cur.pos[3],node_cur.pos[4],node_cur.pos[5],
                    node_cur.pos[6],node_cur.pos[7],node_cur.pos[8],node_cur.pos[9],node_cur.pos[10],node_cur.pos[11]))

                waypoints.append(node_cur)

            ###### if node_end and node_start have a common_ancestor so lets see which ancestor is it in node_start bloodline and add all the ancestors which connecting between start_node to end_node #####

            if(node_cur in node_start_ancestors):

                common_ancestor=node_start_ancestors.index(node_cur)
        
                for i in range(0,common_ancestor):

                    path.append(node_start_ancestors[common_ancestor-i].pos)

                    waypoints.append(node_start_ancestors[common_ancestor-i])
                
                self.waypoints=waypoints

                self.path=path

                return path


    def extract_waypoint(self, node_end):

        waypoints = [self.s_goal]

        node_now = node_end

        while node_now.parent is not None:

            node_now = node_now.parent

            waypoints.append(node_now)

        return waypoints


    def get_distance_and_direction(self,node_start, node_end):

        direction=np.array(node_end.pos) - np.array(node_start.pos)

        dist = np.linalg.norm(direction)

        if(dist==0):

            pass

        else:

            direction=direction/dist

        return dist,direction



#####################################################################
############### Kinematics Class ####################################
#####################################################################
        
class Kinematics(object):

    def __init__(self):

        self.arm1_init_point = (0.3,0.0,1.2)

        self.arm2_init_point = (-0.3,0.0,1.2)

        self.arm1_init_orientation=(3.14,0.0,3.14) # yaw pitch roll

        self.arm2_init_orientation=(0,0.0,3.14) # yaw pitch roll

        #self.arm2_init_orientation=(3.14,0.0,0) # yaw pitch roll

   
    def from_transition_matrix_to_pose(self,T):


        return (T[0][3],T[1][3],T[2][3])


  
    def forward_kinematics(self,arm_ID,joints_state_array):
        j_pos=joints_state_array
        d1=0.159
        a1=0.2659
        a2=0.03
        a3=0.134
        d3=0.258
        a4=0.03

        a5=0.08
        d4=0.12
        joint2_offset=-1.57+0.113
        joint3_offset=-0.113
        joint4_offset=1.57
        joint5_offset=0
        y1=self.arm1_init_orientation[0]
        p1=self.arm1_init_orientation[1]
        r1=self.arm1_init_orientation[2]

        y2=self.arm2_init_orientation[0]
        p2=self.arm2_init_orientation[1]
        r2=self.arm2_init_orientation[2]

      
        if(arm_ID==2):

            T_00=[[np.cos(y2)*np.cos(p2),np.cos(y2)*np.sin(p2)*np.sin(r2)-np.sin(y2)*np.cos(r2),np.cos(y2)*np.sin(p2)*np.sin(r2)+np.sin(y2)*np.sin(r2),self.arm2_init_point[0]],[np.sin(y2)*np.cos(p2),np.sin(y2)*np.sin(p2)*np.sin(r2)+np.cos(y2)*np.cos(r2),np.sin(y2)*np.sin(p2)*np.cos(r2)-np.cos(y2)*np.sin(r2),self.arm2_init_point[1]],[-np.sin(p2),np.cos(p2)*np.sin(r2),np.cos(p2)*np.cos(r2),self.arm2_init_point[2]],[0,0,0,1]]

            pointm1=self.arm2_init_point

        else:

            pointm1=self.arm1_init_point

            T_00=[[np.cos(y1)*np.cos(p1),np.cos(y1)*np.sin(p1)*np.sin(r1)-np.sin(y1)*np.cos(r1),np.cos(y1)*np.sin(p1)*np.sin(r1)+np.sin(y1)*np.sin(r1),self.arm1_init_point[0]],[np.sin(y1)*np.cos(p1),np.sin(y1)*np.sin(p1)*np.sin(r1)+np.cos(y1)*np.cos(r1),np.sin(y1)*np.sin(p1)*np.cos(r1)-np.cos(y1)*np.sin(r1),self.arm1_init_point[1]],[-np.sin(p1),np.cos(p1)*np.sin(r1),np.cos(p1)*np.cos(r1),self.arm1_init_point[2]],[0,0,0,1]]
           
        T_01=np.array([[np.cos(j_pos[0]),0,-np.sin(j_pos[0]),0],[np.sin(j_pos[0]),0,np.cos(j_pos[0]),0],[0,-1,0,d1],[0,0,0,1]])
    
        T_01=np.linalg.multi_dot([T_00,T_01])

        #T_01_l_1=[[1,0,0,np.cos(j_pos[0])*a5],[0,1,0,0],[0,0,1,np.sin(j_pos[0])*a5],[0,0,0,1]]

        T_01_l_1=[[1,0,0,0],[0,1,0,0],[0,0,1,a5],[0,0,0,1]]

        T_01_l_2=[[1,0,0,0],[0,1,0,0],[0,0,1,-a5],[0,0,0,1]]

        point0=self.from_transition_matrix_to_pose(T_01)

        T_01_l=np.linalg.multi_dot([T_01,T_01_l_1])

        point7=self.from_transition_matrix_to_pose(T_01_l)

        T_01_l=np.linalg.multi_dot([T_01,T_01_l_2])

        point8=self.from_transition_matrix_to_pose(T_01_l)

        T_12=np.array([[np.cos(j_pos[1]+joint2_offset),-np.sin(j_pos[1]+joint2_offset),0,a1*np.cos(j_pos[1]+joint2_offset)],[np.sin(j_pos[1]+joint2_offset),np.cos(j_pos[1]+joint2_offset),0,a1*np.sin(j_pos[1]+joint2_offset)],[0,0,1,0],[0,0,0,1]])

        T_02=np.linalg.multi_dot([T_01,T_12])
         
        point1=self.from_transition_matrix_to_pose(T_02)

        T_23=[[np.cos(j_pos[2]+joint3_offset),0,-np.sin(j_pos[2]+joint3_offset),a2*np.cos(j_pos[2]+joint3_offset)],[np.sin(j_pos[2]+joint3_offset),0,np.cos(j_pos[2]+joint3_offset),a2*np.sin(j_pos[2]+joint3_offset)],[0,-1,0,0],[0,0,0,1]]

        T_03=np.linalg.multi_dot([T_01,T_12,T_23])
           
        point2=self.from_transition_matrix_to_pose(T_03)

        T_34=[[np.cos(j_pos[3]),0,np.sin(j_pos[3]),0],[np.sin(j_pos[3]),0,-np.cos(j_pos[3]),0],[0,1,0,d3],[0,0,0,1]]

        T_04=np.linalg.multi_dot([T_01,T_12,T_23,T_34])
        
        point3=self.from_transition_matrix_to_pose(T_04)

        T_45=[[np.cos(j_pos[4]+joint4_offset),0,-np.sin(j_pos[4]+joint4_offset),a3*np.cos(j_pos[4]+joint4_offset)],[np.sin(j_pos[4]+joint4_offset),0,np.cos(j_pos[4]+joint4_offset),a3*np.sin(j_pos[4]+joint4_offset)],[0,-1,0,0],[0,0,0,1]]
    
        T_05=np.linalg.multi_dot([T_01,T_12,T_23,T_34,T_45])
           
        point4=self.from_transition_matrix_to_pose(T_05)

        T_56_l_1=[[1,0,0,d4],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

        T_56_l_2=[[1,0,0,0],[0,1,0,np.cos(j_pos[5])*a4],[0,0,1,np.sin(j_pos[5])*a4],[0,0,0,1]]

        T_56_l=np.linalg.multi_dot([T_56_l_1,T_56_l_2])

        T_56_r_1=[[1,0,0,d4],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

        T_56_r_2=[[1,0,0,0],[0,1,0,np.cos(j_pos[5]+3.14)*a4],[0,0,1,np.sin(j_pos[5]+3.14)*a4],[0,0,0,1]]

        T_56_r=np.linalg.multi_dot([T_56_r_1,T_56_r_2])

        T_06_l=np.linalg.multi_dot([T_01,T_12,T_23,T_34,T_45,T_56_l])

        point5=self.from_transition_matrix_to_pose(T_06_l)

        T_06_r=np.linalg.multi_dot([T_01,T_12,T_23,T_34,T_45,T_56_r])

        point6=self.from_transition_matrix_to_pose(T_06_r)

        return [pointm1,point0,point1,point2,point3,point4,point5,point6,point7,point8]




    def inverse_kinematics(self,arm_id,pos, ori, i=0, t6=0, t1=0):

        pos_array=[pos[0],pos[1],pos[2]]

        if(arm_id==1):

            pos_array[0]=pos_array[0]-self.arm1_init_point[0]

            pos_array[1]=pos_array[1]-self.arm1_init_point[1]

            pos_array[2]=pos_array[2]-self.arm1_init_point[2]

            

            R_x = np.array([[1 , 0 , 0] , [0 , np.cos(self.arm1_init_orientation[2]) ,-np.sin(self.arm1_init_orientation[2]) ] , [0 , np.sin(self.arm1_init_orientation[2]) , np.cos(self.arm1_init_orientation[2])]])

            R_y = np.array([[np.cos(self.arm1_init_orientation[1]) , 0 , np.sin(self.arm1_init_orientation[1])] , [0 , 1, 0 ] , [-np.sin(self.arm1_init_orientation[1]) , 0 , np.cos(self.arm1_init_orientation[1])]])

            R_z = np.array([[np.cos(self.arm1_init_orientation[0]) , -np.sin(self.arm1_init_orientation[0]) , 0] , [np.sin(self.arm1_init_orientation[0]) , np.cos(self.arm1_init_orientation[0]) , 0] , [0 , 0 , 1]])

        else:

            pos_array[0]=pos_array[0]-self.arm2_init_point[0]

            pos_array[1]=pos_array[1]-self.arm2_init_point[1]

            pos_array[2]=pos_array[2]-self.arm2_init_point[2]

            R_x = np.array([[1 , 0 , 0] , [0 , np.cos(self.arm2_init_orientation[2]) ,-np.sin(self.arm2_init_orientation[2]) ] , [0 , np.sin(self.arm2_init_orientation[2]) , np.cos(self.arm2_init_orientation[2])]])

            R_y = np.array([[np.cos(self.arm2_init_orientation[1]) , 0 , np.sin(self.arm2_init_orientation[1])] , [0 , 1, 0 ] , [-np.sin(self.arm2_init_orientation[1]) , 0 , np.cos(self.arm2_init_orientation[1])]])

            R_z = np.array([[np.cos(self.arm2_init_orientation[0]) , -np.sin(self.arm2_init_orientation[0]) , 0] , [np.sin(self.arm2_init_orientation[0]) , np.cos(self.arm2_init_orientation[0]) , 0] , [0 , 0 , 1]])

        R=np.linalg.multi_dot([R_z,R_y,R_x])

        pos_array=np.dot(R,pos_array)


        ori=np.dot(R,ori)


        gripper_length = 178.61 + 50
        gripper_length = 50

        #gripper_length = 0

        gripper_LDO_length = 12 + 25

        fas = [1.4576, 1.3419,np.pi]

        l_i = [0.159, 0.2656991, 0.2597383, 0.123 +0.011+ gripper_length/1000.0]

        s = np.array(
            [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]])

        R06 = np.array([[-np.sin(ori[1]), -np.sin(ori[2]) * np.cos(ori[1]), np.cos(ori[1]) * np.cos(ori[2]), 0],
                        [np.sin(ori[0]) * np.cos(ori[1]),
                         -np.sin(ori[0]) * np.sin(ori[1]) * np.sin(ori[2]) + np.cos(ori[0]) * np.cos(ori[2]),
                         np.sin(ori[0]) * np.sin(ori[1]) * np.cos(ori[2]) + np.sin(ori[2]) * np.cos(ori[0]), 0],
                        [-np.cos(ori[0]) * np.cos(ori[1]),
                         np.sin(ori[0]) * np.cos(ori[2]) + np.sin(ori[1]) * np.sin(ori[2]) * np.cos(ori[0]),
                         np.sin(ori[0]) * np.sin(ori[2]) - np.sin(ori[1]) * np.cos(ori[0]) * np.cos(ori[2]), 0], [0, 0, 0, 1]])

        pc = np.array([[pos_array[0] - (l_i[3]) * np.cos(ori[1]) * np.cos(ori[2])],
                       [pos_array[1] - (l_i[3]) * (
                               (np.sin(ori[0])) * np.sin(ori[1]) * np.cos(ori[2]) + np.sin(ori[2]) * np.cos(ori[0]))],
                       [pos_array[2] - (l_i[3]) * (
                               (np.sin(ori[0])) * np.sin(ori[2]) - np.sin(ori[1]) * np.cos(ori[0]) * np.cos(ori[2]))]])

        R06 = R06[:3, :3]

        gova = pc[2, 0] - l_i[0]

        yeter = s[i, 0] * (pc[0, 0] ** 2 + pc[1, 0] ** 2) ** 0.5

        D = ((yeter ** 2 + gova ** 2 - l_i[1] ** 2 - l_i[2] ** 2) / (2 * l_i[1] * l_i[2]))

        if pc[0, 0] == 0 and pc[1, 0] == 0:

            t1 = t1
        else:

            t1 = np.arctan2(pc[1, 0] / (yeter), pc[0, 0] / (yeter))

        if D > 1:

            raise ArmCannotReachPosition('Arm cannot reach position')

        t3 = np.arctan2(s[i, 1] * (1 - D ** 2) ** 0.5, D)

        t2 = -(np.arctan2(gova, yeter) - np.arctan2(l_i[2] * np.sin(-t3), l_i[1] + l_i[2] * np.cos(-t3)))

        t3 = t3 - fas[1]

        t2 = t2 + fas[0]

        R03 = np.array([[-np.sin(t2 + t3) * np.cos(t1), -np.sin(t1), np.cos(t1) * np.cos(t2 + t3)],
                        [-np.sin(t1) * np.sin(t2 + t3), np.cos(t1), np.sin(t1) * np.cos(t2 + t3)],
                        [-np.cos(t2 + t3), 0, -np.sin(t2 + t3)]])

        # http://aranne5.bgu.ac.il/Free_publications/mavo-lerobotica.pdf
        R36 = np.dot(R03.T, R06)

        st5 = -s[i, 2] * (R36[0, 2] ** 2 + R36[1, 2] ** 2) ** 0.5

        if abs(st5) < 1e-4:

            t5 = 0

            t6 = t6

            t4 = np.arctan2(R36[1, 0], R36[0, 0]) + t6

        else:

            t5 = np.arctan2(st5, R36[2, 2])

            t4 = np.arctan2(R36[1, 2] / st5, R36[0, 2] / st5)

            t6 = np.arctan2(R36[2, 1] / st5, -R36[2, 0] / st5)
        
        

        theta_values = np.array([[t1, t2, t3, t4, t5, t6]])
        
        if np.isnan(np.sum(theta_values)):

            raise ArmCannotReachPosition('Arm cannot reach position')

        return theta_values


class ArmCannotReachPosition(Exception):
    """Raised when the arm cannot reach the position target"""
    pass




class Mission:


    def __init__(self,algo,kinematics):

        self.algo=algo

        self.kinematics=kinematics

        self.goal_arm1 , self.goal_arm2  =  np.zeros(12) , np.zeros(12)

        self.t=3






    def move_to_initial_pose(self):

        inv_arm1=self.kinematics.inverse_kinematics(1,[0.2,0.0,0.6], [-1.57,0,-1.57], 0)[0]

        inv_arm2=self.kinematics.inverse_kinematics(2,[-0.2,0.0,0.6], [-1.57,0,1.57], 0)[0]

        self.goal_arm1=[inv_arm1[0],inv_arm1[1],inv_arm1[2],inv_arm1[3],inv_arm1[4],inv_arm1[5],0.5]

        self.goal_arm2=[inv_arm2[0],inv_arm2[1],inv_arm2[2],inv_arm2[3],inv_arm2[4],inv_arm2[5],0.5]

        self.algo.set_joints_state(self.goal_arm1,self.goal_arm2)



    def two_manipulators_navigate_to_first_cubes(self):

        inv_arm1=self.kinematics.inverse_kinematics(1,self.algo.cubes_poses_arm1[0], [-1.57,0,-1.57], 0)[0]

        inv_arm2=self.kinematics.inverse_kinematics(2,self.algo.cubes_poses_arm2[0], [-1.57,0,1.57], 0)[0]

        self.goal_arm1=[inv_arm1[0],inv_arm1[1],inv_arm1[2],inv_arm1[3],inv_arm1[4],inv_arm1[5],0.5]

        self.goal_arm2=[inv_arm2[0],inv_arm2[1],inv_arm2[2],inv_arm2[3],inv_arm2[4],inv_arm2[5],0.5]

        self.algo.navigation(self.goal_arm1 , self.goal_arm2)


    
    def right_manipulator_move_down_to_grab_the_cube(self,cube_id):

        self.algo.cubes_poses_arm2[0][2]-=0.1

        inv_arm2=self.kinematics.inverse_kinematics(2,self.algo.cubes_poses_arm2[0], [-1.57,0,1.57], 0)[0]

        self.goal_arm2=[inv_arm2[0],inv_arm2[1],inv_arm2[2],inv_arm2[3],inv_arm2[4],inv_arm2[5],0.5]

        self.algo.set_joints_state(self.goal_arm1,self.goal_arm2)




    def left_manipulator_move_down_to_grab_the_cube(self,cube_id):

        self.algo.cubes_poses_arm1[cube_id][2]-=0.11

        inv_arm1=self.kinematics.inverse_kinematics(1,self.algo.cubes_poses_arm1[cube_id], [-1.57,0,-1.57], 0)[0]

        self.goal_arm1=[inv_arm1[0],inv_arm1[1],inv_arm1[2],inv_arm1[3],inv_arm1[4],inv_arm1[5],0.5]

        self.algo.set_joints_state(self.goal_arm1,self.goal_arm2)




    def right_manipulator_grab_the_cube_and_move_up(self,cube_id):

        self.algo.cubes_poses_arm2[cube_id][2]+=0.1

        inv_arm2=self.kinematics.inverse_kinematics(2,self.algo.cubes_poses_arm2[0], [-1.57,0,1.57], 0)[0]

        self.goal_arm2=[inv_arm2[0],inv_arm2[1],inv_arm2[2],inv_arm2[3],inv_arm2[4],inv_arm2[5],0.9]

        self.algo.set_joints_state(self.goal_arm1,self.goal_arm2)



    def left_manipulator_grab_the_cube_and_move_up(self,cube_id):

        self.algo.cubes_poses_arm1[cube_id][2]+=0.11

        inv_arm1=self.kinematics.inverse_kinematics(1,self.algo.cubes_poses_arm1[cube_id], [-1.57,0,-1.57], 0)[0]

        self.goal_arm1=[inv_arm1[0],inv_arm1[1],inv_arm1[2],inv_arm1[3],inv_arm1[4],inv_arm1[5],0.9]

        self.algo.set_joints_state(self.goal_arm1,self.goal_arm2)




    def left_manipulator_navigate_to_cube_right_manipulator_navigate_to_destination_pose(self,cube_id):

        inv_arm1=self.kinematics.inverse_kinematics(1,self.algo.cubes_poses_arm1[cube_id], [-1.57,0,-1.57], 0)[0]

        inv_arm2=self.kinematics.inverse_kinematics(2,self.algo.destination_pose2, [-1.57,0,1.57], 0)[0]

        self.goal_arm1=[inv_arm1[0],inv_arm1[1],inv_arm1[2],inv_arm1[3],inv_arm1[4],inv_arm1[5],0.9]

        self.goal_arm2=[inv_arm2[0],inv_arm2[1],inv_arm2[2],inv_arm2[3],inv_arm2[4],inv_arm2[5],0.9]

        self.algo.navigation(self.goal_arm1 , self.goal_arm2)


    def right_manipulator_navigate_to_cube_left_manipulator_navigate_to_destination_pose(self,cube_id):

        inv_arm1=self.kinematics.inverse_kinematics(1,self.algo.destination_pose, [-1.57,0,-1.57], 0)[0]

        inv_arm2=self.kinematics.inverse_kinematics(2,self.algo.cubes_poses_arm2[cube_id], [-1.57,0,1.57], 0)[0]

        self.goal_arm1=[inv_arm1[0],inv_arm1[1],inv_arm1[2],inv_arm1[3],inv_arm1[4],inv_arm1[5],0.9]

        self.goal_arm2=[inv_arm2[0],inv_arm2[1],inv_arm2[2],inv_arm2[3],inv_arm2[4],inv_arm1[5],0.5]

        self.algo.navigation(self.goal_arm1 , self.goal_arm2)




    def right_manipulator_open_gripper(self):

        inv_arm2=self.kinematics.inverse_kinematics(2,self.algo.destination_pose2, [-1.57,0,1.57], 0)[0]

        self.goal_arm2=[inv_arm2[0],inv_arm2[1],inv_arm2[2],inv_arm2[3],inv_arm2[4],inv_arm2[5],0.5]

        self.algo.set_joints_state(self.goal_arm1,self.goal_arm2)


    def left_manipulator_open_gripper(self):

        inv_arm1=self.kinematics.inverse_kinematics(1,self.algo.destination_pose, [-1.57,0,-1.57], 0)[0]

        self.goal_arm1=[inv_arm1[0],inv_arm1[1],inv_arm1[2],inv_arm1[3],inv_arm1[4],inv_arm1[5],0.5]

        self.algo.set_joints_state(self.goal_arm1,self.goal_arm2)






    def mission_A(self):

        time.sleep(0.5)

        #self.move_to_initial_pose()

        #time.sleep(5)

        #self.two_manipulators_navigate_to_first_cubes()

        #time.sleep(self.t)

        #self.right_manipulator_move_down_to_grab_the_cube(0)

        #time.sleep(self.t)

        #self.right_manipulator_grab_the_cube_and_move_up(0)

        #time.sleep(self.t)

        for cube_id in range(0,len(self.algo.cubes_poses_arm1)):

            self.left_manipulator_navigate_to_cube_right_manipulator_navigate_to_destination_pose(cube_id)

            #self.right_manipulator_open_gripper()

            #time.sleep(self.t)

            #self.left_manipulator_move_down_to_grab_the_cube(cube_id)

            #time.sleep(self.t)

            #self.left_manipulator_grab_the_cube_and_move_up(cube_id)

            #time.sleep(self.t)

            self.right_manipulator_navigate_to_cube_left_manipulator_navigate_to_destination_pose(cube_id+1)

            time.sleep(self.t)

            self.left_manipulator_open_gripper()

            time.sleep(self.t)

            self.right_manipulator_move_down_to_grab_the_cube(cube_id+1)

            time.sleep(self.t)

            self.right_manipulator_grab_the_cube_and_move_up(cube_id+1)

            time.sleep(self.t)



       

      




######################################
########## main function #############
######################################


def main():

    rospy.init_node('thesis_algo', anonymous=True)

    rate = rospy.Rate(10) # 10hz

    start_time=time.time()

    algo=Algorithm()

    kinematics=Kinematics()

    t=time.time()


    
    while not rospy.is_shutdown():

        algo.current_joints_points_arm1 , algo.current_joints_points_arm2 = algo.forward_kinematics_on_manipulator(1,algo.joint_state_data.position) , algo.forward_kinematics_on_manipulator(2,algo.joint_state_data_arm2.position)

        

        if(algo.calibrate_lidar):

            velodyne_pcl = rospy.wait_for_message("velodyne_pcl",PointCloud)

            obstacles_array=algo.create_obstacles_array()

            #algo.visualize_joints()

        else:

            #move_cubes_plane_B(algo,kinematics,t)

            mission=Mission(algo,kinematics)

            mission.mission_A()



        rate.sleep()

    





if __name__ == '__main__':
    try:
       main()
    except rospy.ROSInterruptException:
        pass