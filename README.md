# ttb_dqn
 터틀봇 DQN

## about this repo
This is noetic version of [turtlebot3 machine learning](https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/#machine-learning).

| Origin repo | >> | This repo |
|---|---|---|
|Kinetic| |Noetic|
|TensorFlow| |PyTorch|

## env
* python 3.8
* torch 11.7

## simulation
1. make sure you change file excutable
   `chmod +x turtlebot3_dqn_stage_1`
2. catkin build
3. `roslaunch turtlebot3_gazebo turtlebot3_dqn_stage_2.launch`
4. `roslaunch dqn_ttb turtlebot3_dqn_stage_30.launch`
