# 2. Trust Region Policy Optimization
코드는 로보티즈의 [머신러닝 튜토리얼](https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/#machine-learning) 코드와 [RL Collection](https://github.com/hcnoh/rl-collection-pytorch)을 기반으로 만들어졌습니다.

with continuous state and action spaces


### Youtube

Upload when model trained

### Simulation

Detailed instruciton followes [Deep Q-Learning](). Below is to launch node and run algorithm.

1. Gazebo 환경에 `월드`와 `머신`을 로드해주세요.

   ```
   roslaunch turtlebot3_gazebo turtlebot3_dqn_stage_2.launch
   ```

1. 알고리즘을 동작시켜주세요.
   ```
   roslaunch dqn_ttb turtlebot3_dqn_stage_30.launch
   ```


### TODO

* "steps" dosent go 0 at early end.
* evaluation code is not made.
* Sophisticate reward function may be required.
* Code is not yet trained yet.