# 2. Trust Region Policy Optimization
코드는 로보티즈의 [머신러닝 튜토리얼](https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/#machine-learning) 코드와 [RL Collection](https://github.com/hcnoh/rl-collection-pytorch)을 기반으로 만들어졌습니다.

with continuous state and action spaces


### About "Reinforcement Learning" and "Reward Function"

All of the Reinforcement Learning requires a good reward function.

If reward is not given appropriately, [this](https://youtu.be/R5-SdIQ1RFQ) will happen. I've trained model for half a day but it dose not work well.

The reward function was 

    `reward = round(8 * (self.goal_distance - current_distance) * min_scan_range, 2)`

, which calculates distance between robot and goal point and distance between robot and closest obstacle in map.

Poor reward function may gives positive reward even if the robot rotates in one place(where local minma is).

Setting time out might solve the problem, but not enough to solve(mine like above).

Taking delta of distance can be solution, since it incentivizes the robot to move towards the goal.

The changed reward function is

    `delta_distance = self.previous_distance - current_distance
        self.previous_distance = current_distance

        reward = max(0.1, abs(normalized_distance)) * delta_distance * 100`


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