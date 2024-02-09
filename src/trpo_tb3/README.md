# 2. Trust Region Policy Optimization
코드는 로보티즈의 [머신러닝 튜토리얼](https://emanual.robotis.com/docs/en/platform/turtlebot3/machine_learning/#machine-learning) 코드와 [RL Collection](https://github.com/hcnoh/rl-collection-pytorch)을 기반으로 만들어졌습니다.

TRPO는 대표적은 Policy Based Learning 알고리즘으로 앞선 DQN처럼 Q-value를 maximize하는 것이 아닌 `값의 미분치`를 이용하므로 동일한 조건에서 DQN보다 더 빠른 학습시간을 기대할 수 있습니다. 또한 Trust Region이란 이름처럼 학습이후 무조건 성능이 좋다고 합니다.

하지만 코드에서 DQN은 몇 번의 반복으로도 쉽게 학습하는 것으로 보이는데, 이는 단순히 DQN의 action space를 discrete하게 설정하여 알고리즘이 탐색해야할 경우의 수를 많이 줄였기 때문입니다.

TRPO는 continuous state space, continuous action spcae 에서도 좋은 학습성능을 보일 것으로 기대했기 때문에, DQN보다 가혹한 환경에서 학습을 하고 이에 따라 DQN 보다는 성능이 떨어져 보일 수 있으나, 학습환경이 달라 1대1로 비교할 수 업습니다.

대신 이로써 discrete action 과 continuous action의 차이를 느낄 수 있을 것입니다.


### Youtube

Upload when model trained

### Simulation

기본적인 작동방식은 [Deep Q-Learning]()과 동일 합니다. 따라서 노드와 알고리즘 실행만 적어두겠습니다.

1. Gazebo 환경에 `월드`와 `머신`을 로드해주세요.

   ```
   roslaunch turtlebot3_gazebo turtlebot3_dqn_stage_2.launch
   ```

1. 알고리즘을 동작시켜주세요.
   ```
   roslaunch dqn_ttb turtlebot3_dqn_stage_30.launch
   ```

---

### About "Reinforcement Learning" and "Reward Function"

모든 강화학습 알고리즘은 reward function 이라는 개념을 필수적으로 요구합니다. 이는 stage와 action의 입력에 reward라는 값을 반환합니다. 이런 반환된 보상을 바탕으로 알고리즘이 학습을 하므로 목표에 맞춰 엄밀하게 짜여진 보상함수는 좋은 성능의 뒷받침이 됩니다.

상황에 적당하지 않은 보상은 [다음 영상](https://youtu.be/R5-SdIQ1RFQ)과 같이 약 반나절을 학습에도 불구하고 어설픈 동작을 야기합니다.

위의 영상에서 사용된 보상함수는 다음과 같습니다.

    `reward = round(8 * (self.goal_distance - current_distance) * min_scan_range, 2)`

목표위치와의 거리를 계산하고 근처 장애물과의 거리를 고려하여 값을 반환하는 함수입니다.

영상을 자세히 보신다면 제자리에서 도는 행위로 소소한 보상을 모으며 local minima에 빠진 모습을 볼 수 있습니다.

(TRPO의 성능 향상을 위해 reward function을 수정하고 있습니다. 추후 업데이트 됩니다.)

<del>정해진 시간 내에 목표위치에 도달하지 않는 상황에 패널티를 주어 해결할 수 있지만, 위 영상을 보니 충분히 해결하지 못 한 것 같습니다.

<del>따라서 조금 더 엄밀한 보상함수를 설계한다면 더 나은 성능을 기대할 수 있게 되고, 따라서 목표위치까지의 거리에 대한 미분치를 이용하여 로봇이 목표위치로 향해 움직일 때 그리고 목표위치를 향해 빠르게 움직일 때 더 큰 보상을 받도록 할 수 있습니다.

<del>수정된 보상함수는 다음과 같습니다.

    `delta_distance = self.previous_distance - current_distance
        self.previous_distance = current_distance

        reward = max(0.1, abs(normalized_distance)) * delta_distance * 100`

### Imitation Learning

보상함수를 수정하는 방식으로 성능을 좋게 할 순 있지만, 잘 동작하는 알고리즘을 바탕으로 다른 알고리즘도 학습시킬 수 있다면 좋을 것 같습니다.

그러한 맥락에서 발전한 알고리즘이 모방학습입니다. 모방학습은 단순히 알고리즘을 학습시키는 의의뿐만 아니라 교시데이터만으로도 학습시킬 수 있다는 잠재력이 풍부한 알고리즘입니다.

Generative Adversarial Imiation Learning도 구현해놓았으니 자세한 내용은 해당 마크다운 파일에 적어놓겠습니다.

