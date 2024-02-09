from collections import namedtuple, deque
import dill
import random
import os, time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .diffusion import Diffusion
from .model import MLP
from .helpers import EMA

from .env import Env


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Diffusion_QL(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        discount,
        tau,
        max_q_backup=False,
        eta=1.0,
        beta_schedule="linear",
        n_timesteps=100,
        ema_decay=0.995,
        step_start_ema=1000,
        update_ema_every=5,
        lr=3e-4,
        lr_decay=False,
        lr_maxt=1000,
        grad_norm=1.0,
    ):
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(
            state_dim=state_dim,
            action_dim=action_dim,
            model=self.model,
            beta_schedule=beta_schedule,
            n_timesteps=n_timesteps,
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(
                self.actor_optimizer, T_max=lr_maxt, eta_min=0.0
            )
            self.critic_lr_scheduler = CosineAnnealingLR(
                self.critic_optimizer, T_max=lr_maxt, eta_min=0.0
            )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {"bc_loss": [], "ql_loss": [], "actor_loss": [], "critic_loss": []}
        for _ in range(iterations):
            # Sample replay buffer as minbatch size
            temp_buffer = replay_buffer.sample(batch_size)
            state = temp_buffer[0].state
            action = temp_buffer[0].action
            next_state = temp_buffer[0].next_state
            reward = temp_buffer[0].reward
            for i in range(1, len(temp_buffer)):
                state = torch.cat((state, temp_buffer[i].state), 0)
                action = torch.cat((action, temp_buffer[i].action), 0)
                next_state = torch.cat((next_state, temp_buffer[i].next_state), 0)
                reward = torch.cat((reward, temp_buffer[i].reward), 0)

            # print("##################\n", state, "\n##################")

            # state, action, next_state, reward, not_done = replay_buffer.sample(
            #     batch_size
            # )

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1, target_q2 = self.critic_target(
                    next_state_rpt, next_action_rpt
                )
                target_q1 = target_q1.view(batch_size, 10, -1).max(dim=1, keepdim=True)[
                    0
                ]
                target_q2 = target_q2.view(batch_size, 10, -1).max(dim=1, keepdim=True)[
                    0
                ]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            target_q = (reward + self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
                current_q2, target_q
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=self.grad_norm, norm_type=2
                )
            self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.actor.loss(action, state)
            new_action = self.actor(state)

            q1_new_action, q2_new_action = self.critic(state, new_action)
            if np.random.uniform() > 0.5:
                q_loss = -q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = -q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=self.grad_norm, norm_type=2
                )
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar(
                        "Actor Grad Norm", actor_grad_norms.max().item(), self.step
                    )
                    log_writer.add_scalar(
                        "Critic Grad Norm", critic_grad_norms.max().item(), self.step
                    )
                log_writer.add_scalar("BC Loss", bc_loss.item(), self.step)
                log_writer.add_scalar("QL Loss", q_loss.item(), self.step)
                log_writer.add_scalar("Critic Loss", critic_loss.item(), self.step)
                log_writer.add_scalar(
                    "Target_Q Mean", target_q.mean().item(), self.step
                )

            metric["actor_loss"].append(actor_loss.item())
            metric["bc_loss"].append(bc_loss.item())
            metric["ql_loss"].append(q_loss.item())
            metric["critic_loss"].append(critic_loss.item())

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f"{dir}/actor_{id}.pth")
            torch.save(self.critic.state_dict(), f"{dir}/critic_{id}.pth")
        else:
            torch.save(self.actor.state_dict(), f"{dir}/actor.pth")
            torch.save(self.critic.state_dict(), f"{dir}/critic.pth")

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f"{dir}/actor_{id}.pth"))
            self.critic.load_state_dict(torch.load(f"{dir}/critic_{id}.pth"))
        else:
            self.actor.load_state_dict(torch.load(f"{dir}/actor.pth"))
            self.critic.load_state_dict(torch.load(f"{dir}/critic.pth"))


# def eval_policy(policy, env_name, eval_episodes=10):
#     eval_env = env.make(env_name)

#     scores = []
#     for _ in range(eval_episodes):
#         traj_return = 0.0
#         state, done = eval_env.reset(), False
#         while not done:
#             action = policy.sample_action(np.array(state))
#             state, reward, done, _ = eval_env.step(action)
#             traj_return += reward
#         scores.append(traj_return)

#     avg_reward = np.mean(scores)
#     std_reward = np.std(scores)

#     normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
#     avg_norm_score = eval_env.get_normalized_score(avg_reward)
#     std_norm_score = np.std(normalized_scores)

#     print(
#         "Evaluation over {} episodes: {}: {:.2f} {:.2f}".format(
#             eval_episodes, avg_reward, avg_norm_score
#         )
#     )
#     return avg_reward, std_reward, avg_norm_score, std_norm_score
