import numpy as np
import torch
from torch import nn, Tensor
from typing import Optional, Callable

import utils

EPSILON = 1e-10

def get_returns(rollout_buffer: utils.RolloutBuffer, discount_factor=0.95):
    """ Calculate the returns, given the rollout buffer. The rollout buffer should contain some number of rollouts,
        concatenated together.
        Inputs:
            rollout_buffer: utils.RolloutBuffer
                Contains the rollout buffer. Note that this contains multiple rollouts, so you need to re-initialize
                the reward for terminal states.
                That is, if rollout_buffer.terminateds[i] is True, then rollout_buffer.rewards[i] is the terminal reward
                for a rollout, and index i+1 starts the recording of a new rollout.
            discount_factor: float
                Gamma. Multiplied to the reward geometrically (see notebook)
        Outputs:
            torch.Tensor[torch.float32], timesteps x 1
                Returns for the entire set of rollouts
    """
    # YOUR CODE HERE
    # raise NotImplementedError()

    # initialize empty reward return list 
    rewardReturn = []

    maxT = float('inf')
    # iterate through all time steps
    for time in range (rollout_buffer.rewards.size()[0]):
        # find the next maxT if necessary
        if maxT == float('inf'):
            # loop to find maxT
            temp = time 
            while(rollout_buffer.terminateds[temp] == False):
                temp = temp + 1
            maxT = temp

        # find the reward of the current time step
        i = time
        runningSum = 0
        while i <= maxT:
            runningSum = runningSum + rollout_buffer.rewards[i] * (discount_factor**(i - time))
            i = i + 1
        rewardReturn.append([runningSum])

        # if this timestep is the terminal reward for the rollout, reset maxT to inf to start the next rollout
        if rollout_buffer.terminateds[time] == True:
            maxT = float('inf')

    return torch.FloatTensor(rewardReturn)

def get_advantages(value_net: nn.Module,
                   observations: torch.Tensor,
                   returns: torch.Tensor):
    """ Get the advantages for the given rollout buffer.
        Inputs:
            value_net: nn.Module
                value net (critic) to get the training loss for
            observations: torch.Tensor[torch.float32], batch x obs_dim
                Contains the observations
            returns: torch.Tensor[torch.float32], batch x 1.
                Future returns associated with the given observations
        Outputs:
            torch.Tensor[torch.float32], singleton
                Value network loss for the given returns
    """
    # YOUR CODE HERE
    # Remember to use torch.no_grad!
    # You should calculate the advantage, then standardize it (subtract out mean, then divide by standard deviation
    # plus epsilon.) Use 1e-10 for epsilon (defined as EPSILON at top of file). Epsilon is solely there to prevent
    # divide-by-zero errors.
    # raise NotImplementedError()

    # calculate V(st) or mean future returns
    with torch.no_grad():
        meanFutureReturn = value_net(observations)
    
    # calculate advantage 
    advantage = returns - meanFutureReturn

    # standardize the advantage
    standardizedAdvantage = (advantage - torch.mean(advantage)) / (torch.std(advantage) + EPSILON)

    return standardizedAdvantage

def get_value_net_loss(value_net: nn.Module,
                       observation: Tensor,
                       returns: Tensor,
                       **kwargs):
    """ Get the training loss for the value network V(s_t)
        Inputs:
            value_net: nn.Module
                value net (critic) to get the training loss for
            observation: torch.Tensor[torch.float32], batch x obs_dim
                Observation from the environment, given to the policy
            returns: torch.Tensor[torch.float32], batch x 1.
                Future returns associated with the given observations
        Outputs:
            torch.Tensor[torch.float32], singleton
                Value network loss for the given returns
    """
    # YOUR CODE HERE
    # raise NotImplementedError()

    # define MSE loss (arguments are input/training loss and target/returns)
    MSEloss = nn.MSELoss() 

    return MSEloss(value_net(observation), returns)

def get_vanilla_policy_gradient_loss(policy: nn.Module,
                                observation: Tensor,
                                action: Tensor,
                                return_or_advantage: Tensor,
                                **kwargs):
    """ Get the return-based policy gradient loss for a minibatch. Each batch element represents a different timestep
        Inputs:
            policy: nn.Module
                Policy to get the policy gradient loss for
            observation: torch.Tensor[torch.float32], batch x obs_dim
                Observation from the environment, given to the policy
            action: torch.Tensor[torch.float32], batch x 1
                Action taken in the rollout for the associated observation
            return_or_advantage: torch.Tensor[torch.float32], batch x 1.
                Future return, if taking associated action, or separately computed advantage
        Outputs:
            torch.Tensor[torch.float32], singleton
                Vanilla policy gradient loss for the given return or advantage
    """
    # YOUR CODE HERE
    # raise NotImplementedError()

    # find the log of actions and states
    policyLog = policy(observation)

    # select the log values that are used (based on which action taken in each state)
    selectedLog = torch.tensor([policyLog[i][act].item() for i, act in enumerate(action)], requires_grad=True)
    selectedLog = selectedLog.to(return_or_advantage.dtype) # convert to float tensor to avoid dot product error

    # calculate the final value (make return_or_advantage single dimension in order to perform dot product)
    finalValue = (torch.dot(selectedLog, return_or_advantage.squeeze(dim=1))) / policyLog.size()[0]

    return -finalValue 

def collect_rollouts(env: utils.EnvInterface,
                     policy: nn.Module,
                     num_rollouts: int,
                     base_rollout: int = 0, # for logging only
                     num_total_rollouts: int = None, # for logging only
                     seed = None):
    num_total_rollouts = num_rollouts if num_total_rollouts is None else num_total_rollouts
    rollout_buffer = utils.RolloutBuffer()
    final_reward_mean = []
    for i in range(num_rollouts):
        print(f"Runing rollout {base_rollout + i}/{num_total_rollouts}", end="\r")
        obs = env.reset()
        terminated = False
        policy.eval() # Put the policy in eval mode
        while not terminated:
            # YOUR CODE HERE.
            # Steps:
            # 1) Evaluate the policy to get logits. Remember to use torch.no_grad here!
            # 2) Sample an action based on the logits. Use the provided line for this! The seed is important for grading.
            #       Provided line: action = utils.distribution_sample(logits, seed=seed)
            # 3) Execute the action on the environment with env.step
            # 4) Save the step to the rollout buffer with rollout_buffer.add. Provide this method with
            #       the action taken, the network logit outputs, the observation, whether the environment terminated,
            #       and the reward
            # 5) Think about what the observation should be for the next step.
            # Note: final_reward_mean is not required for grading, but not having breaks the notebook. Variable reward
            # should be the last reward of the rollout for it to work.
            # raise NotImplementedError()

            # evaluate the policy to get logits
            with torch.no_grad():
                logits = policy(obs)

            # sample an action based on the logits
            action = utils.distribution_sample(logits, seed=seed)

            # execute the action on the environment with env.step (step outputs observation, whether the environment has terminated, and reward)
            observation, terminated, reward = env.step(action)

            # save the step to the rollout buffer (function header rollout_buffer.add(self, action, logits, observation, terminated, reward))
            rollout_buffer.add(action, logits, obs, terminated, reward)

            # update observation
            obs = observation

            # END YOUR CODE
        final_reward_mean.append(reward)
    policy.train() # Put the policy back in train mode
    rollout_buffer.finalize()

    return rollout_buffer, np.mean(final_reward_mean)

def train_policy_gradient(env: utils.EnvInterface,
                          policy: nn.Module,
                          optimizer: torch.optim.Optimizer,
                          get_policy_gradient_loss: Callable,
                          get_returns: Callable,
                          lr_sched: torch.optim.lr_scheduler.LRScheduler = None,
                          rollouts: int = 1,
                          rollouts_before_training: int = 1,
                          training_epochs_per_rollout: int = 1,
                          minibatch_size: int = 64,
                          ppo_clip: float = 0.2,
                          critic_loss_multiplier: float = 0.0,
                          value_net: Optional[nn.Module] = None,
                          get_advantages: Optional[Callable] = None,
                          get_value_net_loss: Optional[Callable] = lambda **kwargs: torch.tensor([0.]),
                          rollout_seed=None):
    """ Trains policy gradient for the given number of rollouts.
        Inputs:
            env: EnvInterface
                Environment. Has methods reset() and step(). reset outputs observation only, while step outputs
                observation and reward alongside whether the environment has terminated.
                All observations are returned as 1 x obs_dim torch tensors, and actions should be 1 x action_dim
                torch tensors.
            policy: nn.Module
                policy to train
            optimizer:
                optimizer for the policy
            get_policy_gradient_loss: Callable
                Function called on a collection of kwargs (provided) that calculates the policy gradient
            get_returns: Callable
                Function called on a RolloutBuffer that gives the total return at each timestep
            lr_sched:
                lr scheduler for the optimizer
            rollouts: int
                Number of rollouts (reset call to step returning a termination) to run
            rollouts_before_training: int
                Number of rollouts to collect before training
            training_epochs_per_rollout: int
                Number of training epochs to run for one rollout
            minibatch_size: int
                Minibatch size for training
            ppo_clip: float
                clip constant passed through to the ppo loss
            value_net: nn.Module
                Value estimation network, optional.
            get_advantages: Callable
                Function called on the value net, a RolloutBuffer, and a list of returns that gives the advantage
                at each timestep
            get_value_net_loss: Callable
                Function called on a collection of kwargs (provided) that calculates the policy gradient
            rollout_seed
                Passthrough to collect_rollouts. For grading purposes.
    """
    r = 0
    losses_actor = []
    losses_critic = []
    final_rewards = []
    lr = []
    while r < rollouts:
        rollout_buffer, final_reward = collect_rollouts(env, policy, rollouts_before_training, r, rollouts, seed=rollout_seed)
        r += rollouts_before_training
        returns = get_returns(rollout_buffer)
        advantages = returns # added to avoid error of referencing before declaration
        final_rewards.append(final_reward)
        if get_advantages:
            advantages = get_advantages(value_net, rollout_buffer.observations, returns)
        for _ in range(training_epochs_per_rollout):
            idxr_base = np.arange(rollout_buffer.final_size)
            np.random.shuffle(idxr_base)
            for batch_start in range(0, rollout_buffer.final_size, minibatch_size):
                batch_stop = min(batch_start + minibatch_size, rollout_buffer.final_size)
                # YOUR CODE HERE
                # Remember to fill out collect_rollouts() as well!
                # Fill out the items in the policy gradient kwargs dict. Slice the lists in the rollout buffer
                # and/or returns with the batch. return_or_advantage should be advantages if get_advantages is provided
                # (i.e. not None), or returns otherwise.
                # If you have not reached the Advantage section yet, do not worry about this, and just provide correctly
                # sliced returns.
                # Everything should be a torch tensor, as specified by the inputs to get_PPO_policy_gradient_loss and
                # get_vanilla_policy_gradient_loss

                # raise NotImplementedError()
                policy_gradient_kwargs = dict(
                    policy=                 policy, # Fill in 
                    value_net=              value_net, # Fill in
                    critic=                 None, # Fill in
                    observation=            rollout_buffer.observations[batch_start:batch_stop], # Fill in
                    old_logits=             rollout_buffer.old_logits[batch_start:batch_stop], # Fill in
                    action=                 rollout_buffer.actions[batch_start:batch_stop], # Fill in
                    return_or_advantage=    advantages[batch_start:batch_stop], # Fill in
                    returns=                returns[batch_start:batch_stop], # Fill in 
                    ppo_clip=               ppo_clip
                )

                # END YOUR CODE
                optimizer.zero_grad()
                loss_actor = get_policy_gradient_loss(**policy_gradient_kwargs)
                loss_critic = get_value_net_loss(**policy_gradient_kwargs)
                loss = loss_actor + loss_critic * critic_loss_multiplier
                loss.backward()
                optimizer.step()
                losses_actor.append(loss_actor.detach().numpy())
                losses_critic.append(loss_critic.detach().numpy())                
                if lr_sched is not None:
                    lr_sched.step()
                    lr.append(optimizer.param_groups[0]['lr'])
    return losses_actor, losses_critic, final_rewards, lr

def get_PPO_policy_gradient_loss (policy: nn.Module,
                            observation: Tensor,
                            old_logits: Tensor,
                            action: Tensor,
                            return_or_advantage: Tensor,
                            ppo_clip = 0.2,
                            **kwargs):
    """ Get the return-based policy gradient loss for a minibatch. Each batch element represents a different timestep
        Inputs:
            policy: nn.Module.
                Policy to get the policy gradient loss for
            observation: torch.Tensor[torch.float32], batch x obs_dim
                Observation from the environment, given to the policy
            old_logits: torch.Tensor[torch.float32], batch x action_dim
                Logits output by the policy during runtime
            action: torch.Tensor[torch.float32], batch x 1
                Action taken in the rollout for the associated observation
            return_or_advantage: torch.Tensor[torch.float32], batch x 1.
                Separately computed advantage for the associated action
            ppo_clip: float
                PPO clipping epsilon on the probability ratio
    """
    # YOUR CODE HERE
    # raise NotImplementedError()

    # find the newLogits
    newLogits = policy(observation)

    # # select the log values that are used (based on which action taken in each state)
    newLogits = torch.tensor([newLogits[i][act].item() for i, act in enumerate(action)], requires_grad=True)
    old_logits = torch.tensor([old_logits[i][act].item() for i, act in enumerate(action)], requires_grad=True)

    # find the ratio of newLogits/oldLogits (subtract then take exponent since original values were log; if a = e^ln(a) and b = e^ln(b), then a/b = e^(ln(a))/e^(ln(b)) = e^(ln(a) - ln(b))
    ratio = torch.exp(newLogits - old_logits)

    # calculate the final value (make return_or_advantage single dimension in order to perform dot product); make sure to divide by batch size
    finalValue = min(torch.dot(ratio, return_or_advantage.squeeze(dim=1)), torch.dot(torch.clip(ratio, 1 - ppo_clip, 1 + ppo_clip), return_or_advantage.squeeze(dim=1))) / newLogits.size()[0]

    return -finalValue 