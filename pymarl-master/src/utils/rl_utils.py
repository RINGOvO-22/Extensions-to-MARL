import torch as th


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes

    # 初始化返回值张量（ret）：该张量的形状与目标Q值（target_qs）相同，用零初始化。最后一个时间步的值特别设置为目标Q值乘以（1减去每个序列是否结束的总和），这样可以确保只有未结束的序列才会被考虑。
    # 我的理解：
    # th.sum(terminated, dim=1) 的形状为 B1, 其中的每个元素表示相应序列在所有时间步中结束的总次数
    # 只有未结束的 episode 的 (1 - th.sum(terminated, dim=1) 才是0，所以ret中对应行的最后一个值是 target_qs[:, -1] * 1，否则就是<=0的
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))

    # Backwards  recursive  update  of the "forward  view"
    # 反向递归更新：从倒数第二个时间步开始，反向遍历每个时间步。每一步的λ返回值是基于下一时间步的λ返回值、当前的奖励、当前步的目标Q值，以及是否结束的标记计算得到。]
    # 具体来说，每一步的计算考虑了TD(λ)目标的“向前视角”，通过混合即时奖励、未来的预期奖励（通过目标Q值表示），以及通过λ参数平衡的下一步的λ返回值。
    for t in range(ret.shape[1] - 2, -1, -1):
        # 计算TD(λ)目标：通过这种方式，算法有效地将未来奖励的信息反向传播回更早的时间步，同时考虑了每一步是否结束和是否有效（通过mask）。
        # 这种方法既利用了即时奖励信息，也考虑了长期奖励的预期价值，通过λ参数平衡了这两者的贡献。
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))

    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    # 返回结果：函数最终返回从时间步0到T-1的λ返回值张量，形状为BT-1A，这正好对应于输入的目标Q值张量少一个时间步的情况。
    # return ret[:, 0:-1]
    return ret


# 好像ret是从第一个时间步到倒数第二个，target_qs 是从第二个时间步到最后一个, reward也是从第一个时间步到倒数第二个
# 这样的话reward的从第x到x+n-1个值用于计算第x个ret, target_qs的第x+n-1个值被用于计算第x个ret

def build_td_n_targets(rewards, terminated, mask, target_qs, n_agents, gamma, step_n):
    # Assumes <target_qs> in B*T*A and <reward>, <terminated>, <mask> in (at least) B*T-1*1
    # Initialize last n-step return for not terminated episodes

    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - terminated[:, -1])

    # from the first timestep to step T-n
    for t in range(1, ret.shape[1] - step_n):
        if terminated[:, t].any() or mask[:, t].any() == 0:
            continue
        n_step_return = rewards[:, t]
        for i in range(1, step_n):
            if terminated[:, t + i].any():
                break
            n_step_return += (gamma ** i) * mask[:, t + i] * rewards[:, t + i] * (1 - terminated[:, t + i])
        if i == step_n - 1:
            n_step_return += (gamma ** step_n) * target_qs[:, t + step_n - 1] \
                             * (1 - terminated[:, t + step_n])  # Add target_qs from t+n-1
        ret[:, t] = n_step_return

    # the last n step
    for t in range(ret.shape[1] - step_n, ret.shape[1]):
        n_step_return = rewards[:, t]
        # Accumulate rewards from t + 1 to t + n - 1
        for i in range(1, ret.shape[1] - t):
            n_step_return += (gamma ** i) * mask[:, t + i] * rewards[:, t + i] * (1 - terminated[:, t + i])
        ret[:, t] = n_step_return

    # Returns n-step return from t=0 to t=T-1
    return ret
