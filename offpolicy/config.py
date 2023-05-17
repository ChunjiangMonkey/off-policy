import argparse
import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description="OFF-POLICY", formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    # 算法名
    parser.add_argument("--algorithm_name", type=str, default="rmatd3", choices=[
        "rmatd3", "rmaddpg", "rmasac", "qmix", "vdn", "matd3", "maddpg", "masac", "mqmix", "mvdn"])
    # 实验名
    parser.add_argument("--experiment_name", type=str, default="check")
    # 种子
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for numpy/torch")
    # 使用cuda
    parser.add_argument("--cuda", action='store_false', default=True)
    # cuda确定性
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True)
    # torch线程
    # torch.set_num_threads的参数
    parser.add_argument('--n_training_threads', type=int,
                        default=1, help="Number of torch threads for training")
    # 一次训练的并行环境数量
    parser.add_argument('--n_rollout_threads', type=int, default=1,
                        help="Number of parallel envs for training rollout")

    # 一次评估的并行环境数量
    parser.add_argument('--n_eval_rollout_threads', type=int, default=1,
                        help="Number of parallel envs for evaluating rollout")

    # 每个环境训练的step
    # 不清楚这个step值的是不是episode
    parser.add_argument('--num_env_steps', type=int,
                        default=2000000, help="Number of env steps to train for")
    # 是否使用wandb
    parser.add_argument('--use_wandb', action='store_false', default=False,
                        help="Whether to use weights&biases, if not, use tensorboardX instead")

    # 跑程序的用户
    parser.add_argument('--user_name', type=str, default="zoeyuchao")

    # env parameters
    # 要跑的环境名
    parser.add_argument('--env_name', type=str, default="MPE")

    # 是否使用局部观测代替全局状态
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    # 每个episode的长度
    parser.add_argument('--episode_length', type=int,
                        default=80, help="Max length for any episode")
    # buffer大小
    parser.add_argument('--buffer_size', type=int, default=5000,
                        help="Max # of transitions that replay buffer can contain")
    # 是否标准化收益
    parser.add_argument('--use_reward_normalization', action='store_true',
                        default=False, help="Whether to normalize rewards in replay buffer")

    # 不太清楚这个popart是什么意思，看help是标准化loss的工具
    parser.add_argument('--use_popart', action='store_true', default=False,
                        help="Whether to use popart to normalize the target loss")

    # popart更新的频率
    parser.add_argument('--popart_update_interval_step', type=int, default=2,
                        help="After how many train steps popart should be updated")

    # prioritized experience replay

    # 是否使用prioritized experience replay
    parser.add_argument('--use_per', action='store_true', default=False,
                        help="Whether to use prioritized experience replay")
    # PER权重形成中最大TD误差的权重
    parser.add_argument('--per_nu', type=float, default=0.9,
                        help="Weight of max TD error in formation of PER weights")
    # PER中的alpha
    parser.add_argument('--per_alpha', type=float, default=0.6,
                        help="Alpha term for prioritized experience replay")
    # PER中的eps
    parser.add_argument('--per_eps', type=float, default=1e-6,
                        help="Eps term for prioritized experience replay")
    # PER中的beta的start
    parser.add_argument('--per_beta_start', type=float, default=0.4,
                        help="Starting beta term for prioritized experience replay")

    # network parameters
    # 是否使用中心化的Q函数
    parser.add_argument("--use_centralized_Q", action='store_false',
                        default=True, help="Whether to use centralized Q function")
    # agent是否共享策略参数
    parser.add_argument('--share_policy', action='store_false',
                        default=True, help="Whether agents share the same policy")
    # actor/critic网络的hidden_size大小
    parser.add_argument('--hidden_size', type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks")
    #  actor/critic网络hidden层的数量
    parser.add_argument('--layer_N', type=int, default=1,
                        help="Number of layers for actor/critic networks")
    # 是否使用relu激活层
    parser.add_argument('--use_ReLU', action='store_false',
                        default=True, help="Whether to use ReLU")
    # 是否将特征标准化
    parser.add_argument('--use_feature_normalization', action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    # 是否对权重使用正交初始化，对偏差使用0初始化（不懂）
    parser.add_argument('--use_orthogonal', action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    # 最后一个动作层 gain（不懂）
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")
    # 是否使用1d卷积
    parser.add_argument("--use_conv1d", action='store_true',
                        default=False, help="Whether to use conv1d")
    # hidden layer的维度（但是为什么要叫stacked_frames呢）
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")

    # recurrent parameters
    # 是否将之前的动作作为输入的一部分
    parser.add_argument('--prev_act_inp', action='store_true', default=False,
                        help="Whether the actor input takes in previous actions as part of its input")
    # 是否使用循环神经网络
    parser.add_argument("--use_rnn_layer", action='store_false',
                        default=True, help='Whether to use a recurrent policy')
    # 是否使用原生神经网络
    parser.add_argument("--use_naive_recurrent_policy", action='store_false',
                        default=True, help='Whether to use a naive recurrent policy')
    # TODO now only 1 is support
    # 循环N（不懂）
    parser.add_argument("--recurrent_N", type=int, default=1)

    # 用于通过BPTT进行训练的组块的时间长度（不懂）
    parser.add_argument('--data_chunk_length', type=int, default=80,
                        help="Time length of chunks used to train via BPTT")
    # RNN训练的磨合时间长度，见R2D2论文（不懂）
    parser.add_argument('--burn_in_time', type=int, default=0,
                        help="Length of burn in time for RNN training, see R2D2 paper")

    # attn parameters
    # 是否使用attention
    parser.add_argument("--attn", action='store_true', default=False)
    # attention N
    parser.add_argument("--attn_N", type=int, default=1)
    # attention 的大小
    parser.add_argument("--attn_size", type=int, default=64)
    # attention的head数
    parser.add_argument("--attn_heads", type=int, default=4)
    # dropout
    parser.add_argument("--dropout", type=float, default=0.0)
    # 使用平均池
    parser.add_argument("--use_average_pool",
                        action='store_false', default=True)
    # 使用自连接
    parser.add_argument("--use_cat_self", action='store_false', default=True)

    # optimizer parameters
    # Adam的学习率
    parser.add_argument('--lr', type=float, default=5e-4,
                        help="Learning rate for Adam")
    # RMSprop的优化epsilon
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    # 优化器的weight_decay
    parser.add_argument("--weight_decay", type=float, default=0)

    # algo common parameters
    # batch的大小
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Number of buffer transitions to train on at once")

    # 折扣因子
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Discount factor for env")

    # 是否使用最大梯度正则化
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True)
    # 梯度正则化（不懂）
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')

    # 使用huber loss
    parser.add_argument('--use_huber_loss', action='store_true',
                        default=False, help="Whether to use Huber loss for critic update")
    parser.add_argument("--huber_delta", type=float, default=10.0)

    # soft update parameters
    # 使用soft update
    parser.add_argument('--use_soft_update', action='store_false',
                        default=True, help="Whether to use soft update")
    # Polyak更新的一个参数
    parser.add_argument('--tau', type=float, default=0.005,
                        help="Polyak update rate")
    # hard update parameters
    # target更新的episode频率
    parser.add_argument('--hard_update_interval_episode', type=int, default=200,
                        help="After how many episodes the lagging target should be updated")
    # target更新的timestep频率
    parser.add_argument('--hard_update_interval', type=int, default=200,
                        help="After how many timesteps the lagging target should be updated")
    # rmatd3 parameters
    # noise的标准差
    parser.add_argument("--target_action_noise_std", default=0.2, help="Target action smoothing noise for matd3")
    # rmasac parameters
    # 初始温度
    parser.add_argument('--alpha', type=float, default=1.0,
                        help="Initial temperature")
    # 目标熵相关系数
    parser.add_argument('--target_entropy_coef', type=float,
                        default=0.5, help="Initial temperature")
    # 是否使用中心化的critic
    parser.add_argument('--automatic_entropy_tune', action='store_false',
                        default=True, help="Whether use a centralized critic")
    # qmix parameters
    parser.add_argument('--use_double_q', action='store_false',
                        default=True, help="Whether to use double q learning")
    parser.add_argument('--hypernet_layers', type=int, default=2,
                        help="Number of layers for hypernetworks. Must be either 1 or 2")
    parser.add_argument('--mixer_hidden_dim', type=int, default=32,
                        help="Dimension of hidden layer of mixing network")
    parser.add_argument('--hypernet_hidden_dim', type=int, default=64,
                        help="Dimension of hidden layer of hypernetwork (only applicable if hypernet_layers == 2")

    # exploration parameters
    # 使用随机动作添加到buffer的episode数目
    parser.add_argument('--num_random_episodes', type=int, default=5,
                        help="Number of episodes to add to buffer with purely random actions")
    # epsilon-greedy探索的epsilon初始值
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help="Starting value for epsilon, for eps-greedy exploration")
    # epsilon-greedy探索的epsilon终值
    parser.add_argument('--epsilon_finish', type=float, default=0.05,
                        help="Ending value for epsilon, for eps-greedy exploration")
    # 达到epsilon终值的episode数量, 这决定了探索降低的速度
    parser.add_argument('--epsilon_anneal_time', type=int, default=50000,
                        help="Number of episodes until epsilon reaches epsilon_finish")
    # 动作噪声
    parser.add_argument('--act_noise_std', type=float,
                        default=0.1, help="Action noise")

    # train parameters
    # 每更新多少次critic, 更新一次actor
    parser.add_argument('--actor_train_interval_step', type=int, default=2,
                        help="After how many critic updates actor should be updated")
    # 对actor和critic更新一次中间所隔的step数
    parser.add_argument('--train_interval_episode', type=int, default=1,
                        help="Number of env steps between updates to actor/critic")
    # 对actor和critic更新一次中间所隔的episode数
    parser.add_argument('--train_interval', type=int, default=100,
                        help="Number of episodes between updates to actor/critic")
    # 使用价值激活掩码 (应该是一个trick)
    parser.add_argument("--use_value_active_masks",
                        action='store_true', default=False)

    # eval parameters
    # 是否使用执行模式
    parser.add_argument('--use_eval', action='store_false',
                        default=True, help="Whether to conduct the evaluation")
    # 每多少episode对policy进行一次评估
    parser.add_argument('--eval_interval', type=int, default=10000,
                        help="After how many episodes the policy should be evaled")
    # 每次评估policy需要多少episode的数据
    parser.add_argument('--num_eval_episodes', type=int, default=32,
                        help="How many episodes to collect for each eval")

    # save parameters
    # 训练时每多少episode保存一次模型
    parser.add_argument('--save_interval', type=int, default=100000,
                        help="After how many episodes of training the policy model should be saved")

    # log parameters
    # 训练时每多少episode输出一次训练信息
    parser.add_argument('--log_interval', type=int, default=1000,
                        help="After how many episodes of training the policy model should be saved")

    # pretained parameters
    # 保存和加载模型的路径
    parser.add_argument("--model_dir", type=str, default=None)

    return parser
