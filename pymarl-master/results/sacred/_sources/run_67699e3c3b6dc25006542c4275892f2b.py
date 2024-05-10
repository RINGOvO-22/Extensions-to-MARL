import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

# run.py 文件中run函数的主要作用是构建实验参数变量 args 以及一个自定义 Logger 类的记录器 logg
def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)
    # 内置变量_config的拷贝作为参数传入到了run函数中，_config 是字典变量，因此查看参数时，需要利用 _config[key]=value，
    # 在 run 函数中，构建了一个namespace类的变量args，将_config中的参数都传给了 args，
    # 这样就可以通过args.key=value的方式查看参数了。
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    # pymarl自定义了一个utils.logging.Logger类的对象logger对ex的内置变量_run和_log进行封装，
    # 最终所有的实验结果通过 logger.log_stat(key, value, t, to_sacred=True) 记录在了./results/sacred/实验编号/info.json文件中。
    # 在整个实验中，logger主要对runner和learner两个对象所产生的实验数据进行了记录，包括训练数据和测试数据的如下内容：
    # runner对象：
    # 一系列环境特定的实验数据，即env_info，在SC2中，包括"battle_won_mean"，"dead_allies_mean"， "dead_enemies_mean";
    # 训练相关的实验数据：包括"ep_length_mean"，"epsilon"，"return_mean"；
    # learner对象：
    # 训练相关的实验数据："loss"，"grad_norm"，"td_error_abs" ，"q_taken_mean"，"target_mean"。
    logger = Logger(_log)

    # altered
    # logger.console_logger.setLevel(logger.INFO)

    _log.info("Experiment Parameters:")
    # pformat是调整输出格式的,嵌套会自动缩进
    # indent：指定缩进的空格数，默认为1。缩进用于表示嵌套对象的层次结构。
    # width：指定输出的最大宽度，默认为80。如果输出的字符串超过了这个宽度，会自动进行换行。
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

# 可以说整个run函数中最重要的就是run_sequential函数，其他的代码都是用来做log或者调整终端输出格式的语句

# run_sequential 是实验运行的主要函数，作用是首先是构建如下自定义类的对象：
    # EpisodeRunner类的环境运行器对象runner ，
    # ReplayBuffer类的经验回放池对象buffer，
    # BasicMAC类的智能体控制器对象mac，
    # QLearner类的智能体学习器对象learner，
    # 最后进行实验，即训练智能体，记录实验结果，定期测试并保存模型。
def run_sequential(args, logger):


    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    # 这一段都是从args和配置文件里面读设置，代码不用改，想改的话改配置文件就行了
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    # 从环境中收集轨迹用于离线训练，这个东西印象里做的也不多，一般没什么需要改的（就抽样嘛哪有什么需要改的）
    # buffer对象属于自定义的components.episode_buffer.ReplayBuffer(EpisodeBatch)类，该对象的主要作用是存储样本以及采样样本。
        # ReplayBuffer的父类是EpisodeBatch。EpisodeBatch类对象用于存储episode的样本，
        # ReplayBuffer(EpisodeBatch)类对象则用于存储所有的off-policy样本，
    # 也即EpisodeBatch类变量的样本会持续地补充到ReplayBuffer(EpisodeBatch)类的变量中。
    # 同样由于QMix用的是DRQN结构，因此EpisodeBatch与ReplayBuffer中的样本都是以episode为单位存储的。
    # 在EpisodeBatch中数据的维度是[batch_size, max_seq_length, *shape]，
    # EpisodeBatch中Batch Size表示此时batch中有多少episode，
    # ReplayBuffer类数据的维度是[buffer_size, max_seq_length, *shape]。
    # ReplayBuffer中episodes_in_buffer表示此时buffer中有多少个episode的有效样本。
    # max_seq_length则表示一个episode的最大长度。

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here

    '''
    这一段的代码和结构是更改的重点之一，因为最终执行的就是智能体和网络。
    智能体控制器和QMIX类的网络是最终的执行算法的代码，这里的代码初始化了智能体网络，
    许多智能体方面的改进都从此处进行更改操作。因此这里面的代码是重点。
    '''

    # mac对象中的一个重要属性就是nn.module类的智能体对象mac.agent，线性层，GRU，线性层，
        # 该对象定义了各个智能体的局部Q网络，即接收观测作为输入，输出智能体各个动作的隐藏层值和Q值。
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    # 运行环境，产生训练样本
    # runner对象中的一个重要属性就是env.multiagentenv.MultiAgentEnv类的环境对象runner.env，即环境，
    # 另一个属性是components.episode_buffer.EpisodeBatch类的episode样本存储器对象runner.batch，
        # 该对象用于以episode为单位存储环境运行所产生的样本。

    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    '''
    可以看到，最终mac也集成进了learner模块里面。learner的更新中有着QMIX模块和RNN智能体模块，
    最终的学习也在学习器里面进行，所以对于Q值网络的模拟的如Qtran的修改，都在此模块进行。重中之重。
    '''
    # Learner:
    # 该对象的主要作用是依据特定算法对智能体参数进行训练更新
    # 在QMix算法与VDN算法中，均有nn.module类的混合网络learner.mixer，
    # 因此learner对象需要学习的参数包括各个智能体的局部Q网络参数mac.parameters()，以及混合网络参数learner.mixer.parameters()，
    # 两者共同组成了learner.params，然后用优化器learner.optimiser进行优化。
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    # 如果有已经保存的模型，就读取此模型，接着训练
    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        # 如果路径不存在就会结束run函数，并报错路径不存在
        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # 检测路径文件夹下的文件夹（就是那些以数字命名的文件夹1，2，3，4，5，6......
        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        # 选择load哪个文件模型文件
        if args.load_step == 0:
            # choose the max timestep 选择最大步数的
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step 自定义选择离某个load_step最近的
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        # 获得模型路径
        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
        # 打印模型路径,输出
        logger.console_logger.info("Loading model from {}".format(model_path))
        # 加载模型
        learner.load_models(model_path)
        # 把模型步数,加载到环境步数上
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))


    # 进入训练循环，每个循环中，首先执行一个完整的 episode，并将 episode 存储到 replay buffer 中。然后，从 replay buffer 中抽样出一个 batch，并用于更新模型参数。
    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch) # 把batch给buffer即插入buffer

        # if buffer里面的batch足够抽样
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            # 将批处理截断为仅填充的时间步长
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # 得到 replaybuffer,进行训练
            learner.train(episode_sample, runner.t_env, episode)

        # 定期执行测试运行，以监控训练的进度和性能。在测试运行中，不会更新模型参数，仅用于评估当前模型的性能。
        # Execute test runs once in a while
        # n_test_runs是测试几轮的意思, 测试多少论取其胜率的平均值做胜率
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            # 时间消耗,Estimated,估计
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # 如果设置了保存模型参数的间隔，则会定期保存模型参数，以便在训练过程中定期备份模型。
        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        # 当达到最大训练时间时，结束训练。
        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # 最后，关闭环境，并输出训练结束信息
    runner.close_env()
    logger.console_logger.info("Finished Training")


# args_sanity_check用于检查test_nepisode是不是batch_size_run的整数倍。
# 但具体为啥一定得是batch_size_run的整数倍呢？因为测试的时候和训练的时候最好是整数倍，
# 比如训练用5batch，那测试的时候用5的倍数的batch测试才能最好地适应网络参数

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
