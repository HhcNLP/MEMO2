# 导入所需的库和模块
import copy # 用于复制对象
import datetime # 用于处理日期和时间
import json # 用于处理JSON数据
import logging # 用于记录日志信息
import os # 用于操作文件和目录
import sys # 用于与Python解释器进行交互
import time # 用于处理时间

import torch
from utils import factory # 自定义模块，工厂函数
from utils.data_manager import DataManager # 数据管理器
from utils.toolkit import ConfigEncoder, count_parameters, save_fc, save_model # 自定义工具函数

# 定义了一个train函数，用于训练神经网络模型
def train(args):
    # 复制一份种子(seed)列表和设备(device)信息
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    # 针对每个种子(seed)进行训练
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)

# 定义了一个_train函数，用于执行具体的训练任务
def _train(args):
    # 获取当前时间作为时间字符串
    time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    args['time_str'] = time_str
    # 根据参数创建实验名称
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    exp_name = "{}_{}_{}_{}_B{}_Inc{}".format(
        args["time_str"],
        args["dataset"],
        args["convnet_type"],
        args["seed"],
        init_cls, # 初始化学习类别
        args["increment"], # 增量学习的类别数
    )
    args['exp_name'] = exp_name

    # 创建日志文件的路径
    if args['debug']:
        logfilename = "logs/debug/{}/{}/{}/{}".format( 
            args["prefix"],
            args["dataset"],
            args["model_name"],
            args["exp_name"]
        )
    else:
        logfilename = "logs/{}/{}/{}/{}".format(
            args["prefix"],
            args["dataset"],
            args["model_name"],
            args["exp_name"]
        )

    args['logfilename'] = logfilename

    # 创建CSV文件名
    csv_name = "{}_{}_{}_B{}_Inc{}".format( 
        args["dataset"],
        args["seed"],
        args["convnet_type"],
        init_cls,
        args["increment"],
    )
    args['csv_name'] = csv_name
    os.makedirs(logfilename, exist_ok=True)

    log_path = os.path.join(args["logfilename"], "main.log")
    # 程序会将日志消息记录到指定的文件（main.log）中，并在控制台上显示相同的日志消息。
    # 这有助于跟踪程序的运行、诊断问题和记录有关程序执行的重要信息。根据需要，你可以根据实际需求来调整日志级别和格式。
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=log_path),
            logging.StreamHandler(sys.stdout), # 在控制台上显示相同的日志消息
        ],
    )
    # 记录当前时间
    logging.info(f"Time Str >>> {args['time_str']}")
    # 保存配置文件
    config_filepath = os.path.join(args["logfilename"], 'configs.json')
    with open(config_filepath, "w") as fd:
            json.dump(args, fd, indent=2, sort_keys=True, cls=ConfigEncoder)
    # 设置随机种子
    _set_random()
    # 设置计算设备
    _set_device(args)
    # 打印参数信息
    # print_args(args)
    # 创建数据管理器
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    # 创建模型
    model = factory.get_model(args["model_name"], args)

    # 初始化性能指标曲线和相关变量
    cnn_curve, nme_curve, no_nme = {"top1": [], "top5": []}, {"top1": [], "top5": []}, True
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime('%m%d-%H-%M-%S-%f')[:-3]
    logging.info(f"Start time:{start_time}")
    # 循环训练每个任务
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )

        # 增量式训练模型
        model.incremental_train(data_manager)
        if task == data_manager.nb_tasks-1:
            cnn_accy, nme_accy = model.eval_task(save_conf=True)
            no_nme = True if nme_accy is None else False
        else:
            cnn_accy, nme_accy = model.eval_task(save_conf=False)
        model.after_task()

        # 记录性能指标
        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

    # 记录训练结束时间
    end_time = datetime.datetime.now()
    end_time_str = end_time.strftime('%m%d-%H-%M-%S-%f')[:-3]
    logging.info(f"Start Time:{start_time_str}")
    logging.info(f"End Time:{end_time_str}")
    cost_time = (end_time - start_time)/3600
    print("花费的时间（小时）：", cost_time)
    # 保存训练时间
    save_time(args, cost_time)
    # 保存性能指标
    save_results(args, cnn_curve, nme_curve, no_nme)
    if args['model_name'] not in ["podnet", "coil"]:
        save_fc(args, model)
    else:
        save_model(args, model)

def _set_device(args):
    device_type = args["device"]
    gpus = [0]

    # for device in device_type:
    #     if device_type == -1:
    #         device = torch.device("cpu")
    #     else:
    #         device = torch.device("cuda:{}".format(device))
    #     gpus.append(device)

    args["device"] = gpus

def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

def save_time(args, cost_time):
    _log_dir = os.path.join("./results/", "times", f"{args['prefix']}")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    with open(_log_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']}, {cost_time} \n")

def save_results(args, cnn_curve, nme_curve, no_nme=False):
    cnn_top1, cnn_top5 = cnn_curve["top1"], cnn_curve['top5']
    nme_top1, nme_top5 = nme_curve["top1"], nme_curve['top5']
    
    #-------CNN TOP1----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top1")
    os.makedirs(_log_dir, exist_ok=True)

    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]} \n")
    else:
        assert args['prefix'] in ['fair', 'auc']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top1[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top1[-1]} \n")

    #-------CNN TOP5----------
    _log_dir = os.path.join("./results/", f"{args['prefix']}", "cnn_top5")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
    if args['prefix'] == 'benchmark':
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},")
            for _acc in cnn_top5[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top5[-1]} \n")
    else:
        assert args['prefix'] in ['auc', 'fair']
        with open(_log_path, "a+") as f:
            f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
            for _acc in cnn_top5[:-1]:
                f.write(f"{_acc},")
            f.write(f"{cnn_top5[-1]} \n")


    #-------NME TOP1----------
    if no_nme is False:
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top1")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")
        else:
            assert args['prefix'] in ['fair', 'auc']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top1[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top1[-1]} \n")       

        #-------NME TOP5----------
        _log_dir = os.path.join("./results/", f"{args['prefix']}", "nme_top5")
        os.makedirs(_log_dir, exist_ok=True)
        _log_path = os.path.join(_log_dir, f"{args['csv_name']}.csv")
        if args['prefix'] == 'benchmark':
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},")
                for _acc in nme_top5[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top5[-1]} \n")
        else:
            assert args['prefix'] in ['auc', 'fair']
            with open(_log_path, "a+") as f:
                f.write(f"{args['time_str']},{args['model_name']},{args['memory_size']},")
                for _acc in nme_top5[:-1]:
                    f.write(f"{_acc},")
                f.write(f"{nme_top5[-1]} \n") 
