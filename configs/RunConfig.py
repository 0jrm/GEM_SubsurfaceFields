from enum import Enum

class RunConfig(Enum):
    data_folder = "/home/jmiranda/SubsurfaceFields/latest_naiveGEMS"
    training_folder = "/unity/f1/ozavala/OUTPUTS/GEM_SubSurface"

    # val_perc =
    # test_perc = config[TrainingParams.test_percentage]
    # eval_metrics = config[TrainingParams.evaluation_metrics]
    # loss_func = config[TrainingParams.loss_function]
    # batch_size = config[TrainingParams.batch_size]
    # epochs = config[TrainingParams.epochs]
    # model_name_user = config[TrainingParams.config_name]
    # optimizer = config[TrainingParams.optimizer]