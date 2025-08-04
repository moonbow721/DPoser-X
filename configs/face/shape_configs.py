from configs.general_configs import get_general_configs


def get_default_configs():
    config = get_general_configs()
    # original
    config.devices = [0, 1]
    config.name = 'default'
    config.dataset = 'mica'

    # data
    data = config.data
    data.normalize = True
    data.rot_rep = 'axis'  # rot6d or axis
    data.min_max = False  # Z-score or min-max Normalize
    data.train_dataset_names = ['facewarehouse', 'florence', 'frgc', 'ft', 'stirling',
                                'lyhm_train', 'wcpa_train']
    data.val_dataset_names = ['lyhm_valid', 'wcpa_valid']
    data.replica = 50   # number of replicas for each dataset
    data.num_expressions = 100
    data.num_betas = 100

    # training
    training = config.training
    config.training.batch_size = 128
    training.n_iters = 200000
    training.log_freq = 10
    training.eval_freq = 5000
    training.save_freq = 8000
    training.auxiliary_loss = False  # not recommended
    training.denoise_steps = 10  # for computing auxiliary loss
    training.render = True  # render results while validating
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = True
    # Ambient Diffusion
    training.random_mask = False
    training.min_mask_rate = 0.0
    training.max_mask_rate = 0.2
    training.observation_type = 'zeros'

    # evaluation
    evaluate = config.eval
    evaluate.batch_size = 50

    # optimization
    optim = config.optim
    optim.optimizer = 'RAdam'
    optim.lr = 1e-4
    optim.weight_decay = 0.0
    optim.warmup = 5000

    config.seed = 42

    return config
