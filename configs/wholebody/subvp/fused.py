from configs.wholebody.default_configs import get_default_configs


def get_config():
    config = get_default_configs()
    # data
    config.dataset = 'wholebody'
    data = config.data
    data.normalize = True
    data.rot_rep = 'axis'  # rot6d or axis
    data.min_max = False  # Z-score or min-max Normalize
    data.dataset_names = ['EMAGE', 'GRAB', 'EgoBody', 'Arctic']
    data.num_expressions = 100

    # training
    training = config.training
    training.sde = 'subvpsde'
    training.continuous = True

    training.batch_size = 1280
    training.n_iters = 400000
    training.log_freq = 50
    training.eval_freq = 10000
    training.save_freq = 15000
    training.auxiliary_loss = False  # not recommended
    training.denoise_steps = 10  # for computing auxiliary loss
    training.render = False  # render results while validating
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = True
    # mixed training
    training.random_part_mask = False
    training.mask_prob = 0.2
    training.apply_loss = False

    # evaluation
    evaluate = config.eval
    evaluate.batch_size = 50

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'

    # model
    model = config.model
    model.type = 'Finetune'
    model.body_config = 'configs.body.subvp.timefc.get_config'
    model.body_ckpt = './pretrained_models/body/BaseMLP/last.ckpt'
    model.hand_config = 'configs.hand.subvp.timefc.get_config'
    model.hand_ckpt = './pretrained_models/hand/BaseMLP/last.ckpt'
    model.face_config = 'configs.face.subvp.pose_timefc.get_config'
    model.face_ckpt = './pretrained_models/face/BaseMLP/last.ckpt'

    model.HIDDEN_DIM = 512
    model.EMBED_DIM = 256
    model.N_BLOCKS = 2
    model.dropout = 0.1
    model.fourier_scale = 16
    model.scale_by_sigma = True
    model.ema_rate = 0.9999
    model.nonlinearity = 'swish'
    model.embedding_type = 'positional'  # Or 'fourier'

    return config