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

    # sampling
    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'

    # model
    model = config.model
    model.type = 'Combiner'
    model.body_config = 'configs.body.subvp.timefc.get_config'
    model.body_ckpt = './pretrained_models/body/BaseMLP/last.ckpt'
    model.hand_config = 'configs.hand.subvp.timefc.get_config'
    model.hand_ckpt = './pretrained_models/hand/BaseMLP/last.ckpt'
    model.face_config = 'configs.face.subvp.pose_timefc.get_config'
    model.face_ckpt = './pretrained_models/face/BaseMLP/last.ckpt'

    return config