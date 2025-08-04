from configs.wholebody.default_configs import get_default_configs


def get_config():
    config = get_default_configs()
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
    model.pose_config = 'configs.face.subvp.pose_timefc.get_config'
    model.pose_ckpt = './pretrained_models/face/BaseMLP/last.ckpt'
    model.shape_config = 'configs.face.subvp.shape_timefc.get_config'
    model.shape_ckpt = './pretrained_models/face_shape/BaseMLP/last.ckpt'

    return config