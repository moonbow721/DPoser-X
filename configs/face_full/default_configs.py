from configs.general_configs import get_general_configs


def get_default_configs():
    config = get_general_configs()
    # original
    config.devices = [4, 5, 6, 7]
    config.name = 'default'

    # data
    data = config.data
    data.normalize = True
    data.rot_rep = 'axis'  # rot6d or axis
    data.min_max = False  # Z-score or min-max Normalize

    config.seed = 42

    return config
