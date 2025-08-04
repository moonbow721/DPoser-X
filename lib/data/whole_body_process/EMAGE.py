import csv
import glob
import os
import os.path as osp
import shutil
import sys

import numpy as np
import torch

torch.set_grad_enabled(False)

save_betas = False
num_betas = 10


def logger_sequencer(logger_list, prefix=None):
    def post_text(text):
        if prefix is not None: text = '{} -- '.format(prefix) + text
        for logger_call in logger_list: logger_call(text)

    return post_text


class log2file():
    def __init__(self, logpath=None, prefix='', auto_newline=True, write2file_only=False):
        if logpath is not None:
            makepath(logpath, isfile=True)
            self.fhandle = open(logpath, 'a+')
        else:
            self.fhandle = None

        self.prefix = prefix
        self.auto_newline = auto_newline
        self.write2file_only = write2file_only

    def __call__(self, text):
        if text is None: return
        if self.prefix != '': text = '{} -- '.format(self.prefix) + text
        # breakpoint()
        if self.auto_newline:
            if not text.endswith('\n'):
                text = text + '\n'
        if not self.write2file_only: sys.stderr.write(text)
        if self.fhandle is not None:
            self.fhandle.write(text)
            self.fhandle.flush()


def makepath(*args, **kwargs):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    isfile = kwargs.get('isfile', False)
    import os
    desired_path = os.path.join(*args)
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)): os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path


def prepare_smplx_datasets(smplx_dataset_dir, amass_splits, amass_dir, logger=None, num_expressions=100):
    ds_logger = log2file(makepath(smplx_dataset_dir, 'dataset.log', isfile=True), write2file_only=True)
    logger = ds_logger if logger is None else logger_sequencer([ds_logger, logger])

    logger(f'Creating pytorch dataset at {smplx_dataset_dir}')
    logger(f'Using AMASS body parameters from {amass_dir}')

    shutil.copy2(__file__, smplx_dataset_dir)

    def fetch_from_amass(ds_name):
        keep_rate = 0.3

        npz_fnames = []
        file_path = glob.glob(osp.join(amass_dir, ds_name, '*/*.npz'))
        npz_fnames.extend(file_path)
        logger('Found {} sequences from {}.'.format(len(file_path), ds_name))
        split_csv_path = osp.join(amass_dir, ds_name, 'train_test_split.csv')
        type_dict = {}
        with open(split_csv_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                type_dict[row['id']] = row['type']

        for npz_fname in npz_fnames:
            print(npz_fname)
            cdata = np.load(npz_fname, allow_pickle=True)

            N = len(cdata['poses'])  # 165 (3+63+90+3+6) [global_orient, body_pose, two_hands, jaw, eye]
            # skip first and last frames to avoid initial standard poses, e.g. T pose
            cdata_ids = np.random.choice(list(range(int(0.1 * N), int(0.9 * N), 1)), int(keep_rate * 0.8 * N),
                                         replace=False)
            if len(cdata_ids) < 1: continue
            fullpose = cdata['poses'][cdata_ids].astype(np.float32)
            expression = cdata['expressions'][cdata_ids].astype(np.float32)
            padded_expression = np.zeros((len(cdata_ids), num_expressions - expression.shape[1]), dtype=np.float32)
            result_dict = {'global_orient': fullpose[:, :3], 'body_pose': fullpose[:, 3:66],
                           'jaw_pose': fullpose[:, 66:69],
                           # 'leye_pose': fullpose[:, 69:72], 'reye_pose': fullpose[:, 72:75],
                           'left_hand_pose': fullpose[:, 75:75+45],
                           'right_hand_pose': fullpose[:, 120:120+45],
                           'expression': np.concatenate([expression, padded_expression], axis=1)}

            filename_key = os.path.basename(npz_fname)[:-4]  # remove .npz
            if filename_key in type_dict:
                result_dict['type'] = type_dict[filename_key]
            else:
                result_dict['type'] = 'unknown'
            if save_betas:
                result_dict['betas'] = np.tile(cdata['betas'][:num_betas], (len(cdata_ids), 1))

            yield result_dict

    train_data_fields, val_data_fields, test_data_fields = {}, {}, {}

    for ds_name in amass_splits:
        logger(f'Preparing SMPLX data for {ds_name}')

        for data in fetch_from_amass(ds_name):
            if data['type'] == 'val':
                for k in data.keys():
                    if k == 'type': continue
                    if k not in val_data_fields: val_data_fields[k] = []
                    val_data_fields[k].append(data[k])
            elif data['type'] == 'test':
                for k in data.keys():
                    if k == 'type': continue
                    if k not in test_data_fields: test_data_fields[k] = []
                    test_data_fields[k].append(data[k])
            else:
                for k in data.keys():
                    if k == 'type': continue
                    if k not in train_data_fields: train_data_fields[k] = []
                    train_data_fields[k].append(data[k])

    for k, v in train_data_fields.items():
        outpath = makepath(smplx_dataset_dir, 'train', '{}.pt'.format(k), isfile=True)
        v = np.concatenate(v)
        torch.save(torch.from_numpy(v), outpath)
    for k, v in val_data_fields.items():
        outpath = makepath(smplx_dataset_dir, 'val', '{}.pt'.format(k), isfile=True)
        v = np.concatenate(v)
        torch.save(torch.from_numpy(v), outpath)
    for k, v in test_data_fields.items():
        outpath = makepath(smplx_dataset_dir, 'test', '{}.pt'.format(k), isfile=True)
        v = np.concatenate(v)
        torch.save(torch.from_numpy(v), outpath)

    logger(
        f'{len(v)} datapoints dumped for {ds_name}. ds_meta_pklpath: {osp.join(smplx_dataset_dir, )}')

    logger(f'Dumped final pytorch dataset at {smplx_dataset_dir}')


if __name__ == '__main__':
    expr_code = 'version1'

    save_datadir = makepath('/data3/ljz24/projects/3d/DPoser/wholebody_data/BEAT')

    logger = log2file(os.path.join(save_datadir, '%s.log' % (expr_code)))
    logger('[%s] Preparing data for training DPoser.' % expr_code)

    source_dir = '/data3/ljz24/projects/3d/DPoser/wholebody_data/beat_v2.0.0'
    amass_splits = ['beat_english_v2.0.0', 'beat_chinese_v2.0.0', 'beat_japanese_v2.0.0', 'beat_spanish_v2.0.0']

    num_expressions = 100

    prepare_smplx_datasets(save_datadir, amass_splits, source_dir, logger=logger, num_expressions=num_expressions)
