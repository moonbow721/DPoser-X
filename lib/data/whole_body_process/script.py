import os
import sys

cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..', '..'))

from lib.data.whole_body_process.prepare_data import makepath, log2file
from lib.data.whole_body_process.prepare_data import prepare_smplx_datasets

expr_code = 'version1'

dposer_datadir = makepath('/data3/ljz24/projects/3d/data/human/WholeBodydataset/GRAB')

logger = log2file(os.path.join(dposer_datadir, '%s.log' % (expr_code)))
logger('[%s] Preparing data for training DPoser.' % expr_code)

amassx_dir = '/data3/ljz24/projects/3d/data/human/Bodydataset/amass_smplx'
amass_splits = {
    'train': ['GRAB']
}

num_expressions = 100

prepare_smplx_datasets(dposer_datadir, amass_splits, amassx_dir, logger=logger, num_expressions=num_expressions)




