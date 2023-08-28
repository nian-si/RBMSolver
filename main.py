"""
The main file to solve RBM control solver problems.
"""

import json
import munch
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import equation as eqn
from solver import ControlSolver

flags.DEFINE_string('config_path', 'configs/RBMControlTS_Q1_d6.json',
                    """The path to load json file.""")
flags.DEFINE_string('exp_name', None,
                    """The name of numerical experiments, prefix for logging""")

flags.DEFINE_boolean('dump', True,
                    """generate the data or load the existing data""")

FLAGS = flags.FLAGS
FLAGS.log_dir = './logs'  # directory where to write event logs and output array

def main(argv):
    del argv
    if FLAGS.exp_name is None: # use config name as exp_name
        FLAGS.exp_name = os.path.splitext(os.path.basename(FLAGS.config_path))[0]
    with open(FLAGS.config_path) as json_data_file:
        config = json.load(json_data_file)
    config = munch.munchify(config)
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)


    tf.keras.backend.set_floatx(config.net_config.dtype)
    dim = config.eqn_config.dim


    
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)
    path_prefix = os.path.join(FLAGS.log_dir, FLAGS.exp_name)
    with open('{}_config.json'.format(path_prefix), 'w') as outfile:
        json.dump(dict((name, getattr(config, name))
                       for name in dir(config) if not name.startswith('__')),
                  outfile, indent=2)

    absl_logging.get_absl_handler().setFormatter(logging.Formatter('%(levelname)-6s %(message)s'))
    absl_logging.set_verbosity('info')
            
    logging.info('Begin to solve %s ' % config.eqn_config.eqn_name)
    Control_solver = ControlSolver(config, bsde)
    
    dump_flag = FLAGS.dump
    Control_solver.gen_samples(dump = dump_flag, load = not dump_flag)

    Control_solver.train()

    
if __name__ == '__main__':
    app.run(main)
