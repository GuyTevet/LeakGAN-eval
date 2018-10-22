import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader, Gen_Data_loader_text8, Dis_dataloader, Dis_dataloader_text8
# from generator import Generator
from LeakGANModel import  LeakGAN
from Discriminator import Discriminator
from target_lstm import TARGET_LSTM
import pickle
import os
import collections
import json
from tqdm import tqdm
from Main import *
import argparse

# FIXME - hard coded params
BATCH_SIZE = 64
SEED = 88
START_TOKEN = 0
use_real_world_data = True
real_data_file_path = './data/text8'
RESULTS_PATH = './results.txt'
SEQ_LENGTH = 100#1000 # 1000 is not feasible


def restore_param_from_config(config_file,param):
    with open(config_file,'r') as f:
        line = None
        while line != '':
            line = f.readline()
            if line.startswith(param):
                return int(line.replace(param,'').replace(':',''))

    return None


def convergence_experiment(sess, tested_model, data_loader):

    data_loader.reset_pointer()
    batch = data_loader.next_batch()

    N_MAX = 5000  # hard coded for the meantime
    gap = 10 # hard coded for the meantime

    indexes = np.array(range(0,N_MAX+1,gap),dtype=np.float32)
    results = np.zeros_like(indexes)

    pred = np.zeros([BATCH_SIZE, SEQ_LENGTH, tested_model.vocab_size], dtype=np.float32)
    weight = 0

    for tot_i in range(indexes.shape[0] - 1):
        #clculate the gap prediction
        gap_pred = np.zeros([BATCH_SIZE, SEQ_LENGTH, tested_model.vocab_size], dtype=np.float32)
        for gap_i in range(gap):
            pred_one_hot, real_pred = tested_model.language_model_eval_step(sess, batch)
            gap_pred += pred_one_hot
        gap_pred /= gap * 1.

        #calculate new prediction including gap prediction
        new_pred = (pred * weight) + (gap_pred * gap)
        weight += gap
        new_pred /= weight * 1.

        #calculate inf nnorm error
        inf_norm_err = np.linalg.norm((new_pred - pred),ord=np.inf,axis=2)
        avg_err = np.average(inf_norm_err)
        results[tot_i+1] = avg_err

        #updating pred
        pred = new_pred

    return np.concatenate((np.expand_dims(indexes,axis=1),np.expand_dims(results,axis=1)),axis=1).transpose()


def language_model_evaluation(sess, tested_model, data_loader, is_test=False):

    data_loader.reset_pointer()
    BPC_direct_list = []
    BPC_approx_list = []

    for it in tqdm(range(data_loader.num_batch)):
        batch = data_loader.next_batch()
        if is_test:
            real_pred,approx_pred = tested_model.language_model_eval_step(sess, batch)
            BPC_approx = pred2BPC(approx_pred, tested_model, batch)
            BPC_approx_list.append(BPC_approx)
        else:
            real_pred = tested_model.language_model_eval_step_direct_only(sess, batch)

        BPC_direct = pred2BPC(real_pred, tested_model, batch)
        BPC_direct_list.append(BPC_direct)

    if is_test:
        return np.average(BPC_direct_list), np.average(BPC_approx_list)
    else:
        return np.average(BPC_direct_list), None

def pred2BPC(pred,tested_model,test_data):

    real_pred = np.clip(pred, 1e-20, 1)
    pred_flat = np.reshape(real_pred, [BATCH_SIZE * SEQ_LENGTH, tested_model.vocab_size])
    batch_flat = np.reshape(test_data, [BATCH_SIZE * SEQ_LENGTH])
    BPC = np.average([-np.log2(pred_flat[i, batch_flat[i]]) for i in range(pred_flat.shape[0])])

    return BPC

def main(FLAGS):

    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    if use_real_world_data:
        vocab_size = 27
        # split to train-valid-test
        real_data_train_file = real_data_file_path + '-train'
        real_data_valid_file = real_data_file_path + '-valid'
        real_data_test_file = real_data_file_path + '-test'
        real_data_dict_file = real_data_file_path + '-dict.json'
        if not os.path.exists(real_data_train_file):
            split_text8(real_data_file_path)
        charmap, inv_charmap = create_real_data_dict(real_data_train_file,real_data_dict_file)
        # gen_data_loader = Gen_Data_loader_text8(BATCH_SIZE,charmap,inv_charmap,SEQ_LENGTH)
        # dis_data_loader = Dis_dataloader_text8(BATCH_SIZE,charmap,inv_charmap,SEQ_LENGTH)
    else:
        gen_data_loader = Gen_Data_loader(BATCH_SIZE)
        likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
        vocab_size = 5000
        dis_data_loader = Dis_dataloader(BATCH_SIZE)

    experiments_list = [exp for exp in os.listdir('ckp') if os.path.isdir(os.path.join('ckp',exp))]
    if FLAGS.epoch_exp:
        experiments_list.sort(key=lambda x: int(x.split('_epoch_')[-1]))
        stats = np.zeros([2,len(experiments_list)],dtype=np.float32)

    for i, exp_name in enumerate(experiments_list):
        print('#########################################################################')
        print('loading model [%0s]...'%exp_name)

        # restore generator arch
        try:
            if FLAGS.epoch_exp:
                config = os.path.join('ckp','config_' + exp_name.split('_epoch_')[0] + '.txt')
            else:
                config = os.path.join('ckp', 'config_' + exp_name + '.txt')
            EMB_DIM = restore_param_from_config(config, param= 'gen_emb_dim')
            HIDDEN_DIM = restore_param_from_config(config, param= 'gen_hidden_dim')
            dis_embedding_dim = restore_param_from_config(config, param= 'dis_emb_dim')
            seq_len = restore_param_from_config(config, param= 'seq_len')
        except:
            print("ERROR: CONFIG FILE WAS NOT FOUND - skipping model")
            continue
        assert type(EMB_DIM) == int
        assert type(HIDDEN_DIM) == int
        assert type(dis_embedding_dim) == int
        assert type(seq_len) == int

        if seq_len == 20:
            dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
            dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
            LEARNING_RATE = 0.0015
        elif seq_len == 40:
            dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40]
            dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160, 160]
            LEARNING_RATE = 0.0005
        else:
            exit(0)
        print(SEQ_LENGTH)

        GOAL_OUT_SIZE = sum(dis_num_filters)
        GOAL_SIZE = 16

        tf.reset_default_graph()
        # generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
        discriminator = Discriminator(SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, dis_emb_dim=dis_embedding_dim,
                                      filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                      batch_size=BATCH_SIZE, hidden_dim=HIDDEN_DIM, start_token=START_TOKEN,
                                      goal_out_size=GOAL_OUT_SIZE, step_size=4)
        generator = LeakGAN(SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, emb_dim=EMB_DIM,
                          dis_emb_dim=dis_embedding_dim, filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                          batch_size=BATCH_SIZE, hidden_dim=HIDDEN_DIM, start_token=START_TOKEN,
                          goal_out_size=GOAL_OUT_SIZE, goal_size=GOAL_SIZE, step_size=4, D_model=discriminator,
                          learning_rate=LEARNING_RATE)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # sess.run(tf.global_variables_initializer())

        # restore weights
        save_file = os.path.join('.', 'ckp', exp_name, exp_name)
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(list(zip([x.name.split(':')[0] for x in tf.global_variables()], tf.global_variables())))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
                else:
                    print(("Not loading: %s." % saved_var_name))
        saver = tf.train.Saver(restore_vars)
        print ("Restoring vars:")
        print (restore_vars)
        saver.restore(sess, save_file)

        # if exp_name == 'regular_120_50_200':
        #     print('#########################################################################')
        #     print('Conducting convergence expariment...')
        #     test_data_loader = Gen_Data_loader_text8(BATCH_SIZE,charmap,inv_charmap,SEQ_LENGTH)
        #     test_data_loader.create_batches(real_data_test_file)
        #     results = convergence_experiment(sess, generator, test_data_loader)
        #     print('Saving results...')
        #     np.save('SeqGan_' + exp_name + '_conv_results',results)


        print('###')
        print('Start Language Model Evaluation...')
        test_data_loader = Gen_Data_loader_text8(BATCH_SIZE,charmap,inv_charmap,SEQ_LENGTH)
        if FLAGS.test:
            test_data_loader.create_batches(real_data_test_file)
            print("USING TEXT8 TEST SET")
        else:
            test_data_loader.create_batches(real_data_valid_file)
            print("USING TEXT8 VALID SET")
        BPC_direct, BPC_approx = language_model_evaluation(sess,generator, test_data_loader, is_test=FLAGS.test)

        str = "[%0s] BPC_direct = %f"%(exp_name,BPC_direct)
        with open(RESULTS_PATH, 'a') as f:
            f.write(str + '\n')
        print(str)

        if FLAGS.test:
            str = "[%0s] BPC_approx = %f" % (exp_name, BPC_approx)
            with open(RESULTS_PATH,'a') as f:
                f.write(str + '\n')
            print(str)

        if FLAGS.epoch_exp:
            stats[0, i] = int(exp_name.split('_epoch_')[-1])
            stats[1, i] = BPC_direct

    if FLAGS.epoch_exp:
        np.save('direct_results_epoch_exp',stats)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SeqGAN LM Test on Text8")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--epoch_exp', action='store_true')
    FLAGS = parser.parse_args()

    main(FLAGS)