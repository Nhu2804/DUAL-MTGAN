import os

import argparse


def _parse_base_setting(parser):
    group = parser.add_argument_group('base', 'base settings')
    group.add_argument('--seed', default=6669, type=int)
    group.add_argument('--data_path', default='data', type=str)
    group.add_argument('--dataset', default='mimic3', type=str, choices=['mimic3', 'mimic4'])
    group.add_argument('--result_path', default='result', type=str)


def _parse_preprocess_setting(parser):
    group = parser.add_argument_group('preprocess', 'preprocess settings')
    group.add_argument('--from_saved', action='store_true')
    group.add_argument('--train_num', default=6000, type=int)
    group.add_argument('--sample_num', default=10000, type=int, help='for mimic4')


def _parse_model_structure_setting(parser):
    group = parser.add_argument_group('model', 'model structure setting')
    group.add_argument('--g_hidden_dim', default=256, type=int)
    group.add_argument('--g_attention_dim', default=64, type=int)
    group.add_argument('--c_hidden_dim', default=64, type=int)


def _parse_gan_training_setting(parser):
    group = parser.add_argument_group('gan_training', 'gan training setting')
    group.add_argument('--iteration', default=300000, type=int)
    group.add_argument('--batch_size', default=256, type=int)
    group.add_argument('--g_iter', default=1, type=int)
    group.add_argument('--g_lr', default=1e-4, type=float)
    group.add_argument('--c_iter', default=1, type=int)
    group.add_argument('--c_lr', default=1e-5, type=float)
    group.add_argument('--betas0', default=0.5, type=float)
    group.add_argument('--betas1', default=0.9, type=float)
    group.add_argument('--lambda_', default=10, type=float)
    group.add_argument('--decay_rate', default=0.1, type=float)
    group.add_argument('--decay_step', default=100000, type=int)


def _parse_base_gru_setting(parser):
    group = parser.add_argument_group('base_gru_training', 'base gru training setting')
    group.add_argument('--base_gru_epochs', default=200, type=int)
    group.add_argument('--base_gru_lr', default=1e-3, type=float)


def _parse_log_setting(parser):
    group = parser.add_argument_group('log', 'log setting')
    group.add_argument('--test_freq', default=1000, type=int)
    group.add_argument('--save_freq', default=1000, type=int)
    group.add_argument('--save_batch_size', default=256, type=int)


def _parse_generate_setting(parser):
    group = parser.add_argument_group('generate', 'generate setting')
    group.add_argument('--batch_size', default=256, type=int)
    group.add_argument('--use_iteration', default=-1, type=int)
    group.add_argument('--number', default=1200, type=int)
    group.add_argument('--upper_bound', default=1e7, type=int)


def get_preprocess_args():
    parser = argparse.ArgumentParser(description='Parameters for Data Preprocess')
    _parse_base_setting(parser)
    _parse_preprocess_setting(parser)

    args = parser.parse_args()
    return args


def get_training_args():
    parser = argparse.ArgumentParser(description='Parameters for training MTGAN')
    _parse_base_setting(parser)
    _parse_model_structure_setting(parser)
    _parse_base_gru_setting(parser)
    _parse_gan_training_setting(parser)
    _parse_log_setting(parser)

    args = parser.parse_args()
    return args


def get_generate_args():
    parser = argparse.ArgumentParser(description='Parameters for Generation')
    _parse_base_setting(parser)
    _parse_model_structure_setting(parser)
    _parse_generate_setting(parser)

    args = parser.parse_args()
    return args


def get_paths(args):
    dataset_path = os.path.join(args.data_path, args.dataset)

    result_path = os.path.join(args.result_path, args.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    records_path = os.path.join(result_path, 'records')
    if not os.path.exists(records_path):
        os.mkdir(records_path)
    params_path = os.path.join(result_path, 'params')
    if not os.path.exists(params_path):
        os.mkdir(params_path)
    
    # Lấy code numbers info
    diagnosis_num, procedure_num, total_codes = get_code_numbers(dataset_path)
    
    return dataset_path, records_path, params_path, diagnosis_num, procedure_num, total_codes


def get_code_numbers(dataset_path):
    """Lấy số lượng diagnoses và procedures từ encoded data"""
    import pickle
    encoded_path = os.path.join(dataset_path, 'encoded')
    code_info_path = os.path.join(encoded_path, 'code_info.pkl')
    
    if os.path.exists(code_info_path):
        code_info = pickle.load(open(code_info_path, 'rb'))
        diagnosis_num = code_info['diagnosis_num']
        procedure_num = code_info['procedure_num']
        total_codes = code_info['total_codes']
        return diagnosis_num, procedure_num, total_codes
    else:
        # Fallback: giả định combined data
        code_map_path = os.path.join(encoded_path, 'code_map.pkl')
        if os.path.exists(code_map_path):
            code_map = pickle.load(open(code_map_path, 'rb'))
            total_codes = len(code_map)
            # Giả định 70% diagnoses, 30% procedures (có thể điều chỉnh)
            diagnosis_num = int(total_codes * 0.7)
            procedure_num = total_codes - diagnosis_num
            return diagnosis_num, procedure_num, total_codes
        else:
            raise FileNotFoundError("Cannot find code info files")