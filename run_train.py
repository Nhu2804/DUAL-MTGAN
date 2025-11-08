import random

import torch
import numpy as np

from config import get_training_args, get_paths
from model import Generator, Critic, BaseGRU
from trainer import GANTrainer, BaseGRUTrainer
from datautils.dataloader import get_dual_name_maps, load_meta_data, get_train_test_loader, get_base_gru_train_loader, get_code_numbers


def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # SỬA: Lấy thêm diagnosis_num, procedure_num từ get_paths
    dataset_path, records_path, params_path, diagnosis_num, procedure_num, total_codes = get_paths(args)
    
    # SỬA: Load dual-stream metadata
    (len_dist, diag_visit_dist, diag_patient_dist, proc_visit_dist, proc_patient_dist, 
     code_adj, diagnosis_map, procedure_map, code_info) = load_meta_data(dataset_path)
    
    # SỬA QUAN TRỌNG: Load cả diagnosis và procedure name maps
    diagnosis_name_map, procedure_name_map = get_dual_name_maps(args.data_path)
    
    train_loader, test_loader, max_len = get_train_test_loader(dataset_path, args.batch_size, device)
    

    # SỬA: Sử dụng total_codes thay vì len(code_adj)
    len_dist = torch.from_numpy(len_dist).to(device)

    # SỬA: Khởi tạo dual-stream BaseGRU
    base_gru = BaseGRU(
        diagnosis_num=diagnosis_num, 
        procedure_num=procedure_num, 
        hidden_dim=args.g_hidden_dim, 
        max_len=max_len
    ).to(device)
    
    try:
        base_gru.load(params_path)
    except FileNotFoundError:
        base_gru_trainloader = get_base_gru_train_loader(dataset_path, args.batch_size, device)
        base_gru_trainer = BaseGRUTrainer(args, base_gru, max_len, base_gru_trainloader, params_path)
        base_gru_trainer.train()
    base_gru.eval()

    # SỬA: Khởi tạo dual-stream Generator
    generator = Generator(
        diagnosis_num=diagnosis_num,
        procedure_num=procedure_num,
        hidden_dim=args.g_hidden_dim,
        attention_dim=args.g_attention_dim,
        max_len=max_len,
        device=device
    ).to(device)
    
    # SỬA: Khởi tạo dual-stream Critic
    critic = Critic(
        diagnosis_num=diagnosis_num,
        procedure_num=procedure_num,
        hidden_dim=args.c_hidden_dim,
        generator_hidden_dim=args.g_hidden_dim,
        max_len=max_len
    ).to(device)

    print('Generator params:', count_model_params(generator))
    print('Critic params:', count_model_params(critic))
    print('Total params:', count_model_params(generator) + count_model_params(critic))

    # SỬA QUAN TRỌNG: Truyền cả diagnosis_name_map và procedure_name_map
    trainer = GANTrainer(
        args,
        generator=generator, 
        critic=critic, 
        base_gru=base_gru,
        train_loader=train_loader, 
        test_loader=test_loader,
        len_dist=len_dist, 
        diagnosis_map=diagnosis_map, 
        procedure_map=procedure_map,
        diagnosis_name_map=diagnosis_name_map,  # THÊM DÒNG NÀY
        procedure_name_map=procedure_name_map,  # THÊM DÒNG NÀY
        records_path=records_path, 
        params_path=params_path
    )
    trainer.train()


if __name__ == '__main__':
    args = get_training_args()
    train(args)