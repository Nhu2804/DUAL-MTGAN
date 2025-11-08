import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from generation.generate import generate_ehr
from generation.stat_ehr import get_basic_statistics, get_top_k_disease


class Logger:
    def __init__(self, plot_path, generator, diagnosis_map, procedure_map, 
                 diagnosis_name_map, procedure_name_map, len_dist, save_number, save_batch_size):
        self.plot_path = plot_path
        self.generator = generator
        self.save_number = save_number
        self.save_batch_size = save_batch_size

        # GIỮ NGUYÊN CẤU TRÚC PLOTS NHƯ GỐC
        self.plots = {
            'train': {
                'd_loss': {
                    'data': [],
                    'title': 'Discriminator Loss'
                },
                'g_loss': {
                    'data': [],
                    'title': 'Generator Loss'
                },
                'w_distance': {
                    'data': [],
                    'title': 'Wasserstein Distance'
                }
            },
            'test': {
                'test_d_loss': {
                    'data': [],
                    'title': 'Test Discriminator Loss'
                }
            },
            'generate': {
                'gen_code_type': {
                    'data': [],
                    'title': 'Generated Code Type'
                },
                'gen_code_num': {
                    'data': [],
                    'title': 'Generated Code Number'
                },
                'gen_avg_code_num': {
                    'data': [],
                    'title': 'Generated Average Code Number'
                }
            }
        }

        self.device = generator.device

        self.logfile = open(os.path.join(plot_path, 'output.log'), 'w', encoding='utf-8')

        # SỬA QUAN TRỌNG: TÁCH RIÊNG NAME MAPS
        self.diagnosis_name_map = diagnosis_name_map
        self.procedure_name_map = procedure_name_map
        # XÓA: self.code_name_map
        
        # THÊM MAP CHO DUAL-STREAM
        self.diagnosis_map = diagnosis_map
        self.procedure_map = procedure_map
        self.idiagnosis_map = {v: k for k, v in diagnosis_map.items()}
        self.iprocedure_map = {v: k for k, v in procedure_map.items()}
        self.len_dist = len_dist

    def append_point(self, key, loss_type, loss):
        self.plots[key][loss_type]['data'].append(loss)

    # GIỮ NGUYÊN HÀM NHƯ GỐC
    def add_train_point(self, d_loss, g_loss, w_distance):
        self.append_point('train', 'd_loss', d_loss)
        self.append_point('train', 'g_loss', g_loss)
        self.append_point('train', 'w_distance', w_distance)

    # GIỮ NGUYÊN HÀM NHƯ GỐC
    def add_test_point(self, test_d_loss):
        self.append_point('test', 'test_d_loss', test_d_loss)

    # GIỮ NGUYÊN HÀM NHƯ GỐC
    def add_gen_point(self, gen_code_type, gen_code_num, gen_avg_code_num):
        self.append_point('generate', 'gen_code_type', gen_code_type)
        self.append_point('generate', 'gen_code_num', gen_code_num)
        self.append_point('generate', 'gen_avg_code_num', gen_avg_code_num)

    def plot_dict(self, key, x):
        for item in self.plots[key].values():
            y, title = item['data'], item['title']
            plt.clf()
            plt.plot(x, y)
            plt.xlabel('Iteration')
            plt.ylabel(title)
            plt.savefig(os.path.join(self.plot_path, title.replace(' ', '_') + '.png'))

    def plot_train(self):
        points_num = len(self.plots['train']['d_loss']['data'])
        x = np.arange(1, points_num + 1)
        self.plot_dict('train', x)

    def plot_test(self):
        train_points_num = len(self.plots['train']['d_loss']['data'])
        test_points_num = len(self.plots['test']['test_d_loss']['data'])
        step = train_points_num // test_points_num
        x = np.arange(1, test_points_num + 1) * step
        self.plot_dict('test', x)

    def plot_gen(self):
        train_points_num = len(self.plots['train']['d_loss']['data'])
        gen_points_num = len(self.plots['generate']['gen_code_type']['data'])
        step = train_points_num // gen_points_num
        x = np.arange(1, gen_points_num + 1) * step
        self.plot_dict('generate', x)

    def stat_generation(self):
        fake_diagnoses, fake_procedures, fake_lens = generate_ehr(
            self.generator, self.save_number, self.len_dist, self.save_batch_size
        )
        
        # Tính statistics cho diagnosis
        diag_n_types, diag_n_codes, diag_n_visits, diag_avg_code_num, diag_avg_visit_num = get_basic_statistics(fake_diagnoses, fake_lens)
        # Tính statistics cho procedure  
        proc_n_types, proc_n_codes, proc_n_visits, proc_avg_code_num, proc_avg_visit_num = get_basic_statistics(fake_procedures, fake_lens)
        
        # HIỂN THỊ RIÊNG BIỆT
        log = ('Generating {} samples -- \n'
            'Diagnosis: {} types, {} codes, avg {:.4f} codes/visit\n'
            'Procedure: {} types, {} codes, avg {:.4f} codes/visit\n'
            'Avg visit len: {:.4f}') \
            .format(self.save_number, 
                diag_n_types, diag_n_codes, diag_avg_code_num,
                proc_n_types, proc_n_codes, proc_avg_code_num,
                diag_avg_visit_num)
        self.add_log(log)
        print(log)
        
        # SỬA QUAN TRỌNG: DÙNG ĐÚNG NAME MAPS
        self.add_log("\n=== TOP 10 DIAGNOSES ===")
        get_top_k_disease(fake_diagnoses, fake_lens, self.idiagnosis_map, self.diagnosis_name_map, top_k=10, file=self.logfile)
        
        self.add_log("\n=== TOP 10 PROCEDURES ===")
        # DÙNG PROCEDURE NAME MAP
        if self.procedure_name_map:  # Nếu có procedure name map
            get_top_k_disease(fake_procedures, fake_lens, self.iprocedure_map, self.procedure_name_map, top_k=10, file=self.logfile)
        else:
            # Nếu không có procedure name map, hiển thị codes only
            self.add_log("Procedure codes (codes only):")
            proc_count = {}
            for patient, len_i in zip(fake_procedures, fake_lens):
                for i in range(len_i):
                    admission = patient[i]
                    codes = np.where(admission > 0)[0]
                    for code in codes:
                        proc_code = self.iprocedure_map[code]
                        proc_count[proc_code] = proc_count.get(proc_code, 0) + 1
            sorted_proc = sorted(proc_count.items(), key=lambda x: x[1], reverse=True)[:10]
            for code, count in sorted_proc:
                self.add_log(f"PROC_{code} ; {count}")
        
        self.add_log('\n')
        self.logfile.flush()

        # Vẫn dùng tổng cho plotting
        total_types = diag_n_types + proc_n_types
        total_codes = diag_n_codes + proc_n_codes
        total_avg_code_num = (diag_avg_code_num + proc_avg_code_num) / 2
        self.add_gen_point(total_types, total_codes, total_avg_code_num)
        self.plot_gen()

    def add_log(self, line):
        t = type(line)
        if t is str:
            self.logfile.write(line + '\n')
        elif t is list:
            lines = [line_ + '\n' for line_ in line]
            self.logfile.writelines(lines)
        self.logfile.flush()

    def save(self):
        pickle.dump(self.plots, open(os.path.join(self.plot_path, 'history.log'), 'wb'))