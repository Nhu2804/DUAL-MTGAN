from .generator_trainer import GeneratorTrainer
from .critic_trainer import CriticTrainer
from datautils.data_sampler import get_train_sampler
from logger import Logger


class GANTrainer:
    def __init__(self, args,
                 generator, critic, base_gru,
                 train_loader, test_loader,
                 len_dist, diagnosis_map, procedure_map, 
                 diagnosis_name_map, procedure_name_map,  # THÊM 2 PARAM NÀY
                 records_path, params_path):

        self.generator = generator
        self.critic = critic
        self.base_gru = base_gru
        self.params_path = params_path
        self.test_loader = test_loader

        self.g_trainer = GeneratorTrainer(generator, critic,
                                          batch_size=args.batch_size, train_num=args.g_iter,
                                          lr=args.g_lr, betas=(args.betas0, args.betas1),
                                          decay_step=args.decay_step, decay_rate=args.decay_rate)
        self.d_trainer = CriticTrainer(critic, generator, base_gru,
                                       batch_size=args.batch_size, train_num=args.c_iter,
                                       lr=args.c_lr, lambda_=args.lambda_, betas=(args.betas0, args.betas1),
                                       decay_step=args.decay_step, decay_rate=args.decay_rate)
        
        # SỬA QUAN TRỌNG: Logger nhận cả diagnosis_name_map và procedure_name_map
        self.logger = Logger(records_path, generator, diagnosis_map, procedure_map, 
                           diagnosis_name_map, procedure_name_map,  # THÊM 2 PARAM
                           len_dist, train_loader.size, args.save_batch_size)

        self.test_freq = args.test_freq
        self.save_freq = args.save_freq
        self.device = generator.device
        self.iters = args.iteration
        self.train_sampler = get_train_sampler(train_loader, self.device)
        self.batch_size = train_loader.batch_size

    def train(self):
        for i in range(1, self.iters + 1):
            # 1️⃣ Lấy target codes cho cả diagnoses và procedures
            target_diagnoses, target_procedures = self.generator.get_target_codes(
                self.batch_size, code_type='both'
            )
            
            # 2️⃣ Sample real data (trả về diagnoses, procedures, lens)
            real_diagnoses, real_procedures, real_lens = self.train_sampler.sample(
                target_diagnoses=target_diagnoses, target_procedures=target_procedures
            )

            # 🧠 KHÔNG ép batch, KHÔNG cắt — giống MTGAN gốc
            # Batch thực tế có thể nhỏ hơn nếu code hiếm
            batch_size_real = real_diagnoses.size(0)
            # Bạn chỉ cần đảm bảo các tensor có cùng chiều batch
            # Nếu generator dùng batch cố định, chỉ lấy phần tương ứng
            if target_diagnoses is not None:
                target_diagnoses = target_diagnoses[:batch_size_real]
            if target_procedures is not None:
                target_procedures = target_procedures[:batch_size_real]

            # 3️⃣ Train critic với dual-stream data
            (d_loss, w_distance, 
            d_real_diag, d_real_proc,
            d_fake_diag, d_fake_proc) = self.d_trainer.step(
                real_diagnoses, real_procedures, real_lens,
                target_diagnoses, target_procedures
            )

            # 4️⃣ Train generator
            g_loss, g_diagnosis_output, g_procedure_output = self.g_trainer.step(
                target_diagnoses, target_procedures, real_lens
            )

            # 5️⃣ Log chi tiết hơn (dual-stream)
            self.logger.add_train_point(d_loss, g_loss, w_distance)

            if i % self.test_freq == 0:
                test_d_loss, test_diagnosis_score, test_procedure_score = self.d_trainer.evaluate(
                    self.test_loader, self.device
                )
                self.logger.add_test_point(test_d_loss)

                line = (
                    f"{i} / {self.iters} iterations:\n"
                    f" D_Loss -- {d_loss:.6f} | G_Loss -- {g_loss:.6f} | W_dist -- {w_distance:.6f} | Test_D_Loss -- {test_d_loss:.6f}\n"
                    f"   Diag(D_real={d_real_diag:.4f}, D_fake={d_fake_diag:.4f}) | "
                    f"Proc(D_real={d_real_proc:.4f}, D_fake={d_fake_proc:.4f})"
                )
                print("\r" + line)
                self.logger.add_log(line)
                self.logger.plot_train()
                self.logger.plot_test()
                self.logger.stat_generation(i) 
                self.logger.plot_dual_stream_losses()
            else:
                line = (
                    f"{i} / {self.iters} iterations: "
                    f"D_Loss -- {d_loss:.6f} | G_Loss -- {g_loss:.6f} | W_dist -- {w_distance:.6f} "
                    f"| Diag(D_real={d_real_diag:.4f}, D_fake={d_fake_diag:.4f}) "
                    f"| Proc(D_real={d_real_proc:.4f}, D_fake={d_fake_proc:.4f})"
                )
                print("\r" + line, end='')


            if i % self.save_freq == 0:
                self.generator.save(self.params_path, f'generator.{i}.pt')
                self.critic.save(self.params_path, f'critic.{i}.pt')

        # 6️⃣ Lưu model sau khi huấn luyện
        self.generator.save(self.params_path)
        self.critic.save(self.params_path)
        self.logger.save()
