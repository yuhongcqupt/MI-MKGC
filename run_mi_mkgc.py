from email.generator import Generator

import numpy as np
import torch
from mmkgc.config import Tester, AdvMixTrainer, WCGTrainerGP,Trainer
from mmkgc.module.model import AdvMixRotatE
from mmkgc.module.loss import SigmoidLoss
from mmkgc.module.model.VBRotatE import VBRotatE
from mmkgc.module.strategy import NegativeSampling
from mmkgc.data import TrainDataLoader, TestDataLoader
from mmkgc.adv.modules import MultiGenerator, CombinedGenerator

from args import get_args

def apply_missing_modalities(embeddings, missing_ratio):
    """
    根据给定的缺失比例丢失嵌入数据。
    :param embeddings: 嵌入数据
    :param missing_ratio: 丢失比例
    :return: 修改后的嵌入数据
    """
    missing = int(embeddings.size(0) * missing_ratio)  # 计算丢失的样本数量
    missing_indices = np.random.choice(embeddings.size(0), missing, replace=False)  # 随机选择丢失的索引
    embeddings[missing_indices] = 0  # 用0填充丢失的样本，也可以选择用None
    return embeddings

if __name__ == "__main__":
    args = get_args()
    print(args)
    # set the seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/" + args.dataset + '/',
        batch_size=args.batch_size,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=args.neg_num,
        neg_rel=0
    )
    # dataloader for test
    test_dataloader = TestDataLoader(
        "./benchmarks/" + args.dataset + '/', "link")
    img_emb = torch.load('./embeddings/' + args.dataset + '-visual.pth')
    text_emb = torch.load('./embeddings/' + args.dataset + '-textual.pth')

    # # 应用固定比例的丢失
    # missing_ratio = 0.8  # 设定丢失比例
    # img_emb = apply_missing_modalities(img_emb, missing_ratio)  # 模拟图片模态丢失
    # text_emb = apply_missing_modalities(text_emb, missing_ratio)  # 模拟文本模态丢失

    # define the model
    kge_score = AdvMixRotatE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=args.dim,
        margin=args.margin,
        epsilon=2.0,
        img_emb=img_emb,
        text_emb=text_emb
    )
    print(kge_score)

    # define the loss function
    model = NegativeSampling(
        model=kge_score,
        loss=SigmoidLoss(adv_temperature=args.adv_temp),
        batch_size=train_dataloader.get_batch_size(),
        regul_rate=0.00001
    )

    adv_generator = CombinedGenerator(
        noise_dim=64,
        structure_dim=2 * args.dim,
        img_dim=2 * args.dim
    )

    # train the model
    trainer = WCGTrainerGP.WCGTrainerGP(
        model=model,
        data_loader=train_dataloader,
        train_times=args.epoch,
        alpha=args.learning_rate,
        use_gpu=True,
        opt_method='Adam',
        generator=adv_generator,
        lrg=args.lrg,
        mu=args.mu)



    trainer.run()
    kge_score.save_checkpoint(args.save)

    # test the model
    kge_score.load_checkpoint(args.save)
    tester = Tester(model=kge_score, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)
