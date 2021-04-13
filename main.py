import torch
from torch.utils.data import DataLoader
import argparse
import time
from torch import optim
from torch.nn import functional as F
from network import *
from utils import *
from dataset import *
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100000, help='Iterations')
    parser.add_argument('--size', type=int, default=256, help='Image size')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--fm_U', type=int, default=3, help='feature matching for disc U')
    parser.add_argument('--fm_AB', type=int, default=2, help='feature matching for disc A,B')
    parser.add_argument('--n_downsample', type=int, default=3, help='n_downsample')
    parser.add_argument('--dim', type=int, default=64, help='Encoder dim')
    parser.add_argument('--style_dim', type=int, default=64, help='style_dim')
#     parser.add_argument('--n_upsample', type=int, default=2, help='n_upsample')
    parser.add_argument('--n_res', type=int, default=4, help='n_res')
    parser.add_argument('--norm', type=str, default='none', help='norm')
    parser.add_argument('--activ', type=str, default='relu', help='activ')
    parser.add_argument('--pad_type', type=str, default='reflect', help='pad_type')
    parser.add_argument('--beta1', type=float, default=0, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 for Adam optimizer')
    parser.add_argument('--print', type=int, default=100, help='loss print interval')
    parser.add_argument('--model_save', type=int, default=5000, help='model save interval')
    parser.add_argument('--image_save', type=int, default=100, help='intermediate image save interval')
    parser.add_argument('--image_path', type=str, default='/home/AniGAN4/exp/face2anime/images/', help='intermediate image result')
    parser.add_argument('--dataset_path', type=str, default='/home/AniGAN4/datasets/face2anime/', help='dataset directory')
    parser.add_argument('--output_path', type=str, default='/home/AniGAN4/exp/face2anime/checkpoints/', help='output directory')
    parser.add_argument('--resume', type=bool, default=False, help='resume')
    parser.add_argument('--log_name', type=str, default='/home/AniGAN4/exp/face2anime/log.txt', help='log file name')
    parser.add_argument('--num_workers', type=int, default=8, help='workers for dataloader')
    return parser.parse_args()


def Adv_D_loss(real_logit, fake_logit):
    loss = torch.mean(F.softplus(-real_logit)) + torch.mean(F.softplus(fake_logit))
    return loss


def Adv_G_loss(fake_logit):
    loss = torch.mean(F.softplus(-fake_logit))
    return loss


def recon_loss(recon, images):
    loss = torch.nn.L1Loss()
    return loss(recon, images)


def train():
    s_time = time.time()
    args = parse_args()
    device = torch.device("cuda:0" if (torch.cuda.is_available() > 0) else "cpu")
    train_dataset = ImageDataset(args.dataset_path, size=args.size, unpaired=True, mode='train')
    test_dataset = ImageDataset(args.dataset_path, size=args.size, unpaired=True, mode='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    train_iter = iter(train_loader)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
        num_workers=1
    )
    test_iter = iter(test_loader)

    
    encoder = Encoder(32).to(device)
    decoder = Decoder(args.dim, output_dim=3, input_dim=512, num_ASC_layers=4, num_FST_blocks=5).to(device)
    discriminator_U = Discriminator_U(args.size).to(device)
    discriminator_A = Discriminator_X(16).to(device)
    discriminator_B = Discriminator_Y(16).to(device)

    activation_U = {}
    activation_A = {}
    activation_B = {}

    def get_activation(name, activation):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    for i in range(args.fm_U):  # 0,1,2
        discriminator_U.blocks[i].register_forward_hook(get_activation(str(i), activation_U))

    for i in range(args.fm_AB):  # 0,1
        discriminator_A.blocks[i].register_forward_hook(get_activation(str(i), activation_A))
        discriminator_B.blocks[i].register_forward_hook(get_activation(str(i), activation_B))

    g_optim = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr,
        betas=(args.beta1, args.beta2)
    )

    d_optim = optim.Adam(
        list(discriminator_A.parameters()) + list(discriminator_B.parameters()) + list(discriminator_U.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2)
    )

  
    iteration = 1

    while True:
        try:
            images = next(train_iter)
        except StopIteration:
            print('train dataloader re-initialized')
            train_iter = iter(train_loader)
            images = next(train_iter)
        images_A = images['images_A'].to(device)                                                                       
        images_B = images['images_B'].to(device)                                                                       
        
        #################################### Discriminator ######################

        requires_grad(discriminator_A, True)
        requires_grad(discriminator_B, True)
        requires_grad(discriminator_U, True)
        requires_grad(encoder, False)
        requires_grad(decoder, False)

        c_code_A, s_code_A = encoder(images_A)                                                                                 c_code_B, s_code_B = encoder(images_B) 
        fake_B = decoder(c_code_A, s_code_B)                                                                                   fake_A = decoder(c_code_B, s_code_A)                                                                           
        
        real_logit_A = discriminator_A(discriminator_U(images_A))
        fake_logit_A = discriminator_A(discriminator_U(fake_A.detach()))

        real_logit_B = discriminator_B(discriminator_U(images_B))
        fake_logit_B = discriminator_B(discriminator_U(fake_B.detach()))

        d_loss = Adv_D_loss(real_logit_A, fake_logit_A) + Adv_D_loss(real_logit_B, fake_logit_B)
        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()


        ############### Generator (adv loss) ########################

        requires_grad(discriminator_A, False)
        requires_grad(discriminator_B, False)
        requires_grad(discriminator_U, False)
        requires_grad(encoder, True)
        requires_grad(decoder, True)
        
        c_code_A, s_code_A = encoder(images_A)
        c_code_B, s_code_B = encoder(images_B)

        fake_B = decoder(c_code_A, s_code_B)  # 1 x 3 x 256 x 256
        fake_A = decoder(c_code_B, s_code_A)  # 1 x 3 x 256 x 256

        fake_logit_A = discriminator_A(discriminator_U(fake_A))
        fake_logit_B = discriminator_B(discriminator_U(fake_B))

        adv_g_loss = Adv_G_loss(fake_logit_A) + Adv_G_loss(fake_logit_B)
        g_optim.zero_grad()
        adv_g_loss.backward()
        g_optim.step()

        ################ Generator (Feature matching) ######################## 

        fm_loss_UA = 0
        fm_loss_AA = 0
        fm_loss_UB = 0
        fm_loss_BB = 0
        
        ### Domain A
        c_code_A, s_code_A = encoder(images_A)  
        recon_A = decoder(c_code_A, s_code_A)  
        
        _ = discriminator_A(discriminator_U(recon_A)) 
        recon_U_feature_A = copy.deepcopy(activation_U) 
        recon_A_feature_A = copy.deepcopy(activation_A) 
        
        _ = discriminator_A(discriminator_U(images_A))
        real_U_feature_A = copy.deepcopy(activation_U)
        real_A_feature_A = copy.deepcopy(activation_A)

        for _ in range(args.fm_U):
            fm_loss_UA += recon_loss(recon_U_feature_A[str(i)].mean([2,3]),real_U_feature_A[str(i)].mean([2,3]))

        for _ in range(args.fm_AB):
            fm_loss_AA += recon_loss(recon_A_feature_A[str(i)].mean([2,3]),real_A_feature_A[str(i)].mean([2,3]))

        ### Domain B
        c_code_B, s_code_B = encoder(images_B) 
        recon_B = decoder(c_code_B, s_code_B)  

        _ = discriminator_B(discriminator_U(recon_B))  
        recon_U_feature_B = copy.deepcopy(activation_U)  
        recon_B_feature_B = copy.deepcopy(activation_B) 

        _ = discriminator_B(discriminator_U(images_B))
        real_U_feature_B = copy.deepcopy(activation_U)
        real_B_feature_B = copy.deepcopy(activation_B)

        for _ in range(args.fm_U):
            fm_loss_UB += recon_loss(recon_U_feature_B[str(i)].mean([2, 3]), real_U_feature_B[str(i)].mean([2, 3]))

        for _ in range(args.fm_AB):
            fm_loss_BB += recon_loss(recon_B_feature_B[str(i)].mean([2, 3]), real_B_feature_B[str(i)].mean([2, 3]))


        fm_loss = fm_loss_UA + fm_loss_AA + fm_loss_UB + fm_loss_BB
        fm_loss.requires_grad= True
        fm_loss = fm_loss
        g_optim.zero_grad()
        fm_loss.backward()
        g_optim.step()

        
        ################ Generator (recon loss) ########################
        
        c_code_A, s_code_A = encoder(images_A)
        recon_A = decoder(c_code_A, s_code_A)  # 1 x 3 x 256 x 256
        recon_loss_AA = recon_loss(recon_A, images_A)
        recon_loss_AA = recon_loss_AA * 10
        
        c_code_B, s_code_B = encoder(images_B)
        recon_B = decoder(c_code_B, s_code_B)  # 1 x 3 x 256 x 256
        recon_loss_BB = recon_loss(recon_B, images_B)
        recon_loss_BB = recon_loss_AA * 10
        recon_loss_T = recon_loss_AA + recon_loss_BB

        g_optim.zero_grad()
        recon_loss_T.backward()
        g_optim.step()


        if iteration % args.print == 0:
            time_elapsed = timer(s_time, time.time())
            s = f'iteration: {iteration}, d loss: {d_loss:.3f} adv_g_loss: {adv_g_loss:.3f}  recon_loss_T: {recon_loss_T:.3f} time: {time_elapsed}'
#             print(s)
            with open(args.log_name, "a") as log_file:
                log_file.write('%s\n' % s)


        if iteration % args.model_save == 0:
            model_save_path = os.path.join(args.output_path, f'model_{str(iteration).zfill(6)}.pt')
            torch.save({
                'iter': iteration,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'discriminator_U': discriminator_U.state_dict(),
                'discriminator_A': discriminator_A.state_dict(),
                'discriminator_B': discriminator_B.state_dict(),
                'g_optimizer': g_optim.state_dict(),
                'd_optimizer': d_optim.state_dict(),
            }, model_save_path
            )
            print(f'model_{str(iteration)} saved')

        if iteration % args.image_save == 0:
            try:
                test_images = next(test_iter)
            except StopIteration:
                test_iter = iter(test_iter)
                test_images = next(test_iter)

            test_images_A = test_images['images_A'].to(device)
            test_images_B = test_images['images_B'].to(device)

            with torch.no_grad():
                test_c_code_A, test_s_code_A = encoder(test_images_A)
                test_c_code_B, test_s_code_B = encoder(test_images_B)
                test_recon_A = decoder(test_c_code_A, test_s_code_A)
                test_recon_B = decoder(test_c_code_B, test_s_code_B)
                test_hybrid_A = decoder(test_c_code_B, test_s_code_A)
                test_hybrid_B = decoder(test_c_code_A, test_s_code_B)
            img_save(test_images_A, 'image_A', iteration, args.image_path)
            img_save(test_images_B, 'image_B', iteration, args.image_path)
            img_save(test_recon_A, 'recon_A', iteration, args.image_path)
            img_save(test_hybrid_A, 'hybrid_A', iteration, args.image_path)
            img_save(test_recon_B, 'recon_B', iteration, args.image_path)
            img_save(test_hybrid_B, 'hybrid_B', iteration, args.image_path)
            print(f'images_{str(iteration)} saved')


        if iteration >= args.iterations:
            break
        else:
            iteration += 1


if __name__ == '__main__':
    print('train started')
    train()
    print('training finished')










