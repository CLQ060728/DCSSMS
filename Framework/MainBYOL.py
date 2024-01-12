from config import *
from DataSetLoader import *
from Models import *
from Losses import *


def get_args_parser():
    parser = argparse.ArgumentParser('CMDHMSBYOL', add_help=False)
    # parser.add_argument bool value, default must be False, even you make it to True, it is still False, when set in the command line, it's always True;
    parser.add_argument('--gpu_id', default="0", type=str, help="""Define GPU id.""")
    parser.add_argument('--data_dir', default="", type=str, help="""Path to dataset.""")
    parser.add_argument('--suffix', default="10", type=str, help="""Specify training data set suffix""")
    # parser.add_argument('--add_neg_set', default=False, type=bool, help="""Specify whether to add negtive samples (out of top # samples)""")
    parser.add_argument('--batch_size', default=256, type=int, help="""Training mini-batch size.""")
    parser.add_argument('--init_lr', default=1e-7, type=float, help="""Initial learning rate.""")
    parser.add_argument('--max_lr', default=0.05, type=float, help="""Max learning rate in OneCycle.""")
    parser.add_argument('--weight_decay', default=1e-4, type=float, help="""Weight decay.""")
    parser.add_argument('--num_layers', default=3, type=int, help="""# of embedding layers.""")
    parser.add_argument('--out_sizes', nargs='+', type=int, help="""Embedding layer output feature sizes.""")
    parser.add_argument('--use_momentum', default=False, type=bool, help="""whether to use EMA mechanism""")
    parser.add_argument('--check_point', default=1, type=int, help="""check_pointing in every `--check_point` epochs.""")
    parser.add_argument('--tau', default=0.996, type=float, help="""BYOL moving average parameter.""")
    parser.add_argument('--max_epoch', default=100, type=int, help="""Number of epochs to train.""")
    parser.add_argument('--output_dir', default="", type=str, help="""Path to save logs and checkpoints.""")

    return parser


def prepare_data_loader(args):
    dataset_train = DataSet_Uns(args.data_dir, args.suffix)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    return train_loader


def prepare_training_objects(args, device, total_num_train_steps):
    in_feature_size = 181
    out_sizes = T.tensor(args.out_sizes).to(device)
    cmdhmsbyol_net = CMDHMSBYOL(in_feature_size, args.num_layers, out_sizes, projection_size=512,
                                moving_average_decay = args.tau, use_momentum = args.use_momentum).to(device)
    cmdhmsbyol_net.to_device(device)
    optimizer = T.optim.AdamW(cmdhmsbyol_net.online_net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    byol_loss = BYOLLoss().to(device)
    scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=total_num_train_steps, epochs=args.max_epoch)
    
    return cmdhmsbyol_net, optimizer, byol_loss, scheduler
    

def run_epoch(args, train_loader, cmdhmsbyol_net, optimizer, byol_loss, pi, total_num_train_steps, device):
    m_Loss = 0.0
    Loss = 0.0
    for i, data in enumerate(train_loader):
        X1 = data[:, 0, :]
        X2 = data[:, 1, :]
        X1 = X1.to(device, non_blocking=True, dtype=T.float32)
        X2 = X2.to(device, non_blocking=True, dtype=T.float32)
        # zero the parameter gradients
        optimizer.zero_grad()
        
        X1_online, X1_target, X2_online, X2_target = cmdhmsbyol_net(X1, X2)
        loss = byol_loss(X1_online, X2_online, X1_target, X2_target)
        
        loss.backward()
        optimizer.step()
        
        if args.use_momentum:
            cmdhmsbyol_net.update_moving_average()
            new_tau = 1 - (1 - args.tau) * (0.5 * (T.cos(pi * ((i + 1) / total_num_train_steps)) + 1))
            cmdhmsbyol_net.get_ema_obj().update_tau(new_tau)
        
        Loss += loss.item()
        m_Loss += loss.item()
    
        if (i+1) % 50 == 0:    # print every 50 mini-batches
            print('[%3d] Loss: %.4f' %(i+1, Loss / 50))
            Loss = 0.0
    
    return m_Loss


def train(args):
    min_Loss = 99999999.0
    device = (f'cuda:{args.gpu_id}' if T.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}\n")
    
    pi = T.tensor(np.pi).to(device)
    train_loader = prepare_data_loader(args)
    total_num_train_steps = len(train_loader)
    neck = "OVER_SAMPLING"
    if args.out_sizes[-1] == 512:
        neck = "NECK"
    output_full_path = args.output_dir + f"best_model_{args.batch_size}_{neck}_{args.suffix}_{args.num_layers}.pth"
    print(f"total_num_train_steps: {total_num_train_steps}")
    print(f"momentum: {args.use_momentum}")
    cmdhmsbyol_net, optimizer, byol_loss, scheduler = prepare_training_objects(args, device, total_num_train_steps)
    cmdhmsbyol_net.float()
    cmdhmsbyol_net.train()
    print('Training starts...')
    
    for epoch in range(args.max_epoch):  # loop over the dataset multiple times
        print('Epoch:', epoch)
        m_Loss = run_epoch(args, train_loader, cmdhmsbyol_net, optimizer, byol_loss, pi, total_num_train_steps, device)
        
        # if (epoch + 1) > args.warm_up:
        scheduler.step()

        m_Loss /= total_num_train_steps
        
        if (epoch + 1) % args.check_point == 0:
            
            if m_Loss < min_Loss:
                min_Loss = m_Loss
                print(f"MIN LOSS: {min_Loss}")
                T.save({
                'epoch': epoch,
                'cmdhmsbyol_model_state_dict': cmdhmsbyol_net.state_dict(),
                'online_model_state_dict': cmdhmsbyol_net.online_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, output_full_path)


if __name__ == '__main__':
    parser_main = argparse.ArgumentParser('CMDHMSBYOL', parents=[get_args_parser()])
    args_main = parser_main.parse_args()
    train(args_main)