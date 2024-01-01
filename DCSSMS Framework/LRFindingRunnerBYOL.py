from LRFinder import *
from MainBYOL import *
from config import *


def run_byol_lr_finder(counter, gpu_id, data_dir, batch_size, suffix, num_layers, use_momentum, finished=False):
    parser = argparse.ArgumentParser('CMDHMSBYOL', parents=[get_args_parser()])
    # gpu_id = 2
    # data_dir = "/root/JUPYTER/ContrastiveMDHMSelection/data_in_use/"
    # batch_size = 256
    # suffix = 10
    # num_layers = 3
    # use_momentum = True    , "2048",   "4096", "8192",   "1024", "512"
    
    if use_momentum:
        args = parser.parse_args(["--gpu_id", str(gpu_id), "--data_dir", data_dir, "--batch_size", str(batch_size),
                                  "--suffix", str(suffix), "--num_layers", str(num_layers),
                                  "--out_sizes", "256", "512", "1024", "2048", "4096", "8192",
                                  "--use_momentum", "True"])
    else:
        args = parser.parse_args(["--gpu_id", str(gpu_id), "--data_dir", data_dir, "--batch_size", str(batch_size),
                                  "--suffix", str(suffix), "--num_layers", str(num_layers),
                                  "--out_sizes", "256", "512", "1024", "2048", "4096", "8192"])
    
    num_iter = 320
                     # 32768 - 320  # 16384 - 600  # 8192 - 1000  # 4096 - 2000
                     # 2048 - 5000  # 1024 - 8000  # 512 - 18000  # 256 - 30000

    device = (f'cuda:{args.gpu_id}' if T.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}\n args.use_momentum: {args.use_momentum}")

    train_loader = prepare_data_loader(args)
    print(f"total # of training steps: {len(train_loader)}")
    # , scheduler
    cmdhmsbyol_net, optimizer, byol_loss = prepare_training_objects(args, device, num_iter)
    cmdhmsbyol_net.float()

    lr_finder = LRFinder(cmdhmsbyol_net, optimizer, byol_loss, args.tau, use_momentum, device=device, cache_dir=args.output_dir,
                         model_name = "byol")

    lr_finder.range_test(train_loader, num_iter=num_iter, step_mode="linear")
    suggested_lr, max_lr = lr_finder.plot(counter, args.output_dir, f"byol_{batch_size}_{suffix}_{num_layers}_{use_momentum}")
    # suggested_lr, max_lr = lr_finder.plot(counter, args.output_dir, f"byolneck_{batch_size}_{suffix}_{num_layers}_{use_momentum}")
    
    results_out_path = ""
    if finished:
        results_out_path = os.path.join(args.output_dir, f"byol_{batch_size}_{suffix}_{num_layers}_{use_momentum}_results.txt")
        # results_out_path = os.path.join(args.output_dir, f"byolneck_{batch_size}_{suffix}_{num_layers}_{use_momentum}_results.txt")
    
    return suggested_lr, max_lr, results_out_path

