from LRFinderLinear import *
from config import *
from DataSetLoader import *
import LinearClassifier as LC


def run_linear_lr_finder(counter, gpu_id, data_dir, embedding_dir, embedding_type, batch_size, suffix, arch,
                         num_layers, outsizes, balanced_train_set, output_dir, finished=False):
    parser = argparse.ArgumentParser('LinearLRFinder', parents=[LC.get_args_parser()])
    # gpu_id = 2
    # data_dir = "/root/JUPYTER/ContrastiveMDHMSelection/data_in_use/"
    # batch_size = 256
    # suffix = 10
    # num_layers = 3
    # use_momentum = True    , "2048",   "4096", "8192",   "1024", "512"
    
    if arch != "":
        if balanced_train_set:
            args = parser.parse_args(["--gpu_id", str(gpu_id), "--data_dir", data_dir, "--embedding_dir", embedding_dir,
                                      "--embedding_type", embedding_type, "--batch_size", str(batch_size),
                                      "--suffix", str(suffix), "--arch", arch, "--num_layers", str(num_layers),
                                      "--output_dir", output_dir, "--out_sizes", *outsizes,
                                      "--balanced_train_set", balanced_train_set])
        else:
            args = parser.parse_args(["--gpu_id", str(gpu_id), "--data_dir", data_dir, "--embedding_dir", embedding_dir,
                                      "--embedding_type", embedding_type, "--batch_size", str(batch_size),
                                      "--suffix", str(suffix), "--arch", arch, "--num_layers", str(num_layers),
                                      "--output_dir", output_dir, "--out_sizes", *outsizes])
    else:
        if balanced_train_set:
            args = parser.parse_args(["--gpu_id", str(gpu_id), "--data_dir", data_dir, "--embedding_dir", embedding_dir,
                                      "--embedding_type", embedding_type, "--batch_size", str(batch_size),
                                     "--suffix", str(suffix), "--num_layers", str(num_layers), "--output_dir", output_dir,
                                      "--out_sizes", *outsizes, "--balanced_train_set", balanced_train_set])
        else:
            args = parser.parse_args(["--gpu_id", str(gpu_id), "--data_dir", data_dir, "--embedding_dir", embedding_dir,
                                      "--embedding_type", embedding_type, "--batch_size", str(batch_size),
                                      "--suffix", str(suffix), "--num_layers", str(num_layers), "--output_dir", output_dir,
                                      "--out_sizes", *outsizes])
    
    num_iter = 90
                     # 32768 - 320  # 16384 - 600  # 8192 - 1000  # 4096 - 2000

    suggested_lr, max_lr, results_out_path = train_in_batch(args, num_iter, counter, finished)

    return suggested_lr, max_lr, results_out_path


def train_in_batch(args, num_iter, counter, finished):
    sampler_names = ["SMOTENC", "SMOTE", "BorderlineSMOTE1", "BorderlineSMOTE2", "SMOTETomek"]
    suggested_lr = 0
    max_lr = 0
    results_out_path = ""
    if args.balanced_train_set:
        suggested_lr, max_lr, results_out_path = get_lr_finding_common_logic(args, "BALANCED",
                                                                             "BALANCED", None,
                                                                             num_iter, counter, finished)
    else:
        for sampler_name in sampler_names:
            params = dict()
            sampler_name_short = sampler_name

            if sampler_name == "SMOTENC":
                params["categorical_encoder"] = None
            elif sampler_name == "BorderlineSMOTE1" or sampler_name == "BorderlineSMOTE2":
                params["kind"] = f"borderline-{sampler_name[-1]}"
                sampler_name_short = sampler_name[:-1]
                print(f"sampler_name: {sampler_name_short}; params['kind'] = {params['kind']}")
            elif sampler_name == "SMOTETomek":
                params["smote"] = None
                params["tomek"] = None

            suggested_lr, max_lr, results_out_path = get_lr_finding_common_logic(args, sampler_name_short,
                                                                                 sampler_name, params, num_iter,
                                                                                 counter, finished)

    return suggested_lr, max_lr, results_out_path


def get_lr_finding_common_logic(args, sampler_name_short, sampler_name, params, num_iter,
                                counter, finished):
    X_train, y_train = LC.prepare_training_dataset(args.data_dir, sampler_name_short, params, args.balanced_train_set)
    X_test, y_test = LC.prepare_test_dataset(args.data_dir)

    num_classes = np.unique(y_train).shape[0]
    print(f"num_classes: {num_classes}; # classes in y_test: {np.unique(y_test).shape[0]}")
    print(f"Sampler: {sampler_name}...")
    print("#" * 60)
    print(f"X_train shape: {X_train.shape}; X_test shape: {X_test.shape}")

    suggested_lr, max_lr, results_out_path = find_lr(args, sampler_name, num_iter, counter,
                                                     X_train, y_train, X_test, y_test, finished)

    return suggested_lr, max_lr, results_out_path


def find_lr(args, sampler_name, num_iter, counter, X_train, y_train, X_test, y_test, finished):
    device = (f'cuda:{args.gpu_id}' if T.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}\n args.arch: {args.arch} args.balanced_train_set: {args.balanced_train_set}")

    train_loader, test_loader, train_dataset = LC.prepare_data_loader(args, X_train, y_train, X_test, y_test)
    total_num_train_steps = len(train_loader)
    print(f"total_num_train_steps: {total_num_train_steps}")
    linear_classifier, optimizer, entropy_loss = LC.prepare_training_objects(args, device, train_dataset.num_classes)
    linear_classifier.float()
    lr_finder = LRFinder(linear_classifier, optimizer, entropy_loss, device=device, cache_dir=args.output_dir)

    lr_finder.range_test(train_loader, test_loader, num_iter=num_iter, step_mode="linear")
    file_prefix = f"linear_{args.batch_size}{args.arch}_{args.num_layers}_Sampler-{sampler_name}_" + \
                  f"{args.balanced_train_set}_{args.embedding_type}"
    suggested_lr, max_lr = lr_finder.plot(counter, args.output_dir, file_prefix)

    results_out_path = ""
    if finished:
        results_out_path = os.path.join(args.output_dir, file_prefix+"_results.txt")

    return suggested_lr, max_lr, results_out_path
