from config import *
import DataSetLoader as DSL
import Classification_Pipeline as CP
from Models import *
from Losses import *


def get_args_parser():
    parser = argparse.ArgumentParser('LinearEvaluator', add_help=False)
    # parser.add_argument bool value, default must be False, even you make it to True, it is still False, when set in the command line, it's always True;
    parser.add_argument('--gpu_id', default="0", type=str, help="""Define GPU id.""")
    parser.add_argument('--data_dir', default="", type=str, help="""Path to dataset.""")
    parser.add_argument('--embedding_dir', default="", type=str, help="""Path to pretrained embeddings.""")
    parser.add_argument('--linear_model_dir', default="", type=str, help="""Path to pretrained linear model.""")
    parser.add_argument('--embedding_type', default="BYOL", type=str, help="""Embedding Type (currently 'BYOL' only).""")
    parser.add_argument('--balanced_train_set', default=False, type=bool, help="""whether to use balanced training set or not...""")
    parser.add_argument('--suffix', default="10", type=str, help="""Specify training data set suffix""")
    parser.add_argument('--threshold', default=10, type=int, help="""Specify test data set NO. of MDH method occurrances""")
    parser.add_argument('--arch', default="", type=str, help="""Embedding architecture type, whether bottleneck...""")
    parser.add_argument('--batch_size', default=256, type=int, help="""Training mini-batch size.""")
    parser.add_argument('--init_lr', default=1e-7, type=float, help="""Initial learning rate.""")
    parser.add_argument('--weight_decay', default=1e-4, type=float, help="""Weight decay.""")
    parser.add_argument('--num_layers', default=3, type=int, help="""# of embedding layers.""")
    parser.add_argument('--out_sizes', nargs='+', type=int, help="""Embedding layer output feature sizes.""")
    parser.add_argument('--check_point', default=1, type=int,
                        help="""check_pointing in every `--check_point` epochs.""")
    parser.add_argument('--max_epoch', default=500, type=int, help="""Number of epochs to train.""")
    parser.add_argument('--output_dir', default="", type=str, help="""Path to save logs and checkpoints.""")

    return parser


def prepare_test_dataset(ds_path):
    assert ds_path is not None and ds_path != "", "Please specify data path!"
    _, test_set, num_classes = DSL.assemble_dataset(ds_path)
    
    X_test, y_test, _ = DSL.one_hot_encode_df_X_cat(test_set)
    
    return X_test, y_test, num_classes


def prepare_real_test_dataset(ds_path, threshold):
    assert ds_path is not None and ds_path != "", "Please specify data path!"
    real_test_set, num_classes = DSL.assemble_real_test(ds_path, threshold)

    X_test, y_test, _ = DSL.one_hot_encode_df_X_cat(real_test_set)

    return X_test, y_test, num_classes


def prepare_training_dataset(ds_path, sampler_name, params, balanced=False):
    assert ds_path is not None and ds_path != "", "Please specify data path!"
    train_set, _, num_classes = DSL.assemble_dataset(ds_path, balanced)

    if balanced:
        X_res, y_res, _ = DSL.one_hot_encode_df_X_cat(train_set)
    else:
        if sampler_name != "SMOTENC":
            X, y, _ = DSL.one_hot_encode_df_X_cat(train_set)
        else:
            X, y = train_set.iloc[:, :-1], train_set.iloc[:, -1]
        
        sampler = CP.getSampler(sampler_name, params)
        X_res, y_res = sampler.fit_resample(X, y)
        
        if sampler_name == "SMOTENC":
            combined_set = pds.concat([X_res, y_res], axis=1, ignore_index=True)
            X_res, y_res, _ = DSL.one_hot_encode_df_X_cat(combined_set)
    
    return X_res, y_res, num_classes
    

def prepare_data_loader(args, X_train, y_train, X_test, y_test):
    dataset_train = DSL.DataSet_Torch(X_train, y_train)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    dataset_test = DSL.DataSet_Torch(X_test, y_test)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    return train_loader, test_loader, dataset_train


def prepare_training_objects(args, device, num_classes):
    in_feature_size = 181
    out_sizes = T.tensor(args.out_sizes).to(device)
    linear_evaluator = LinearEvalModel(args.embedding_type, args.embedding_dir, in_feature_size, args.num_layers,
                                       out_sizes, num_classes, device, projection_size=512).to(device)
    optimizer = T.optim.SGD(linear_evaluator.parameters(), lr=args.init_lr, momentum=0.9,
                            weight_decay=args.weight_decay)
    loss = nn.CrossEntropyLoss().to(device)
    # scheduler = OneCycleLR(optimizer, max_lr=args.max_lr, steps_per_epoch=total_num_train_steps, epochs=args.max_epoch)

    return linear_evaluator, optimizer, loss  # scheduler


def run_train_epoch(train_loader, linear_evaluator, optimizer, cross_entropy_loss, metrics, device):
    linear_evaluator.train()
    print('Training starts...')

    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device, non_blocking=True, dtype=T.float32)
        labels = labels.to(device, non_blocking=True, dtype=T.long)
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = linear_evaluator(inputs)
        loss = cross_entropy_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        metrics_dict = metrics(outputs, labels)

        print(f"{i + 1} Training Metrics: {metrics_dict}")


def run_test_epoch(test_loader, linear_evaluator, metrics, device):
    linear_evaluator.eval()
    print('Test starts...')
    
    with T.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True, dtype=T.float32)
            labels = labels.to(device, non_blocking=True, dtype=T.long)
            # forward pass
            outputs = linear_evaluator(inputs)
            # calculate the loss
            # loss = cross_entropy_loss(outputs, labels)
            
            metrics_dict = metrics(outputs, labels)
            print(f"{i + 1} Test Metrics: {metrics_dict}")


def train_in_batch(args):
    sampler_names = ["TomekLinks", "SMOTENC", "SMOTE", "BorderlineSMOTE1", "BorderlineSMOTE2", "SMOTETomek"]
    if args.balanced_train_set:
        get_train_in_batch_common_logic(args, "BALANCED", "BALANCED", None)
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

            get_train_in_batch_common_logic(args, sampler_name_short, sampler_name, params)


def get_train_in_batch_common_logic(args, sampler_name_short, sampler_name, params):
    X_train, y_train, num_classes = prepare_training_dataset(args.data_dir, sampler_name_short, params, args.balanced_train_set)
    X_test, y_test, _ = prepare_test_dataset(args.data_dir)

    print(f"num_classes: {num_classes}")
    metrics_train = MetricCollection({"bacc": MulticlassAccuracy(num_classes=num_classes, average="micro"),
                                      "f1": MulticlassF1Score(num_classes=num_classes, average="micro"),
                                      "mcc": MulticlassMatthewsCorrCoef(num_classes=num_classes),
                                      "kappa": MulticlassCohenKappa(num_classes=num_classes)})
    metrics_test = MetricCollection({"bacc": MulticlassAccuracy(num_classes=num_classes, average="macro"),
                                     "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
                                     "mcc": MulticlassMatthewsCorrCoef(num_classes=num_classes),
                                     "kappa": MulticlassCohenKappa(num_classes=num_classes)})
    print(f"Sampler: {sampler_name}...")
    print("#" * 60)

    # skf = StratifiedKFold(n_splits = 9, shuffle = True)
    # for i, (train_indices, val_indices) in enumerate(skf.split(X, y)):
    print(f"X_train shape: {X_train.shape}; X_test shape: {X_test.shape}")

    baccs, f1s, mccs, kappas = train(args, sampler_name, X_train, y_train, X_test, y_test, num_classes,
                                     metrics_train, metrics_test)

    print("-" * 80)
    bacc_stats = T.tensor([T.min(baccs), T.max(baccs), T.mean(baccs), T.median(baccs), T.std(baccs)],
                          device=f"cuda:{args.gpu_id}")
    bacc_indices = T.tensor([T.argmin(baccs), T.argmax(baccs)], device=f"cuda:{args.gpu_id}")
    print(f"bacc_stats: {bacc_stats}; bacc_indices: {bacc_indices}")

    f1_stats = T.tensor([T.min(f1s), T.max(f1s), T.mean(f1s), T.median(f1s), T.std(f1s)],
                        device=f"cuda:{args.gpu_id}")
    f1_indices = T.tensor([T.argmin(f1s), T.argmax(f1s)], device=f"cuda:{args.gpu_id}")
    print(f"f1_stats: {f1_stats}; f1_indices: {f1_indices}")

    mcc_stats = T.tensor([T.min(mccs), T.max(mccs), T.mean(mccs), T.median(mccs), T.std(mccs)],
                         device=f"cuda:{args.gpu_id}")
    mcc_indices = T.tensor([T.argmin(mccs), T.argmax(mccs)], device=f"cuda:{args.gpu_id}")
    print(f"mcc_stats: {mcc_stats}; mcc_indices: {mcc_indices}")

    kappa_stats = T.tensor([T.min(kappas), T.max(kappas), T.mean(kappas), T.median(kappas), T.std(kappas)],
                           device=f"cuda:{args.gpu_id}")
    kappa_indices = T.tensor([T.argmin(kappas), T.argmax(kappas)], device=f"cuda:{args.gpu_id}")
    print(f"kappa_stats: {kappa_stats}; kappa_indices: {kappa_indices}")

    print("#" * 80)


def train(args, sampler_name, X_train, y_train, X_test, y_test, num_classes,
          metrics_train, metrics_test):
    best_test_bacc = 0
    best_test_f1 = 0
    best_test_mcc = 0
    best_test_kappa = 0
    device = (f'cuda:{args.gpu_id}' if T.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}\n")
    
    train_loader, test_loader, train_dataset = prepare_data_loader(args, X_train, y_train, X_test, y_test)
    total_num_train_steps = len(train_loader)
    output_file_name = f"best_model_{args.batch_size}{args.arch}_{args.num_layers}_Sampler-{sampler_name}_" + \
                       f"{args.balanced_train_set}_{args.embedding_type}"
    print(f"total_num_train_steps: {total_num_train_steps}")
    linear_evaluator, optimizer, entropy_loss = prepare_training_objects(args, device, num_classes)
    linear_evaluator.float()
    metrics_train = metrics_train.to(device)
    metrics_test = metrics_test.to(device)
    baccs = T.tensor([], device=device)
    f1s = T.tensor([], device=device)
    mccs = T.tensor([], device=device)
    kappas = T.tensor([], device=device)
    
    for epoch in range(args.max_epoch):  # loop over the dataset multiple times
        print('Epoch:', epoch)
        run_train_epoch(train_loader, linear_evaluator, optimizer, entropy_loss, metrics_train, device)

        run_test_epoch(test_loader, linear_evaluator, metrics_test, device)
        
        epoch_metrics_dict = metrics_test.compute()
        baccs = T.cat((baccs, T.tensor([epoch_metrics_dict["bacc"]], device=device)))
        f1s = T.cat((f1s, T.tensor([epoch_metrics_dict["f1"]], device=device)))
        mccs = T.cat((mccs, T.tensor([epoch_metrics_dict["mcc"]], device=device)))
        kappas = T.cat((kappas, T.tensor([epoch_metrics_dict["kappa"]], device=device)))
        epoch_bacc = epoch_metrics_dict["bacc"]
        epoch_f1 = epoch_metrics_dict["f1"]
        epoch_mcc = epoch_metrics_dict["mcc"]
        epoch_kappa = epoch_metrics_dict["kappa"]
        
        if (epoch + 1) % args.check_point == 0:
            if best_test_bacc < epoch_bacc:
                best_test_bacc = epoch_bacc
                print(f"Best Test Accuracy: {best_test_bacc}")
                output_file_name_bacc = output_file_name + "_best_test_bacc.pth"
                save_best_model(args, output_file_name_bacc, epoch, linear_evaluator, optimizer)
            if best_test_f1 < epoch_f1:
                best_test_f1 = epoch_f1
                print(f"Best Test F1-Macro: {best_test_f1}")
                output_file_name_f1 = output_file_name + "_best_test_f1.pth"
                save_best_model(args, output_file_name_f1, epoch, linear_evaluator, optimizer)
            if best_test_mcc < epoch_mcc:
                best_test_mcc = epoch_mcc
                print(f"Best Test MCC: {best_test_mcc}")
                output_file_name_mcc = output_file_name + "_best_test_mcc.pth"
                save_best_model(args, output_file_name_mcc, epoch, linear_evaluator, optimizer)
            if best_test_kappa < epoch_kappa:
                best_test_kappa = epoch_kappa
                print(f"Best Test Kappa: {best_test_kappa}")
                output_file_name_kappa = output_file_name + "_best_test_kappa.pth"
                save_best_model(args, output_file_name_kappa, epoch, linear_evaluator, optimizer)
        
        metrics_train.reset()
        metrics_test.reset()
    
    print(f"baccs: {baccs}")
    print(f"f1s: {f1s}")
    print(f"mccs: {mccs}")
    print(f"kappas: {kappas}")
    
    return baccs, f1s, mccs, kappas


def test(args):
    # X_test, y_test, num_classes = prepare_real_test_dataset(args.data_dir, args.threshold)
    X_test, y_test, num_classes = prepare_test_dataset(args.data_dir)
    device = (f'cuda:{args.gpu_id}' if T.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}\n")
    print(f"num_classes: {num_classes}")
    metrics_test = MetricCollection({"bacc": MulticlassAccuracy(num_classes=num_classes),
                                     "f1": MulticlassF1Score(num_classes=num_classes),
                                     "mcc": MulticlassMatthewsCorrCoef(num_classes=num_classes),
                                     "kappa": MulticlassCohenKappa(num_classes=num_classes)})
    metrics_test = metrics_test.to(device)
    dataset_test = DSL.DataSet_Torch(X_test, y_test)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    in_feature_size = 181
    out_sizes = T.tensor(args.out_sizes).to(device)
    linear_evaluator = LinearEvalModel(args.embedding_type, args.embedding_dir, in_feature_size, args.num_layers,
                                       out_sizes, num_classes, device, projection_size=512).to(device)
    best_model_dict = T.load(args.linear_model_dir, map_location=device)
    linear_evaluator.load_state_dict(best_model_dict['linear_evaluator_state_dict'])
    
    run_test_epoch(test_loader, linear_evaluator, metrics_test, device)
    epoch_metrics_dict = metrics_test.compute()
    epoch_bacc = epoch_metrics_dict["bacc"]
    epoch_f1 = epoch_metrics_dict["f1"]
    epoch_mcc = epoch_metrics_dict["mcc"]
    epoch_kappa = epoch_metrics_dict["kappa"]

    metrics_test.reset()

    print(f"bacc: {epoch_bacc}")
    print(f"f1: {epoch_f1}")
    print(f"mcc: {epoch_mcc}")
    print(f"kappa: {epoch_kappa}")


def save_best_model(args, output_file_name, epoch, linear_evaluator, optimizer):
    output_full_path = os.path.join(args.output_dir, output_file_name)
    T.save({
        'epoch': epoch,
        'linear_evaluator_state_dict': linear_evaluator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_full_path)


if __name__ == '__main__':
    parser_main = argparse.ArgumentParser('LinearEvaluatorMain', parents=[get_args_parser()])
    args_main = parser_main.parse_args()
    # train_in_batch(args_main)
    test(args_main)