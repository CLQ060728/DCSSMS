from DataSetLoader import *
import Classification_Pipeline as CP
from config import *
# import LinearEvaluation as LE


def get_args_parser():
    parser = argparse.ArgumentParser('ClassifiersTrainer', add_help=False)
    parser.add_argument('--classifier_name', default="", type=str, help="""specify to train which classifier.""")
    parser.add_argument('--balanced_train_set', default=False, type=bool, help="""whether to use balanced training set or not...""")
    parser.add_argument('--hidden_size', default=256, type=int, help="""Training MLP hidden layer size.""")
    parser.add_argument('--batch_size', default=256, type=int, help="""Training MLP batch size.""")
    parser.add_argument('--best_model_path', default="", type=str, help="""Path to load saved best models.""")
    parser.add_argument('--output_dir', default="", type=str, help="""Path to save logs and checkpoints.""")

    return parser


def load_dataset(ds_path, balanced=False):
    # suffix = 10
    train_set, test_set, _ = assemble_dataset(ds_path, balanced)
    ori_labels = train_set.iloc[:, -1]

    return train_set, test_set, ori_labels


def train_save_classifiers_in_batch(args):
    ds_path = "/root/JUPYTER/DATA/RAW_DATA/"
    print(f"args.balanced_train_set: {args.balanced_train_set}")
    train_set, test_set, label_array = load_dataset(ds_path, args.balanced_train_set)
    
    sampler_names = ["TomekLinks", "SMOTENC", "SMOTE", "BorderlineSMOTE1", "BorderlineSMOTE2", "SMOTETomek"]
    classifier_names = [args.classifier_name] # "ADA", "RF", "MLP"
    classifier_params = dict()
    classifier_params["hidden"] = args.hidden_size
    classifier_params["batch"] = args.batch_size
    classifier_params["n_est"] = None
    output_path = args.output_dir
    
    for classifier_name in classifier_names:
        print(f"classifier: {classifier_name}")
        if classifier_name == "MLP":
            for classifier_params["nesterov"] in [True, False]:
                print(f"classifier_params['nesterov']: {classifier_params['nesterov']}")
                train_save_single_classifier(args, train_set, test_set, sampler_names, classifier_name, classifier_params, output_path)
                print("%" * 100)
        else:
            for classifier_params["n_est"] in [60, 80, 100]:
                print(f"classifier_params['n_est']: {classifier_params['n_est']}")
                train_save_single_classifier(args, train_set, test_set, sampler_names, classifier_name, classifier_params, output_path)
                print("%" * 100)
        
        print("\n" * 2)


def train_save_furia_in_batch(args):
    ds_path = "/root/JUPYTER/DATA/RAW_DATA/"
    print(f"args.balanced_train_set: {args.balanced_train_set}")
    train_set, test_set, label_array = load_dataset(ds_path, args.balanced_train_set)
    
    sampler_names = ["TomekLinks", "SMOTENC", "SMOTE", "BorderlineSMOTE1", "BorderlineSMOTE2", "SMOTETomek"]
    classifier_names = [args.classifier_name]  # "FURIA"
    classifier_params = dict()
    classifier_params["n_est"] = None
    output_path = args.output_dir
    jvm.start(max_heap_size="50000m", packages=True)
    
    for classifier_name in classifier_names:
        print(f"classifier: {classifier_name}")
        train_save_single_classifier(args, train_set, test_set, sampler_names, classifier_name, classifier_params, output_path)
        print("%" * 100)
        
        print("\n" * 2)


def train_save_single_classifier(args, train_set, test_set, sampler_names, classifier_name, classifier_params, output_path):
    nesterov = "None"
    if "nesterov" in classifier_params:
        nesterov = classifier_params["nesterov"]
    columns = train_set.columns
    dtypes = train_set.dtypes
    
    if args.balanced_train_set:
        if classifier_name != "FURIA":
            X_train, y_train = label_encoder_df_X_cat(train_set)
            X_test, y_test = label_encoder_df_X_cat(test_set)
        else:
            X_train = train_set.iloc[:, :-1]
            y_train = train_set.iloc[:, -1]
            X_test = test_set.iloc[:, :-1]
            y_test = test_set.iloc[:, -1]
        
        all_baccs, all_f1s, all_mccs, all_kappas, all_models = \
            train_save_single_classifier_iteration(X_train, y_train, X_test, y_test, columns, dtypes, None, None, "", None,
                                                   classifier_name, classifier_params, args.balanced_train_set)
            
        print("-" * 80)

        train_save_single_classifier_best_output(args, classifier_name, nesterov, classifier_params, "",
                                                 all_baccs, all_f1s, all_mccs, all_kappas, all_models, output_path)
    
    else:
        for sampler_name in sampler_names:
            params = dict()
            sampler_name_short = sampler_name
            encoder_train, encoder_test = None, None

            if sampler_name != "SMOTENC" and classifier_name != "FURIA":
                X_train, y_train, encoder_train = one_hot_encode_df_X_cat(train_set)
                X_test, y_test, encoder_test = one_hot_encode_df_X_cat(test_set)
            else:
                X_train = train_set.iloc[:, :-1]
                y_train = train_set.iloc[:, -1]
                X_test = test_set.iloc[:, :-1]
                y_test = test_set.iloc[:, -1]
                
            if sampler_name == "SMOTENC":
                params["categorical_encoder"] = None
            elif sampler_name == "BorderlineSMOTE1" or sampler_name == "BorderlineSMOTE2":
                params["kind"] = f"borderline-{sampler_name[-1]}"
                sampler_name_short = sampler_name[:-1]
                print(f"sampler_name: {sampler_name_short}; params['kind'] = {params['kind']}")
            elif sampler_name == "SMOTETomek":
                params["smote"] = None
                params["tomek"] = None
            
            print(f"sampler_name: {sampler_name}")
            all_baccs, all_f1s, all_mccs, all_kappas, all_models = \
                train_save_single_classifier_iteration(X_train, y_train, X_test, y_test, columns, dtypes, encoder_train, encoder_test,
                                                       sampler_name_short, params, classifier_name,
                                                       classifier_params, args.balanced_train_set)
            
            print("-" * 80)
            
            train_save_single_classifier_best_output(args, classifier_name, nesterov, classifier_params, sampler_name,
                                                     all_baccs, all_f1s, all_mccs, all_kappas, all_models, output_path)


def train_save_single_classifier_iteration(X_train, y_train, X_test, y_test, columns, dtypes, encoder_train, encoder_test,
                                           sampler_name, params, classifier_name, classifier_params, balanced_train_set):
    all_baccs = []
    all_f1s = []
    all_mccs = []
    all_kappas = []
    all_models = []
    for iter_counter in range(1, 501, 1):
        print(f"Iteration {iter_counter}...")
        print("#" * 60)

        bacc_value, f1_value, mcc_value, kappa_value, model = CP.runImbalancedClassificationPipeline(X_train, y_train, X_test, y_test,
                                                                                                     columns, dtypes, 
                                                                                                     encoder_train, encoder_test,
                                                                                                     sampler_name, params, classifier_name,
                                                                                                     classifier_params, balanced_train_set)

        all_baccs.append(bacc_value)
        print(f"all_baccs: {all_baccs}")

        all_f1s.append(f1_value)
        print(f"all_f1s: {all_f1s}")

        all_mccs.append(mcc_value)
        print(f"all_mccs: {all_mccs}")

        all_kappas.append(kappa_value)
        print(f"all_kappas: {all_kappas}")

        all_models.append(model)
        # print(f"all_models: {all_models}\n")

        print(f"bacc: {bacc_value}")
        print(f"f1: {f1_value}")
        print(f"mcc: {mcc_value}")
        print(f"kappa: {kappa_value}")
    
    return all_baccs, all_f1s, all_mccs, all_kappas, all_models


def train_save_single_classifier_best_output(args, classifier_name, nesterov, classifier_params, sampler_name,
                                             all_baccs, all_f1s, all_mccs, all_kappas, all_models, output_path):
    best_all_bacc = np.argmax(all_baccs)
    print(f"best_all_bacc: {best_all_bacc}")
    output_bacc_model_name = f"best_bacc_for_{classifier_name}{nesterov}-{classifier_params['n_est']}" + \
                             f"_{sampler_name}_iteration{best_all_bacc + 1}_{args.batch_size}_{args.hidden_size}_{args.balanced_train_set}.joblib"
    output_bacc_full_path = os.path.join(output_path, output_bacc_model_name)
    best_bacc_model = all_models[best_all_bacc]
    bacc_stats = np.array([np.min(all_baccs), np.max(all_baccs), np.mean(all_baccs), np.median(all_baccs), np.std(all_baccs)])
    bacc_indices = np.array([np.argmin(all_baccs), np.argmax(all_baccs)])
    print(f"bacc_stats: {bacc_stats}; bacc_indices: {bacc_indices}")

    best_all_f1 = np.argmax(all_f1s)
    print(f"best_all_f1: {best_all_f1}")
    output_f1_model_name = f"best_f1_for_{classifier_name}{nesterov}-{classifier_params['n_est']}" + \
                           f"_{sampler_name}_iteration{best_all_f1 + 1}_{args.batch_size}_{args.hidden_size}_{args.balanced_train_set}.joblib"
    output_f1_full_path = os.path.join(output_path, output_f1_model_name)
    best_f1_model = all_models[best_all_f1]
    f1_stats = np.array([np.min(all_f1s), np.max(all_f1s), np.mean(all_f1s), np.median(all_f1s), np.std(all_f1s)])
    f1_indices = np.array([np.argmin(all_f1s), np.argmax(all_f1s)])
    print(f"f1_stats: {f1_stats}; f1_indices: {f1_indices}")

    best_all_mcc = np.argmax(all_mccs)
    print(f"best_all_mcc: {best_all_mcc}")
    output_mcc_model_name = f"best_mcc_for_{classifier_name}{nesterov}-{classifier_params['n_est']}" + \
                            f"_{sampler_name}_iteration{best_all_mcc + 1}_{args.batch_size}_{args.hidden_size}_{args.balanced_train_set}.joblib"
    output_mcc_full_path = os.path.join(output_path, output_mcc_model_name)
    best_mcc_model = all_models[best_all_mcc]
    mcc_stats = np.array([np.min(all_mccs), np.max(all_mccs), np.mean(all_mccs), np.median(all_mccs), np.std(all_mccs)])
    mcc_indices = np.array([np.argmin(all_mccs), np.argmax(all_mccs)])
    print(f"mcc_stats: {mcc_stats}; mcc_indices: {mcc_indices}")

    best_all_kappa = np.argmax(all_kappas)
    print(f"best_all_kappa: {best_all_kappa}")
    output_kappa_model_name = f"best_kappa_for_{classifier_name}{nesterov}-{classifier_params['n_est']}" + \
                              f"_{sampler_name}_iteration{best_all_kappa + 1}_{args.batch_size}_{args.hidden_size}_{args.balanced_train_set}.joblib"
    output_kappa_full_path = os.path.join(output_path, output_kappa_model_name)
    best_kappa_model = all_models[best_all_kappa]
    kappa_stats = np.array([np.min(all_kappas), np.max(all_kappas), np.mean(all_kappas), np.median(all_kappas), np.std(all_kappas)])
    kappa_indices = np.array([np.argmin(all_kappas), np.argmax(all_kappas)])
    print(f"kappa_stats: {kappa_stats}; kappa_indices: {kappa_indices}")

    dump(best_bacc_model, output_bacc_full_path)
    dump(best_f1_model, output_f1_full_path)
    dump(best_mcc_model, output_mcc_full_path)
    dump(best_kappa_model, output_kappa_full_path)

    print("#" * 80)


def test(args):
    ds_path = "/root/JUPYTER/DATA/RAW_DATA/"
    print(f"Test Starts...\nargs.balanced_train_set: {args.balanced_train_set}")
    _, test_set, label_array = load_dataset(ds_path, args.balanced_train_set)
    if args.classifier_name == "FURIA":
        jvm.start(max_heap_size="50000m", packages=True)
    classifier = load(args.best_model_path)
    if args.classifier_name != "FURIA":
        X_test, y_test = label_encoder_df_X_cat(test_set)
        predicted_labels = classifier.predict(X_test)

        bacc_value = bacc(y_test, predicted_labels)
        f1_value = f1(y_test, predicted_labels, average='macro')
        mcc_value = mcc(y_test, predicted_labels)
        kappa_value = kappa(y_test, predicted_labels)
    else:
        data_instances, col_indices = createInstancesFromDataFrame(test_set)
        test_labels, predicted_labels = CP.testFuria(test_set, col_indices, classifier, data_instances)

        bacc_value = bacc(test_labels, predicted_labels)
        f1_value = f1(test_labels, predicted_labels, average='macro')
        mcc_value = mcc(test_labels, predicted_labels)
        kappa_value = kappa(test_labels, predicted_labels)

    print(f"Test BACC: {bacc_value}")
    print(f"Test F1: {f1_value}")
    print(f"Test MCC: {mcc_value}")
    print(f"Test KAPPA: {kappa_value}")


if __name__ == '__main__':
    parser_main = argparse.ArgumentParser('ClassifiersTrainMain', parents=[get_args_parser()])
    args_main = parser_main.parse_args()
    if args_main.classifier_name == "MLP" or args_main.classifier_name == "ADA" or args_main.classifier_name == "RF":
        train_save_classifiers_in_batch(args_main)
    elif args_main.classifier_name == "FURIA":
        train_save_furia_in_batch(args_main)

    # test(args_main)
