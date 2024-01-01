import numpy as np

from config import *
import Classification_Pipeline as CP


class DataSetPreparation:
    def __init__(self, ds_path, negative_path, count_path, threshold=10, add_neg_set=False):
        assert ds_path is not None and ds_path != "", "Please specify data set path"
        assert negative_path is not None and negative_path != "", "Please specify negtive data set path"
        assert count_path is not None and count_path != "", "Please specify class count path"
        
        dataframe = pds.read_csv(ds_path, na_values="?")
        count_dataframe = pds.read_csv(count_path, na_values="?")
        self.count_dict_majority = dict()
        count_dict_minority = dict()
        self.count_dict_test = dict()
        for rowIndex in range(count_dataframe.shape[0]):
            row = count_dataframe.iloc[rowIndex, :].to_numpy()
            if row[1] >=2:
                self.count_dict_majority[row[0]] = row[1]
                if row[1] >= threshold:
                    self.count_dict_test[row[0]] = row[1]
            else:
                count_dict_minority[row[0]] = row[1]

        self.transformed_data_set, self.label_encoder = self.preprocess(dataframe)
        self.transformed_data_set_neg = None
        if add_neg_set:
            neg_dataframe = pds.read_csv(negative_path, na_values="?")
            self.transformed_data_set_neg, _ = self.preprocess(neg_dataframe)
            self.transformed_data_set_neg = self.transformed_data_set_neg[:, 1:-1]
        
        classes = count_dict_minority.keys()
        for cls in classes:
            cls_code = self.label_encoder.transform([cls])[0]
            filtered_neg_samples = self.transformed_data_set[self.transformed_data_set[:, -1] == cls_code][:, 1:-1]
            if self.transformed_data_set_neg is None:
                self.transformed_data_set_neg = filtered_neg_samples
            else:
                self.transformed_data_set_neg = np.concatenate((self.transformed_data_set_neg, filtered_neg_samples), axis=0)
        
        self.pair_set = self.assemble_pair_set()
        
    
    def get_negtive_set(self):
        assert self.transformed_data_set_neg is not None, "Please specify the negtive set, currently it's empty"
        self.transformed_data_set_neg = T.from_numpy(self.transformed_data_set_neg)
        return self.transformed_data_set_neg
    
    
    def preprocess(self, ori_data_frame):
        scaler = MinMaxScaler()
        normalized_columns = scaler.fit_transform(ori_data_frame.iloc[:, 1:6])
        encoder = OneHotEncoder(sparse_output=False)
        transformed_categories = encoder.fit_transform(ori_data_frame.iloc[:, -3:-1])
        label_encoder = LabelEncoder()
        label_array = label_encoder.fit_transform(ori_data_frame.iloc[:, -1])
        label_array = np.expand_dims(label_array, axis=1)
        ori_index_col = np.expand_dims(ori_data_frame.iloc[:, 0].to_numpy(dtype=int), axis=1)
        data_frame_combined = np.concatenate((ori_index_col, normalized_columns, 
                                              ori_data_frame.iloc[:, 6:-3].to_numpy(), transformed_categories, label_array), axis=1)
        
        return data_frame_combined, label_encoder
        
    
    def assemble_pair_set(self):
        classes = self.count_dict_majority.keys()
        pair_set = None
        for cls in classes:
            cls_code = self.label_encoder.transform([cls])[0]
            filtered_set = self.transformed_data_set[self.transformed_data_set[:, -1] == cls_code]
            if cls in self.count_dict_test:
                test_set_per_cls = int(filtered_set.shape[0] // 10)
                complete_indices = range(filtered_set.shape[0])
                test_indices = np.random.choice(complete_indices, test_set_per_cls, False)
                train_indices = np.setdiff1d(complete_indices, test_indices)
                # if test_set is None:
                #     test_set = filtered_set[test_indices, :]
                # else:
                #     test_set = np.concatenate((test_set, filtered_set[test_indices, :]), axis=0)

                filtered_set = filtered_set[train_indices, 1:-1]
            else:
                filtered_set = filtered_set[:, 1:-1]
            pairs = np.array(list(itertools.combinations(filtered_set, 2)))
            if pair_set is None:
                pair_set = pairs
            else:
                pair_set = np.concatenate((pair_set, pairs), axis=0)
        # print(f"test_set shape: {test_set.shape}")
        # np.savetxt(test_out_path, test_set, delimiter=",", fmt='%.16f')
        pair_set = T.from_numpy(pair_set)
        
        return pair_set
    
    
    def save_train_test_set_and_classes(self, output_path, train_file_name, test_file_name, class_file_name, clscodes_file_name):
        classes = self.count_dict_majority.keys()
        test_set = None
        train_set = None
        classes_codes = []
        for cls in classes:
            cls_code = self.label_encoder.transform([cls])[0]
            classes_codes.append(int(cls_code))
            filtered_set = self.transformed_data_set[self.transformed_data_set[:, -1] == cls_code]
            if cls in self.count_dict_test:
                test_set_per_cls = int(filtered_set.shape[0] // 10)
                complete_indices = range(filtered_set.shape[0])
                test_indices = np.random.choice(complete_indices, test_set_per_cls, False)
                train_indices = np.setdiff1d(complete_indices, test_indices)
                if test_set is None:
                    test_set = filtered_set[test_indices, 1:]
                else:
                    test_set = np.concatenate((test_set, filtered_set[test_indices, 1:]), axis=0)
                if train_set is None:
                    train_set = filtered_set[train_indices, 1:]
                else:
                    train_set = np.concatenate((train_set, filtered_set[train_indices, 1:]), axis=0)
        
        train_set = T.from_numpy(train_set)
        test_set = T.from_numpy(test_set)
        classes = list(classes)
        print(f"classes size: {len(classes)}")
        train_out_path = os.path.join(output_path, train_file_name)
        test_out_path = os.path.join(output_path, test_file_name)
        class_out_path = os.path.join(output_path, class_file_name)
        clscodes_out_path = os.path.join(output_path, clscodes_file_name)
        print(f"train_set size: {train_set.size()}, test_set size: {test_set.size()}, clscodes length: {len(classes_codes)}")
        T.save(train_set, train_out_path)
        T.save(test_set, test_out_path)
        with open(class_out_path, 'w') as classesjson:
            json.dump(classes, classesjson)
        with open(clscodes_out_path, 'w') as clscodesjson:
            json.dump(classes_codes, clscodesjson)
        # T.save(classes, class_out_path)
    
    
    def save_pair_set(self, output_path, pair_file_name):
        pair_out_path = os.path.join(output_path, pair_file_name)
        print(f"pair_set size: {self.pair_set.size()}")
        T.save(self.pair_set, pair_out_path)
    
    
    def save_neg_set(self, output_path, neg_file_name):
        neg_out_path = os.path.join(output_path, neg_file_name)
        self.transformed_data_set_neg = T.from_numpy(self.transformed_data_set_neg)
        T.save(self.transformed_data_set_neg, neg_out_path)


class DataSet_Uns(Dataset):
    def __init__(self, ds_path, suffix,  add_neg_set=False):
        assert ds_path is not None and ds_path != "", "Please specify data set path"
        super(DataSet_Uns, self).__init__()
        
        pair_set_file_name = f"pair_set_limit{suffix}.pt"
        pair_set_path = os.path.join(ds_path, pair_set_file_name)

        if add_neg_set:
            neg_set_file_name = f"minority_negtive_limit{suffix}.pt"
        else:
            neg_set_file_name = f"minority_only_limit{suffix}.pt"
        neg_set_path = os.path.join(ds_path, neg_set_file_name)
        self.pair_set = T.load(pair_set_path)
        self.neg_set = T.load(neg_set_path)
        
    
    def __len__(self):
        return self.pair_set.size(0)
    
    
    def __getitem__(self, idx):
        assert idx < self.pair_set.size(0), "invalid index" 
        return self.pair_set[idx]


class DataSet_S:
    def __init__(self, ds_path, suffix):
        assert ds_path is not None and ds_path != "", "Please specify data set path"
        
        train_file_name = f"train_limit{suffix}.pt"
        train_path = os.path.join(ds_path, train_file_name)
        test_file_name = f"test_limit{suffix}.pt"
        test_path = os.path.join(ds_path, test_file_name)
        clscodes_file_name = f"clscodes_limit{suffix}.txt"
        clscodes_path = os.path.join(ds_path, clscodes_file_name)
        class_file_name = f"classes_limit{suffix}.txt"
        classes_path = os.path.join(ds_path, class_file_name)
        self.train_set = T.load(train_path)
        self.test_set = T.load(test_path)
        self.clscodes = None
        self.classes = None
        with open(clscodes_path, 'r') as clscodesjson:
            self.clscodes = json.load(clscodesjson)
            self.clscodes = T.tensor(self.clscodes)
        with open(classes_path, 'r') as classesjson:
            self.classes = json.load(classesjson)
        
        self.val_set = None
        
    
    def setBalancedTrainValidationSet(self, threshold=10):
        train_set = None
        val_set = None
        for cls_code in self.clscodes:
            filtered_set = self.train_set[self.train_set[:, -1] == cls_code].numpy()
            sample_size = filtered_set.shape[0]
            if sample_size == 0:
                continue
            if sample_size > threshold:
                complete_indices = np.random.choice(range(sample_size), threshold, False)
                train_num = int(threshold * 0.8)
                train_indices = np.random.choice(complete_indices, train_num, False)
                val_indices = np.setdiff1d(complete_indices, train_indices)
                if train_set is None:
                    train_set = filtered_set[train_indices, :]
                else:
                    train_set = np.concatenate((train_set, filtered_set[train_indices, :]), axis=0)
                if val_set is None:
                    val_set = filtered_set[val_indices, :]
                else:
                    val_set = np.concatenate((val_set, filtered_set[val_indices, :]), axis=0)
            
        self.train_set = T.from_numpy(train_set)
        self.val_set = T.from_numpy(val_set)


class DataSet_Torch(Dataset):
    def __init__(self, X, y):
        super(DataSet_Torch, self).__init__()
        self.data = T.from_numpy(X)
        self.labels = T.from_numpy(y)
        # self.num_classes = T.unique(self.labels).size(0)
        # print(f"self.num_classes: {self.num_classes}")
        
        
    def __len__(self):
        return self.data.size(0)
    
    
    def __getitem__(self, idx):
        assert idx < self.data.size(0), "invalid index"
        return self.data[idx, :], self.labels[idx]
    

def get_type_dict():
    type_dict = {"ID": "float64", "SampleSize": "float64", "NoAttr": "float64", "NoNAttr": "float64", "NoCAttr": "float64", "NoCL": "float64", "TimeEfficiency": "float64",
                 "IEXH": "float64", "IACC": "float64", "NRMSE_MIN": "float64", "NRMSE_Median": "float64", "NRMSE_MAX": "float64", "ChangeDispersionMIN": "float64",
                 "ChangeDispersionMedian": "float64", "ChangeDispersionMAX": "float64", "ChangeUDS": "float64", "ChangeAdaCCBACC": "float64", "ChangeAdaCCF1Macro": "float64",
                 "ChangeAdaCCMCC": "float64", "ChangeAdaCCKappa": "float64","ChangeAdaCNNBACC": "float64", "ChangeAdaCNNF1Macro": "float64", "ChangeAdaCNNMCC": "float64",
                 "ChangeAdaCNNKappa": "float64", "ChangeAdaRENNBACC": "float64", "ChangeAdaRENNF1Macro": "float64", "ChangeAdaRENNMCC": "float64", "ChangeAdaRENNKappa": "float64",
                 "ChangeAdaALLKNNBACC": "float64", "ChangeAdaALLKNNF1Macro": "float64", "ChangeAdaALLKNNMCC": "float64", "ChangeAdaALLKNNKappa": "float64", "ChangeAdaIHTBACC": "float64",
                 "ChangeAdaIHTF1Macro": "float64", "ChangeAdaIHTMCC": "float64", "ChangeAdaIHTKappa": "float64", "ChangeAdaNCRBACC": "float64", "ChangeAdaNCRF1Macro": "float64", 
                 "ChangeAdaNCRMCC": "float64", "ChangeAdaNCRKappa": "float64", "ChangeAdaOSSBACC": "float64", "ChangeAdaOSSF1Macro": "float64", "ChangeAdaOSSMCC": "float64", 
                 "ChangeAdaOSSKappa": "float64", "ChangeAdaRUSBACC": "float64", "ChangeAdaRUSF1Macro": "float64", "ChangeAdaRUSMCC": "float64", "ChangeAdaRUSKappa": "float64", 
                 "ChangeAdaTomekLinksBACC": "float64", "ChangeAdaTomekLinksF1Macro": "float64", "ChangeAdaTomekLinksMCC": "float64", "ChangeAdaTomekLinksKappa": "float64", 
                 "ChangeAdaROSBACC": "float64", "ChangeAdaROSF1Macro": "float64", "ChangeAdaROSMCC": "float64", "ChangeAdaROSKappa": "float64", "ChangeAdaSMOTEBACC": "float64", 
                 "ChangeAdaSMOTEF1Macro": "float64", "ChangeAdaSMOTEMCC": "float64", "ChangeAdaSMOTEKappa": "float64", "ChangeAdaSMOTEBL1BACC": "float64",
                 "ChangeAdaSMOTEBL1F1Macro": "float64","ChangeAdaSMOTEBL1MCC": "float64", "ChangeAdaSMOTEBL1Kappa": "float64", "ChangeAdaSMOTEBL2BACC": "float64", 
                 "ChangeAdaSMOTEBL2F1Macro": "float64", "ChangeAdaSMOTEBL2MCC": "float64", "ChangeAdaSMOTEBL2Kappa": "float64", "ChangeAdaSMOTESVMBACC": "float64", 
                 "ChangeAdaSMOTESVMF1Macro": "float64", "ChangeAdaSMOTESVMMCC": "float64", "ChangeAdaSMOTESVMKappa": "float64", "ChangeAdaSMOTEENNBACC": "float64", 
                 "ChangeAdaSMOTEENNF1Macro": "float64", "ChangeAdaSMOTEENNMCC": "float64", "ChangeAdaSMOTEENNKappa": "float64", "ChangeAdaSMOTETomekBACC": "float64", 
                 "ChangeAdaSMOTETomekF1Macro": "float64", "ChangeAdaSMOTETomekMCC": "float64", "ChangeAdaSMOTETomekKappa": "float64", "ChangeRFCCBACC": "float64",
                 "ChangeRFCCF1Macro": "float64", "ChangeRFCCMCC": "float64", "ChangeRFCCKappa": "float64", "ChangeRFCNNBACC": "float64", "ChangeRFCNNF1Macro": "float64",
                 "ChangeRFCNNMCC": "float64", "ChangeRFCNNKappa": "float64", "ChangeRFRENNBACC": "float64", "ChangeRFRENNF1Macro": "float64", "ChangeRFRENNMCC": "float64", 
                 "ChangeRFRENNKappa": "float64", "ChangeRFALLKNNBACC": "float64", "ChangeRFALLKNNF1Macro": "float64", "ChangeRFALLKNNMCC": "float64", "ChangeRFALLKNNKappa": "float64", 
                 "ChangeRFIHTBACC": "float64", "ChangeRFIHTF1Macro": "float64", "ChangeRFIHTMCC": "float64", "ChangeRFIHTKappa": "float64", "ChangeRFNCRBACC": "float64", 
                 "ChangeRFNCRF1Macro": "float64", "ChangeRFNCRMCC": "float64", "ChangeRFNCRKappa": "float64", "ChangeRFOSSBACC": "float64", "ChangeRFOSSF1Macro": "float64", 
                 "ChangeRFOSSMCC": "float64", "ChangeRFOSSKappa": "float64", "ChangeRFRUSBACC": "float64", "ChangeRFRUSF1Macro": "float64", "ChangeRFRUSMCC": "float64",
                 "ChangeRFRUSKappa": "float64", "ChangeRFTomekLinksBACC": "float64", "ChangeRFTomekLinksF1Macro": "float64", "ChangeRFTomekLinksMCC": "float64",
                 "ChangeRFTomekLinksKappa": "float64", "ChangeRFROSBACC": "float64", "ChangeRFROSF1Macro": "float64", "ChangeRFROSMCC": "float64", "ChangeRFROSKappa": "float64", 
                 "ChangeRFSMOTEBACC": "float64", "ChangeRFSMOTEF1Macro": "float64", "ChangeRFSMOTEMCC": "float64", "ChangeRFSMOTEKappa": "float64", "ChangeRFSMOTEBL1BACC": "float64", 
                 "ChangeRFSMOTEBL1F1Macro": "float64", "ChangeRFSMOTEBL1MCC": "float64", "ChangeRFSMOTEBL1Kappa": "float64", "ChangeRFSMOTEBL2BACC": "float64", 
                 "ChangeRFSMOTEBL2F1Macro": "float64", "ChangeRFSMOTEBL2MCC": "float64", "ChangeRFSMOTEBL2Kappa": "float64", "ChangeRFSMOTESVMBACC": "float64", 
                 "ChangeRFSMOTESVMF1Macro": "float64", "ChangeRFSMOTESVMMCC": "float64", "ChangeRFSMOTESVMKappa": "float64", "ChangeRFSMOTEENNBACC": "float64",
                 "ChangeRFSMOTEENNF1Macro": "float64", "ChangeRFSMOTEENNMCC": "float64", "ChangeRFSMOTEENNKappa": "float64", "ChangeRFSMOTETomekBACC": "float64", 
                 "ChangeRFSMOTETomekF1Macro": "float64", "ChangeRFSMOTETomekMCC": "float64", "ChangeRFSMOTETomekKappa": "float64", "ChangeENSEECBACC": "float64",
                 "ChangeENSEECF1Macro": "float64", "ChangeENSEECMCC": "float64", "ChangeENSEECKappa": "float64", "ChangeENSRUSBCBACC": "float64", "ChangeENSRUSBCF1Macro": "float64", 
                 "ChangeENSRUSBCMCC": "float64", "ChangeENSRUSBCKappa": "float64", "ChangeENSBBCBACC": "float64", "ChangeENSBBCF1Macro": "float64", "ChangeENSBBCMCC": "float64", 
                 "ChangeENSBBCKappa": "float64", "ChangeENSBRFCBACC": "float64", "ChangeENSBRFCF1Macro": "float64", "ChangeENSBRFCMCC": "float64", "ChangeENSBRFCKappa": "float64", 
                 "ChangeIntrinsicF1": "float64", "ChangeIntrinsicF1v": "float64", "ChangeIntrinsicF2": "float64", "ChangeIntrinsicF3": "float64", "ChangeIntrinsicF4": "float64", 
                 "ChangeIntrinsicL1": "float64", "ChangeIntrinsicL2": "float64", "ChangeIntrinsicN1": "float64", "ChangeIntrinsicN2": "float64", "ChangeIntrinsicN3": "float64", 
                 "ChangeIntrinsicL3": "float64", "ChangeIntrinsicN4": "float64", "ChangeIntrinsicT1": "float64", "ChangeIntrinsicT2": "float64", "MissingRate": "float64", 
                 "MissingMechanism": "category", "MissingPattern": "category", "MDHMethod": "category"}   # totally 178 columns, 177 columns without "ID"  
    
    return type_dict


def assemble_dataset(path_root, balanced=False):
    suffix = 10
    ds_path = os.path.join(path_root, f"train_supervised_limit{suffix}.csv")
    count_path = os.path.join(path_root, f"count_limit{suffix}.csv")
    
    data_frame = pds.read_csv(ds_path, na_values="?")
    count_dataframe = pds.read_csv(count_path, na_values="?")
    count_dict_majority = dict()
    count_dict_minority = dict()
    count_dict_test = dict()
    for rowIndex in range(count_dataframe.shape[0]):
        row = count_dataframe.iloc[rowIndex, :].to_numpy()
        if row[1] >=2:
            count_dict_majority[row[0]] = row[1]
            if row[1] >= 10:
                count_dict_test[row[0]] = row[1]
        else:
            count_dict_minority[row[0]] = row[1]
    
    type_dict = get_type_dict()
    data_frame = data_frame.astype(type_dict)
    num_classes = pds.unique(data_frame.iloc[:, -1]).shape[0]
    print(f"data_frame shape: {data_frame.shape}; number of classes in data_frame: {num_classes}")

    # get train test set imbalanced
    classes = count_dict_majority.keys()
    test_set = None
    train_set = None
    
    for cls in classes:
        filtered_set = data_frame.loc[data_frame.iloc[:, -1] == cls]
        if cls in count_dict_test:
            test_set_per_cls = int(filtered_set.shape[0] // 10)
            complete_indices = range(filtered_set.shape[0])
            test_indices = np.random.choice(complete_indices, test_set_per_cls, False)
            train_indices = np.setdiff1d(complete_indices, test_indices)
            if test_set is None:
                test_set = filtered_set.iloc[test_indices, 1:]
            else:
                test_set = pds.concat([test_set, filtered_set.iloc[test_indices, 1:]], axis=0, ignore_index=True)
            if train_set is None:
                train_set = filtered_set.iloc[train_indices, 1:]
            else:
                train_set = pds.concat([train_set, filtered_set.iloc[train_indices, 1:]], axis=0, ignore_index=True)
    type_dict.pop("ID")
    test_set = test_set.astype(type_dict)
    
    if not balanced:
        result_train_set =  train_set.astype(type_dict)
    else:
        # balancing train set
        balanced_train_set = None
        threshold = 9

        for cls in classes:
            filtered_set = train_set.loc[train_set.iloc[:, -1] == cls]
            sample_size = filtered_set.shape[0]
            if sample_size == 0:
                continue
            if sample_size >= threshold:
                train_indices = np.random.choice(range(sample_size), threshold, False)
                # train_num = threshold
                # train_indices = np.random.choice(complete_indices, train_num, False)
                if balanced_train_set is None:
                    balanced_train_set = filtered_set.iloc[train_indices, :]
                else:
                    balanced_train_set = pds.concat([balanced_train_set, filtered_set.iloc[train_indices, :]], axis=0, ignore_index=True)

        result_train_set = balanced_train_set.astype(type_dict)
    
    return result_train_set, test_set, num_classes
    

def assemble_real_test(path_root, threshold):
    suffix = 10
    ds_path = os.path.join(path_root, f"train_supervised_limit{suffix}.csv")
    count_path = os.path.join(path_root, f"count_limit{suffix}.csv")

    data_frame = pds.read_csv(ds_path, na_values="?")
    count_dataframe = pds.read_csv(count_path, na_values="?")
    count_dict_test = dict()
    for rowIndex in range(count_dataframe.shape[0]):
        row = count_dataframe.iloc[rowIndex, :].to_numpy()
        if row[1] < threshold:
            count_dict_test[row[0]] = row[1]

    type_dict = get_type_dict()
    data_frame = data_frame.astype(type_dict)
    num_classes = pds.unique(data_frame.iloc[:, -1]).shape[0]
    print(f"data_frame shape: {data_frame.shape}; number of classes in data_frame: {num_classes}")

    classes = count_dict_test.keys()
    real_test_set = None

    for cls in classes:
        filtered_set = data_frame.loc[data_frame.iloc[:, -1] == cls]
        if cls in count_dict_test:
            complete_indices = range(filtered_set.shape[0])
            real_test_indices = complete_indices
            if real_test_set is None:
                real_test_set = filtered_set.iloc[real_test_indices, 1:]
            else:
                real_test_set = pds.concat([real_test_set, filtered_set.iloc[real_test_indices, 1:]], axis=0, ignore_index=True)
    type_dict.pop("ID")
    real_test_set = real_test_set.astype(type_dict)
    print(f"real_test_set shape: {real_test_set.shape}")

    return real_test_set, num_classes


def preprocess_dataset(ori_data_frame, one_hot=False):
    scaler = MinMaxScaler()
    normalized_columns = scaler.fit_transform(ori_data_frame.iloc[:, :5])
    if not one_hot:
        encoder = OrdinalEncoder()
        label_encoder = Label_Encoder(one_hot)
        label_array = label_encoder.transform(ori_data_frame)

    else:
        encoder = OneHotEncoder(sparse_output=False)
        label_encoder = Label_Encoder(one_hot)
        label_array = label_encoder.transform(ori_data_frame)
        
    transformed_categories = encoder.fit_transform(ori_data_frame.iloc[:, -3:-1])

    if not one_hot:
        label_array = np.expand_dims(label_array, axis=1)
    # ori_index_col = np.expand_dims(ori_data_frame.iloc[:, 0].to_numpy(dtype=int), axis=1)
    data_frame_combined = np.concatenate((normalized_columns, ori_data_frame.iloc[:, 5:-3].to_numpy(), transformed_categories, label_array), axis=1)
    
    return data_frame_combined, encoder, label_encoder


def label_encoder_df_X_cat(data_frame):
    data_frame_combined, _, _ = preprocess_dataset(data_frame)
    X = data_frame_combined[:, :-1]
    y = data_frame_combined[:, -1]
    
    return X, y


def label_encoder_X_cat(X_cat):
    encoder = OrdinalEncoder()
    transformed_categories = encoder.fit_transform(X_cat[:, -2:])
    X = np.concatenate((X_cat[:, :-2], transformed_categories), axis=1)
    
    return X, encoder


def one_hot_encode_df_X_cat(data_frame):
    data_frame_combined, _, _ = preprocess_dataset(data_frame)
    y = data_frame_combined[:, -1]
    encoder = OneHotEncoder(sparse_output=False)
    transformed_categories = encoder.fit_transform(data_frame_combined[:, -3:-1])
    X = np.concatenate((data_frame_combined[:, :-3], transformed_categories), axis=1)
    
    return X, y, encoder


def one_hot_encode_X_cat(X_cat):
    encoder = OneHotEncoder(sparse_output=False)
    transformed_categories = encoder.fit_transform(X_cat[:, -2:])
    X = np.concatenate((X_cat[:, :-2], transformed_categories), axis=1)
    
    return X, encoder


def reverse_one_hot_X_cat(X_cat, encoder):
    ori_cat = encoder.inverse_transform(X_cat[:, -7:])
    X = np.concatenate((X_cat[:, :-7], ori_cat), axis=1)
    
    return X


class Label_Encoder:
    def __init__(self, one_hot):
        self.one_hot = one_hot
        self.label_encoder = None

    def transform(self, data_frame):
        if not self.one_hot:
            label_array = data_frame.iloc[:, -1].cat.codes.to_numpy()
            self.label_encoder = {idx: name for idx, name in enumerate(data_frame.iloc[:, -1].cat.categories)}
        else:
            self.label_encoder = OneHotEncoder(sparse_output=False)
            labels = np.expand_dims(data_frame.iloc[:, -1].to_numpy(), axis=1)
            label_array = self.label_encoder.fit_transform(labels)

        return label_array

    def inverse_transform(self, labels):
        if not self.one_hot:
            ori_labels = np.array([])
            for label in labels:
                ori_labels = np.append(ori_labels, self.label_encoder[label])
        else:
            ori_labels = self.label_encoder.inverse_transform(labels)

        return ori_labels


# java weka furia utility functions
def preprocess_DataFrame(dataFrame):
    data_frame_combined, encoder, label_encoder = preprocess_dataset(dataFrame)
    ori_cat_columns = encoder.inverse_transform(data_frame_combined[:, -3:-1].astype(dtype=np.int32))
    ori_labels = label_encoder.inverse_transform(data_frame_combined[:, -1].astype(dtype=np.int32))
    ori_labels = np.expand_dims(ori_labels, axis=1)
    preprocessed_data = np.concatenate((data_frame_combined[:, :-3], ori_cat_columns, ori_labels), axis=1)
    preprocessed_data_frame = pds.DataFrame(data=preprocessed_data, columns=dataFrame.columns)
    preprocessed_data_frame = preprocessed_data_frame.astype(dtype=dataFrame.dtypes)
    print(f"preprocessed data frame shape: {preprocessed_data_frame.shape}")
    
    return preprocessed_data_frame
    

def getInstancesName():
    instanceId = datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f")
    name = "DataSet" + instanceId
    return name


def createInstancesFromDataFrame(dataFrame):
    cols = list()
    colIndices = list()
    colDict = dict(zip(dataFrame.columns.values, np.arange(dataFrame.columns.shape[0])))
    
    for col in dataFrame.columns:
        colData = dataFrame[col]
        if colData.dtype == "category":
            colDataCategories = colData.cat.categories.astype(dtype="str", copy=False)
            column = Attribute.create_nominal(col, colDataCategories)
            colIndices.append(colDict[col])
        else:
            column = Attribute.create_numeric(col)
        cols.append(column)
    instances = Instances.create_instances(getInstancesName(), cols, 0)
    
    # preprocessing
    data_frame_combined, encoder, label_encoder = preprocess_dataset(dataFrame)
    ori_cat_columns = encoder.inverse_transform(data_frame_combined[:, -3:-1].astype(dtype=np.int32))
    ori_labels = label_encoder.inverse_transform(data_frame_combined[:, -1].astype(dtype=np.int32))
    ori_labels = np.expand_dims(ori_labels, axis=1)
    preprocessed_data = np.concatenate((data_frame_combined[:, :-3], ori_cat_columns, ori_labels), axis=1)
    preprocessed_data_frame = pds.DataFrame(data=preprocessed_data, columns=dataFrame.columns)
    preprocessed_data_frame = preprocessed_data_frame.astype(dtype=dataFrame.dtypes)
    print(f"preprocessed data frame shape: {preprocessed_data_frame.shape}")
    
    index = 0
    while index < preprocessed_data_frame.shape[0]:
        rowArr = preprocessed_data_frame[index:index+1].values.squeeze()
        npRowArr = np.array([], dtype=np.float64)
        for colIndex in colIndices:
            rowArr[colIndex] = preprocessed_data_frame.iloc[index:index+1, colIndex].cat.codes
        for rowArrIndex in range(rowArr.shape[0]):
            npRowArr = np.append(npRowArr, rowArr[rowArrIndex])
        # rowArr = rowArr.astype(dtype=np.float64)
        ins_row = Instance.create_instance(rowArr)
        instances.add_instance(ins_row, index)
        index += 1
    
    return instances, colIndices


def createRebalancedInstancesFromDataFrame(dataFrame, sampler_name, params):
    cols = list()
    colIndices = list()
    colDict = dict(zip(dataFrame.columns.values, np.arange(dataFrame.columns.shape[0])))
    # build the instances structure
    for col in dataFrame.columns:
        colData = dataFrame[col]
        if colData.dtype == "category":
            colDataCategories = colData.cat.categories.astype(dtype="str", copy=False)
            column = Attribute.create_nominal(col, colDataCategories)
            colIndices.append(colDict[col])
        else:
            column = Attribute.create_numeric(col)
        cols.append(column)
    instances = Instances.create_instances(getInstancesName(), cols, 0)
    
    # rebalancing
    encoder = None
    class_encoder = None
    if sampler_name != "SMOTENC":
        data_frame_combined, encoder, _ = preprocess_dataset(dataFrame, True)
        X = data_frame_combined[:, :181]
        class_encoder = LabelEncoder()
        y = class_encoder.fit_transform(dataFrame.iloc[:, -1])
    else:
        X = dataFrame.iloc[:, :-1]
        y = dataFrame.iloc[:, -1]
    sampler = CP.getSampler(sampler_name, params)
    X_res, y_res = sampler.fit_resample(X, y)
    if sampler_name != "SMOTENC":
        ori_cat_columns = encoder.inverse_transform(X_res[:, -7:])
        ori_labels = class_encoder.inverse_transform(y_res)
        ori_labels = np.expand_dims(ori_labels, axis=1)
        rebalanced_data = np.concatenate((X_res[:, :-7], ori_cat_columns, ori_labels), axis=1)
        rebalanced_data_frame = pds.DataFrame(data=rebalanced_data, columns=dataFrame.columns)
        rebalanced_data_frame = rebalanced_data_frame.astype(dtype=dataFrame.dtypes)
    else:
        rebalanced_data_frame = pds.concat([X_res, y_res], axis=1, ignore_index=True)
    print(f"rebalanced data frame shape: {rebalanced_data_frame.shape}")
    
    index = 0
    while index < rebalanced_data_frame.shape[0]:
        rowArr = rebalanced_data_frame[index:index+1].values.squeeze()
        npRowArr = np.array([], dtype=np.float64)
        for colIndex in colIndices:
            rowArr[colIndex] = rebalanced_data_frame.iloc[index:index+1, colIndex].cat.codes
        for rowArrIndex in range(rowArr.shape[0]):
            npRowArr = np.append(npRowArr, rowArr[rowArrIndex])
        # rowArr = rowArr.astype(dtype=np.float64)
        ins_row = Instance.create_instance(rowArr)
        instances.add_instance(ins_row, index)
        index += 1
    
    return instances, colIndices

