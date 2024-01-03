from config import *
import DataSetLoader as DSL


def getSampler(which, params):
    sampler = None
    if which == "ClusterCentroids":
        sampler = under_sampling.ClusterCentroids(estimator = params["estimator"])
    elif which == "OneSidedSelection":
        sampler = under_sampling.OneSidedSelection()
    elif which == "NeighbourhoodCleaningRule":
        sampler = under_sampling.NeighbourhoodCleaningRule(kind_sel = params["kind_sel"])
    elif which == "EditedNearestNeighbours":
        sampler = under_sampling.EditedNearestNeighbours(kind_sel = params["kind_sel"])
    elif which == "AllKNN":
        sampler = under_sampling.AllKNN(kind_sel = params["kind_sel"])
    elif which == "TomekLinks":
        sampler = under_sampling.TomekLinks()
    elif which == "SMOTE":
        sampler = over_sampling.SMOTE()
    elif which == "SMOTENC":
        sampler = over_sampling.SMOTENC(categorical_features="auto", categorical_encoder=params["categorical_encoder"])
    elif which == "BorderlineSMOTE":  # kind{“borderline-1”, “borderline-2”}
        sampler = over_sampling.BorderlineSMOTE(kind = params["kind"])
    elif which == "SMOTEENN":
        sampler = combine.SMOTEENN(smote = params["smote"], enn = params["enn"])
    elif which == "SMOTETomek":
        sampler = combine.SMOTETomek(smote = params["smote"], tomek = params["tomek"])
    
    return sampler


def runImbalancedClassificationPipeline(X_train, y_train, X_test, y_test, columns, dtypes, encoder_train, encoder_test,
                                        sampler_name, params, classifier_name, classifier_params, balanced=False):
    # bacc_value, f1_value, mcc_value, kappa_value, model = None, None, None, None, None
    if balanced:
        if classifier_name != "FURIA":
            X_res, y_res = X_train, y_train
            classifier = None
            if classifier_name == "ADA":
                classifier = AdaBoostClassifier(n_estimators=classifier_params["n_est"])
            elif classifier_name == "RF":
                classifier = RandomForestClassifier(n_estimators=classifier_params["n_est"], oob_score=True)
            elif classifier_name == "MLP":
                X_res, _ = DSL.one_hot_encode_X_cat(X_res)
                X_test, _ = DSL.one_hot_encode_X_cat(X_test) 
                classifier = MLPClassifier(hidden_layer_sizes=classifier_params["hidden"], batch_size=classifier_params["batch"], max_iter=500,
                                           nesterovs_momentum=classifier_params["nesterov"], early_stopping=True)
            
            classifier = classifier.fit(X_res, y_res)
            predicted_labels = classifier.predict(X_test)

            bacc_value = bacc(y_test, predicted_labels)
            f1_value = f1(y_test, predicted_labels, average='macro')
            mcc_value = mcc(y_test, predicted_labels)
            kappa_value = kappa(y_test, predicted_labels)
            model = classifier
            
        else:
            X_res, y_res = X_train, y_train
            
            train_set = pds.DataFrame(data=pds.concat([X_res, y_res], axis=1, ignore_index=True).to_numpy(), columns=columns)
            train_set = train_set.astype(dtype=dtypes)
            test_set = pds.DataFrame(data=pds.concat([X_test, y_test], axis=1, ignore_index=True).to_numpy(), columns=columns)
            test_set = test_set.astype(dtype=dtypes)
            test_set = DSL.preprocess_DataFrame(test_set)
            classifier, data_instances, col_indices = createFuriaClassifier(train_set, sampler_name, params, balanced)
            test_labels, predicted_labels = testFuria(test_set, col_indices, classifier, data_instances)
            
            bacc_value = bacc(test_labels, predicted_labels)
            f1_value = f1(test_labels, predicted_labels, average='macro')
            mcc_value = mcc(test_labels, predicted_labels)
            kappa_value = kappa(test_labels, predicted_labels)
            model = classifier
            
    else:
        if classifier_name != "FURIA":
            sampler = getSampler(sampler_name, params)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            classifier = None
            if classifier_name == "ADA":
                if sampler_name != "SMOTENC":
                    X_res = DSL.reverse_one_hot_X_cat(X_res, encoder_train)
                    X_res, _ = DSL.label_encoder_X_cat(X_res)
                    X_test = DSL.reverse_one_hot_X_cat(X_test, encoder_test)
                    X_test, _ = DSL.label_encoder_X_cat(X_test)
                else:
                    train_set = pds.concat([X_res, y_res], axis=1, ignore_index=True)
                    X_res, y_res = DSL.label_encoder_df_X_cat(train_set)
                    test_set = pds.concat([X_test, y_test], axis=1, ignore_index=True)
                    X_test, y_test = DSL.label_encoder_df_X_cat(test_set)
                classifier = AdaBoostClassifier(n_estimators=classifier_params["n_est"])
            elif classifier_name == "RF":
                if sampler_name != "SMOTENC":
                    X_res = DSL.reverse_one_hot_X_cat(X_res, encoder_train)
                    X_res, _ = DSL.label_encoder_X_cat(X_res)
                    X_test = DSL.reverse_one_hot_X_cat(X_test, encoder_test)
                    X_test, _ = DSL.label_encoder_X_cat(X_test)
                else:
                    train_set = pds.concat([X_res, y_res], axis=1, ignore_index=True)
                    X_res, y_res = DSL.label_encoder_df_X_cat(train_set)
                    test_set = pds.concat([X_test, y_test], axis=1, ignore_index=True)
                    X_test, y_test = DSL.label_encoder_df_X_cat(test_set)
                classifier = RandomForestClassifier(n_estimators=classifier_params["n_est"], oob_score=True)
            elif classifier_name == "MLP":
                if sampler_name == "SMOTENC":
                    train_set = pds.concat([X_res, y_res], axis=1, ignore_index=True)
                    X_res, y_res, _ = DSL.one_hot_encode_df_X_cat(train_set)
                    test_set = pds.concat([X_test, y_test], axis=1, ignore_index=True)
                    X_test, y_test, _ = DSL.one_hot_encode_df_X_cat(test_set)
                classifier = MLPClassifier(hidden_layer_sizes=classifier_params["hidden"], batch_size=classifier_params["batch"], max_iter=500,
                                           nesterovs_momentum=classifier_params["nesterov"], early_stopping=True)
            
            classifier = classifier.fit(X_res, y_res)
            predicted_labels = classifier.predict(X_test)

            bacc_value = bacc(y_test, predicted_labels)
            f1_value = f1(y_test, predicted_labels, average='macro')
            mcc_value = mcc(y_test, predicted_labels)
            kappa_value = kappa(y_test, predicted_labels)
            model = classifier
            
        else:
            X_res, y_res = X_train, y_train
            
            train_set = pds.DataFrame(data=pds.concat([X_res, y_res], axis=1, ignore_index=True).to_numpy(), columns=columns)
            train_set = train_set.astype(dtype=dtypes)
            test_set = pds.DataFrame(data=pds.concat([X_test, y_test], axis=1, ignore_index=True).to_numpy(), columns=columns)
            test_set = test_set.astype(dtype=dtypes)
            test_set = DSL.preprocess_DataFrame(test_set)
            classifier, data_instances, col_indices = createFuriaClassifier(train_set, sampler_name, params, balanced)
            test_labels, predicted_labels = testFuria(test_set, col_indices, classifier, data_instances)
            
            bacc_value = bacc(test_labels, predicted_labels)
            f1_value = f1(test_labels, predicted_labels, average='macro')
            mcc_value = mcc(test_labels, predicted_labels)
            kappa_value = kappa(test_labels, predicted_labels)
            model = classifier
    
    return bacc_value, f1_value, mcc_value, kappa_value, model


# java Furia classifier
def createFuriaClassifier(train_set, sampler_name, params, balanced=False):
    class_name = "weka.classifiers.rules.FURIA"
    options = ["-F", "3", "-O", "10"]
    classifier = Classifier(classname=class_name, options=options)
    if balanced:
        data_instances, col_indices = DSL.createInstancesFromDataFrame(train_set)
    else:
        data_instances, col_indices = DSL.createRebalancedInstancesFromDataFrame(train_set, sampler_name, params)
    data_instances.class_index = train_set.columns.shape[0] - 1
    classifier.build_classifier(data_instances)
    
    return classifier, data_instances, col_indices
    

def testFuria(test_set, col_indices, classifier, data_instances):
    predicted_labels = np.array([])
    print(f"test_set rows: {test_set.shape[0]}")
    for test_index in range(test_set.shape[0]):
        test_row = test_set.iloc[test_index:test_index+1, :]
        classified_row = test_row.values.squeeze()
        for col_index in col_indices:
            classified_row[col_index] = test_row.iloc[:, col_index].cat.codes.array[0]
        classified_row = classified_row.astype(dtype=np.float64)
        classified_instance = Instance.create_instance(classified_row)
        classified_instance.set_missing(test_row.shape[1] - 1)
        classified_instance.dataset = data_instances
        predicted_label = classifier.classify_instance(classified_instance)
        if math.isnan(predicted_label):
            num_classes = test_set.iloc[:, -1].cat.categories.shape[0]
            predicted_label = random.randint(0, num_classes-1)
        
        predicted_labels = np.append(predicted_labels, predicted_label)
        
    return test_set.iloc[:, -1].cat.codes.values, predicted_labels