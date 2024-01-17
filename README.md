# Deep Contrastive Self-Supervised Method Selection Framework (DCSSMS) For Missing Data Handling
The official repository for a deep contrastive self-supervised Missing Data Handling method selection framework.
The structure of the directories:
- **"./DATA/"**, all the dataset files for self-supervised training and linear evaluation training.
	- **"./DATA/Linear_Evaluation/"**, dataset files for linear evaluation fine-tuning.
	- **"./DATA/Self-supervised_Training/"**, dataset files for DCSSMS embedding training.
- **"./Framework/"**, all the code files for the DCSSMS framework training and fine-tuning.
- **"./Embedding/"**, the best pre-trained DCSSMS embedding file.
- **"./ExpResults/"**, more experimental results
    - **"./ExpResults/Linear_Evaluation_Ablation/"**, details about all the linear evaluation experiments, including ablation studies.
	- **"./ExpResults/MDHMS/"**, details about all the MDH method selection experiments.
	- **"./ExpResults/MDHMS/MSD^raw", we make our self-supervised training dataset, i.e., MSD^raw, publicly accessible. This dataset can also provide insights to select the $442$ investigated MDH methods under different MDH circumstances.**
- **"./Supplementary_Material/"**, appendixes file including the details about the 56 real-world imbalanced classification datasets.
- **"./FURIA/"**, the jar file for Furia-related methods, i.e., FIII and Furia-selector.

# DCSSMS Embedding Training
To train the DCSSMS embedding network, follow the steps as shown below:
1. clone the DCSSMS GitHub repository.
1. in CLI, change current path to **"DCSSMS/"**.
1. use the following python script to start training:<br/>
   python ./Framework/MainBYOL.py --gpu_id \[0\] --data_dir "./DATA/Self-supervised_Training/" --init_lr \[0.030280\] --max_lr \[1.287572\] --batch_size \[512\] --num_layers \[3\] --out_sizes \[256 512 1024\] --output_dir \[_"specify your own directory"_\] --use_momentum \[True/False\] \[> _"specify your own log file path"_ 2>&1 &\]
   - "--gpu_id", specify the gpu id.
   - "--data_dir", specify the directory of the embedding training dataset.
   - "--init_lr", specify the initial learning rate for the OneCycle learning rate scheduler.
   - "--max-lr", specify the maximum learning rate for the OneCycle learning rate scheduler.
   - "--batch_size", specify the batch size for the DCSSMS embedding training.
   - "--num_layers", specify the number of layers for the Over-complete encoder network.
   - "--out_sizes", specify the sizes of the hidden layers for the Over-complete encoder network, e.g., 256 512 1024 for 3 layers, 256 512 1024 2048 for 4 layers Over-complete encoder network, etc.
   - "--output_dir", specify the output directory for the best learned embedding model.
   - "--use_momentum", specify whether to use the Stop-gradient mechanism for the "Target" network.
   - "\[> _"specify your own log file path"_ 2>&1 &\]", specify whether to run the script in background, redirect stdout, stderr to log file, e.g., "> ./training512_10_3_true.log 2>&1 &".

# DCSSMS Linear Evaluation Training
To fine-tune the DCSSMS embedding network according to the linear evaluation protocol, follow the steps as shown below:
1. clone the DCSSMS GitHub repository, (if you have already cloned it, directly go to next step).
1. in CLI, change current path to **"DCSSMS/"**, (if you have already done it, directly go to next step).
1. download the pretrained DCSSMS embedding model and put it into the **"./Embedding/"** folder.
1. use the following python script to start fine-tuning:<br/>
   python ./Framework/LinearEvaluation.py --gpu_id \[0\] --data_dir "./DATA/Linear_Evaluation/" --embedding_dir ./Embedding/best_model_8192_10_3_True.pth --init_lr \[0.03\] --weight_decay \[1e-4\] --batch_size \[128\] --num_layers \[3\] --out_sizes \[256 512 1024\] --output_dir \[_"specify your own directory"_\] \[> _"specify your own log file path"_ 2>&1 &\]
   - "--gpu_id", specify the gpu id.
   - "--data_dir", specify the directory of the fine-tuning dataset.
   - "--embedding_dir", specify the directory to store the best pre-trained embedding model.
   - "--init_lr", specify the constant learning rate for the fine-tuning.
   - "--weight_decay", specify the constant weight decay value to regulate the fine-tuning network.
   - "--batch_size", specify the batch size for the DCSSMS fine-tuning.
   - "--num_layers", specify the number of layers for the Over-complete encoder network (here using this option to keep consistent with the pre-trained embedding model).
   - "--out_sizes", specify the sizes of the hidden layers for the Over-complete encoder network, e.g., 256 512 1024 for 3 layers, 256 512 1024 2048 for 4 layers Over-complete encoder network, etc. (here using this option to keep consistent with the pre-trained embedding model)
   - "--output_dir", specify the output directory for the best fine-tuned model.
   - "\[> _"specify your own log file path"_ 2>&1 &\]", specify whether to run the script in background, redirect stdout, stderr to log file, e.g., "> ./training512_10_3_true.log 2>&1 &".

# Detailed Information About The 442 Investigated MDH Methods & The 20 Re-balancing Algorithms
- 442 MDH Methods
  - 12 ad-hoc imputation methods, the implemented 12 ad-hoc imputation methods are MMI, MeMI, TMMI, CMMI, CMeMI, CTMMI, and their incremental versions, IMMI, IMeMI, ITMMI, ICMMI, ICMeMI, ICTMMI, in which, the previously imputed attributes values are used for the subsequent imputations of the missing attributes values. 
  - 320 KNN-related imputation methods, i.e., KNNI, sKNNI, WKNNI, sWKNNI, each with K nearest neighbours within range \[1, 20\], combined with 2 distance functions, HEOM (Heterogeneous Euclidean Overlap Metric) and HVDM (Heterogeneous Value Difference Metric) (i.e., 2 x 20 x 4 = 160). And 4 imbalance-specific KNN-related imputation methods, i.e., KNNII, sKNNII, WKNNII, sWKNNII, each with K nearest neighbours within range \[1, 20\], combined with 2 distance functions, HEOM (Heterogeneous Euclidean Overlap Metric) and HVDM (Heterogeneous Value Difference Metric) (i.e., 2 x 20 x 4 = 160).
  - 100 K-Means-related imputation methods, i.e., KMI with K nearest neighbours within range \[1, 20\], FKMI, with fuzzification hyper-parameter belongs to {1.25, 1.5, 1.75, 2.0}, combined with K nearest neighbours within range \[1, 20\] (i.e., 20 + (4 x 20) = 100).
  - 4 MICE-related imputation methods, MICE-Me_PMM, MICE-Me_CART, MICE-Me_SAMPLE, MICE-Me_RF. We adopt the mice package in R (https://amices.org/mice/), use mice method with arguments m=1, maxit=10, default values for all other arguments. To use mice R package in python, at first, install rpy2 python package. 
  - 2 MIDASpy-related imputation methods (i.e., MIDASpy with argument vae_layer=True and without it, default values for all other arguments. https://github.com/MIDASverse/MIDASpy/tree/master).
  - 2 missRanger (i.e., missRanger with argument splitrule = "extratrees" and without it, default values for all other arguments. https://github.com/mayer79/missRanger). To use missRanger R package in python, at first, install rpy2 python package.
  - 1 Furia-related (Fuzzy Unordered Rule Induction Algorithm) imputation method FIII. To use Furia-related methods, i.e., FIII and Furia-selector for linear evaluation experiments, at first, install openJDK 11, python-javabridge and python-weka-wrapper3. Copy "./FURIA/fuzzyUnorderedRuleInduction.jar" file into "PYTHON_INSTALLATION_PATH/Lib/site-packages/weka/lib/". https://weka.sourceforge.io/doc.packages/fuzzyUnorderedRuleInduction/weka/classifiers/rules/FURIA.html.
  - 1 ANN-related (Artificial Neural Network) imputation method ANNI.
- 20 Re-balancing Algorithms <br/>
  For the 20 class re-balancing algorithms, we adopt the implementations from the imbalanced-learn package, https://imbalanced-learn.org/stable/. We use the default values of the arguments of the 20 re-balancing methods as shown in the imbalanced-learn library.

# Requirements
   - python >= 3.10
   - pytorch >= 2.0
   - scikit-learn >= 1.3.1
   - imbalanced-learn >= 0.11
   - torchmetrics >= 1.0.3
   - pandas >= 2.1.1
   - numpy >= 1.26.0
   - scipy >= 1.11.3
   - matplotlib >= 3.7.1
   - tqdm >= 4.65.0
   - rpy2 >= 3.5.15
   - mice R package >= 3.16.0
   - MIDASpy >= 1.3.1
   - openJDK 11
   - python-javabridge >= 4.0.3
   - python-weka-wrapper3 >= 0.2.14