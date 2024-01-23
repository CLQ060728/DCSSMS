# Deep Contrastive Self-Supervised Method Selection Framework (DCSSMS) Of Missing Data Handling For Imbalanced Classification Analyses
The official repository for the Deep Contrastive Self-Supervised missing data handling Method Selection (DCSSMS) framework.
The structure of the directories:
````bash
.
├── DATA				----> all the dataset files for self-supervised training and linear evaluation training.
│   ├── Linear_Evaluation 		----> dataset files for linear evaluation fine-tuning.
│   └── Self-supervised_Training	----> dataset files for DCSSMS embedding training.
│       └── README.md
├── Embedding				----> the best pre-trained DCSSMS embedding network.
│   └── README.md
├── ExpResults				----> more experimental results
│   ├── Linear_Evaluation_Ablation	----> details about all the linear evaluation experiments, including ablation studies.
│   └── MDHMS				----> details about all the MDH method selection experiments.
├── FURIA				----> the jar file for Furia-related methods, i.e., FIII and Furia-selector.
├── Framework					---->  all the source code files for the DCSSMS framework training and fine-tuning.
├── LICENSE
├── README.md
└── Supplementary_Material	----> appendixes file including details about the 56 real-world imbalanced classification datasets.
    └── Appendix.pdf
````
- ! Missing **"./ExpResults/"**, more experimental results
	- **"./ExpResults/MDHMS/MSD^raw", we make our self-supervised training dataset, i.e., MSD^raw, publicly accessible. This dataset can also provide insights to select the $442$ investigated MDH methods under different imbalanced classification circumstances.**

# DCSSMS Embedding Training
To train the DCSSMS embedding network, follow the steps as shown below:
1. clone the DCSSMS GitHub repository.
1. in CLI, change current path to **"DCSSMS/"**.
1. use the following python script to start training:<br/>
```bash
 python ./Framework/MainBYOL.py  --gpu_id \[0\] \\
				 --data_dir "./DATA/Self-supervised_Training/" \\
				 --init_lr \[0.030280\]	\\
				 --max_lr \[1.287572\]	\\
				 --batch_size \[512\]	\\
				 --num_layers \[3\]	\\
				 --out_sizes \[256 512 1024\] \\
				 --output_dir \[_"specify your own directory"_\] \\
				 --use_momentum \[True/False\] \[> _"specify your own log file path"_ 2>&1 &\]
```
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
  - 320 implemented KNN-related imputation methods, i.e., KNNI, sKNNI, WKNNI, sWKNNI, each with K nearest neighbours within range \[1, 20\], combined with 2 distance functions, HEOM (Heterogeneous Euclidean Overlap Metric) and HVDM (Heterogeneous Value Difference Metric) (i.e., 2 x 20 x 4 = 160). And 4 imbalance-specific KNN-related imputation methods, i.e., KNNII, sKNNII, WKNNII, sWKNNII, each with K nearest neighbours within range \[1, 20\], combined with 2 distance functions, HEOM (Heterogeneous Euclidean Overlap Metric) and HVDM (Heterogeneous Value Difference Metric) (i.e., 2 x 20 x 4 = 160).
  - 100 implemented K-Means-related imputation methods, i.e., KMI with K nearest neighbours within range \[1, 20\] and FKMI with fuzzification hyper-parameter belongs to {1.25, 1.5, 1.75, 2.0}, combined with K nearest neighbours within range \[1, 20\] (i.e., 1 x 20 \[KMI\] + (4 x 20) \[FKMI\] = 100).
  - 4 MICE-related imputation methods, MICE-Me_PMM, MICE-Me_CART, MICE-Me_SAMPLE, MICE-Me_RF. We adopt the mice package in R (https://amices.org/mice/), use mice method with arguments (method=\["pmm", "cart", "sample", "rf"\], m=1, maxit=10), default values for all other arguments. To use mice R package in python, at first, install rpy2 python package. 
  - 2 MIDASpy-related imputation methods (i.e., MIDASpy with argument (vae_layer=True) and without it, default values for all other arguments. https://github.com/MIDASverse/MIDASpy/tree/master).
  - 2 missRanger (i.e., missRanger with argument (splitrule="extratrees") and without it, default values for all other arguments. https://github.com/mayer79/missRanger). To use missRanger R package in python, at first, install rpy2 python package.
  - 1 Furia-related (Fuzzy Unordered Rule Induction Algorithm) imputation method FIII. To use Furia-related methods, i.e., FIII and Furia-selector for linear evaluation experiments, at first, install openJDK 11, python-javabridge and python-weka-wrapper3. Copy "./FURIA/fuzzyUnorderedRuleInduction.jar" file into "PYTHON_INSTALLATION_PATH/Lib/site-packages/weka/lib/". https://weka.sourceforge.io/doc.packages/fuzzyUnorderedRuleInduction/weka/classifiers/rules/FURIA.html.
  - 1 ANN-related (Artificial Neural Network) imputation method ANNI.
- 20 Re-balancing Algorithms <br/>
  For the 20 class re-balancing algorithms, we adopt the implementations from the imbalanced-learn package, https://imbalanced-learn.org/stable/. We use the default values of the arguments of the 20 re-balancing methods as shown in the imbalanced-learn library.

# MDH Method Selection Website For Each Of The 56 Real-world Imbalanced Classification Datasets
  Coming Soon ...

# Requirements
```$ pip install -r requirements.txt```
