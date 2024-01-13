# Deep Contrastive Self-Supervised Method Selection Framework (DCSSMS) For Missing Data Handling
The official repository for a deep contrastive self-supervised Missing Data Handling method selection framework.
The structure of the directories:
- **"./DATA/"**, all the dataset files for self-supervised training and linear evaluation training.
	- **"./DATA/Linear_Evaluation/\*.\*"**, dataset files for linear evaluation fine-tuning.
	- **"./DATA/Self-supervised_Training/\*.\*"**, dataset files for DCSSMS embedding training.
- **"./Framework/"**, all the code files for the DCSSMS framework training and fine-tuning.
- **"./Embedding/"**, the best pre-trained DCSSMS embedding file.
- **"./ExpResults/"**, more experimental results
    - **"./ExpResults/Linear_Evaluation_Ablation/\*.\*"**, details about all the linear evaluation experiments, including ablation studies.
	- **"./ExpResults/MDHMS/\*.\*"**, details about all the MDH method selection experiments.
	- **"./ExpResults/MDHMS/MSD^raw", we make our self-supervised training dataset, i.e., MSD^raw, publicly accessible. This dataset can also provide insights to select the $442$ investigated MDH methods under different MDH circumstances.**

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
1. to test fine-tuned linear evaluation model, use the following python script:<br/>
   python ./Framework/LinearClassifier.py --gpu_id \[0\] --data_dir "./DATA/Linear_Evaluation/" --linear_model_dir ./Embedding/best_model_8192_10_3_True.pth --batch_size \[128\] --num_layers \[3\] --out_sizes \[256 512 1024\] \[> _"specify your own log file path"_ 2>&1 &\]
   - "--gpu_id", specify the gpu id.
   - "--data_dir", specify the directory of the test dataset.
   - "--linear_model_dir", specify the directory to store the best fine-tuned models.
   - "--batch_size", specify the batch size to test the fine-tuned model.
   - "--num_layers", specify the number of layers for the Over-complete encoder network (here using this option to keep consistent with the pre-trained embedding model).
   - "--out_sizes", specify the sizes of the hidden layers for the Over-complete encoder network, e.g., 256 512 1024 for 3 layers, 256 512 1024 2048 for 4 layers Over-complete encoder network, etc. (here using this option to keep consistent with the pre-trained embedding model)
   - "\[> _"specify your own log file path"_ 2>&1 &\]", specify whether to run the script in background, redirect stdout, stderr to log file, e.g., "> ./training512_10_3_true.log 2>&1 &".
