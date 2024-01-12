# Deep Contrastive Self-Supervised Method Selection Framework (DCSSMS) For Missing Data Handling
The official repository for a deep contrastive self-supervised Missing Data Handling method selection framework.\n
The structure of the directories:\n
- **"./DATA/"**, all the dataset files for self-supervised training and linear evaluation training.\n
	- **"./DATA/Linear_Evaluation/\*.\*"**, dataset files for linear evaluation fine-tuning.
	- **"./DATA/Self-supervised_Training/\*.\*"**, dataset files for DCSSMS embedding training.
- **"./Framework/"**, all the code files for the DCSSMS framework training and fine-tuning.

# DCSSMS embedding training
To train the DCSSMS embedding network, use the following python script:\n
python ./Framework/MainBYOL.py --gpu_id 0 --data_dir "./DATA/Self-supervised_Training/" --init_lr \[0.030280\] --max_lr \[1.287572\] --suffix \[10\] --batch_size \[512\] --num_layers \[3\] --out_sizes \[256 512 1024\] --output_dir ./ --use_momentum \[True/False\] \[> ./training512_10_3_true.log 2>&1 &\]\n
- "--gpu_id", specify the gpu id; 
- "--data_dir", specify the directory of the embedding training dataset;
- "--init_lr", specify the initial learning rate for the OneCycle learning rate scheduler;
- "--max-lr", specify the maximum learning rate for the OneCycle learning rate scheduler;
- "--suffix", fix to 10;
- "--batch_size", specify the batch size for the DCSSMS embedding training;
- "--num_layers", specify the number of layers for the Over-complete encoder network;
- "--out_sizes", specify the sizes of the hidden layers for the Over-complete encoder network, e.g., 256 512 1024 for 3 layers, 256 512 1024 2048 for 4 layers Over-complete encoder network, etc.
- "--output_dir", specify the output directory for the best learned embedding models;
- "--use_momentum", specify whether to use the Stop-gradient mechanism for the "Target" network;
- "\[> ./training512_10_3_true.log 2>&1 &\]", specify whether to run the script in background.





