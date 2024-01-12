# Deep Contrastive Self-Supervised Method Selection Framework (DCSSMS) For Missing Data Handling
The official repository for a deep contrastive self-supervised Missing Data Handling method selection framework. 

The structure of the directories:

	+ ** ./DATA/ **, all the dataset files for self-supervised training and linear evaluation training.
	
		1. ** ./DATA/Linear_Evaluation/\*.\* **, dataset files for linear evaluation fine-tuning.
		
		2. ** ./DATA/Self-supervised_Training/\*.\* **, dataset files for DCSSMS embedding training.
		
	+ **./Framework/ **, all the code files for the DCSSMS framework training and fine-tuning.

# DCSSMS embedding training
To train the DCSSMS embedding network, use the following python script:

python ./Framework/MainBYOL.py --gpu_id 0 --data_dir "./DATA/Self-supervised_Training/" --init_lr 0.030280 --max_lr 1.287572 --suffix 10 --batch_size 512 --num_layers 3 --out_sizes 256 512 1024 --output_dir ./ --use_momentum True > ./training512_10_3_true.log \[2>&1 &\]

"--gpu_id", specify the gpu id; "--data_dir", specify the directory of the embedding training dataset; "--init_lr", specify the initial learning rate for the One-cycle learning rate scheduler; "--max-lr", specify the maximum learning rate for the One-cycle learning rate scheduler; 



