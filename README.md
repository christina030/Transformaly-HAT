# Final project - Transformaly-HAT

Transformaly: https://github.com/MatanCohen1/Transformaly

HAT: https://github.com/jiawangbai/HAT

## Setup
```
cd <path-to-Transformaly-directory>
git clone https://github.com/MatanCohen1/Transformaly.git
cd Transformaly
conda create -n transformalyenv python=3.7
conda activate transformalyenv
conda install --file requirements.txt
```

## Unimodal Training And Evaluation
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --data_path ./data --epochs 30 --batch_size 6 --eval_every 5 --unimodal
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset cifar10 --batch_size 2 --data_path ./data --whitening_threshold 0.9 --unimodal
```

## Multimodal Training And Evaluation
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --data_path ./data --epochs 30 --batch_size 6 --eval_every 5
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset cifar10 --batch_size 2 --data_path ./data --whitening_threshold 0.9 
```

## What have done
Based on the github of Transformaly,

Add `filter.py` and `filtered_dataset.py` for low- and high-pass filtering.
