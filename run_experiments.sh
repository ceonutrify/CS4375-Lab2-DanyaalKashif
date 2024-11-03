#!/bin/bash

echo "Starting RNN experiments..."

# Experiment 1: Default parameters
#echo "Running Experiment 1: Default parameters"
#python rnn.py --hidden_dim 50 --epochs 5 --train_data ./training.json --val_data ./validation.json \
#--note "Default parameters with hidden_dim=50 and epochs=5." --learning_rate 0.01 --batch_size 16 --weight_decay 0.0

echo "Running Experiment 2: Increased hidden dimension"
python rnn.py --hidden_dim 100 --epochs 5 --train_data ./training.json --val_data ./validation.json \
--note "Increased hidden_dim to 100 for higher capacity." --learning_rate 0.01 --batch_size 16 --weight_decay 0.0

echo "Running Experiment 3: Added extra RNN layer"
python rnn.py --hidden_dim 50 --epochs 5 --train_data ./training.json --val_data ./validation.json \
--note "Added an extra RNN layer to increase model depth." --learning_rate 0.01 --batch_size 16 --num_layers 2 --weight_decay 0.0

echo "Running Experiment 4: Reduced learning rate"
python rnn.py --hidden_dim 50 --epochs 5 --train_data ./training.json --val_data ./validation.json \
--note "Reduced learning rate to 0.001 for more stable learning." --learning_rate 0.001 --batch_size 16 --weight_decay 0.0

echo "Running Experiment 5: Increased learning rate with weight decay"
python rnn.py --hidden_dim 50 --epochs 5 --train_data ./training.json --val_data ./validation.json \
--note "Increased learning rate to 0.1 with weight decay of 0.01 for regularization." --learning_rate 0.1 --batch_size 16 --weight_decay 0.01

echo "Running Experiment 6: Increased batch size"
python rnn.py --hidden_dim 50 --epochs 5 --train_data ./training.json --val_data ./validation.json \
--note "Increased batch size to 32 to observe effects on training stability." --learning_rate 0.01 --batch_size 32 --weight_decay 0.0

echo "Running Experiment 7: Reduced batch size"
python rnn.py --hidden_dim 50 --epochs 5 --train_data ./training.json --val_data ./validation.json \
--note "Reduced batch size to 8 for potential generalization improvements." --learning_rate 0.01 --batch_size 8 --weight_decay 0.0

echo "Running Experiment 8: Extended epochs for baseline"
python rnn.py --hidden_dim 50 --epochs 10 --train_data ./training.json --val_data ./validation.json \
--note "Extended epochs to 10 for baseline to observe long-term behavior." --learning_rate 0.01 --batch_size 16 --weight_decay 0.0

echo "Running Experiment 9: Increased hidden_dim and epochs"
python rnn.py --hidden_dim 100 --epochs 10 --train_data ./training.json --val_data ./validation.json \
--note "Increased hidden_dim to 100 and extended epochs with learning rate decay." --learning_rate 0.01 --batch_size 16 --weight_decay 0.0 --num_layers 1

echo "All experiments completed!"