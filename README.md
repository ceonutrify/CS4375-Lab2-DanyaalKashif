# CS4375 Lab 2: Neural Networks for Sentiment Analysis
By Danyaal Kashif (DHK200000)

## Project Overview
This project implements and evaluates Feedforward Neural Networks (FFNN) and Recurrent Neural Networks (RNN) for performing 5-class sentiment analysis on Yelp reviews. The models predict star ratings (1-5) based on review text.

## Dataset
- Training Set: 8,000 examples
- Validation Set: 2,000 examples
- Test Set: 2,000 examples
- Format: Text reviews with associated star ratings
- Word embeddings provided in `word_embedding.pkl`

## Project Structure
```
.
├── ffnn.py                    # Feedforward Neural Network implementation
├── rnn.py                     # Recurrent Neural Network implementation
├── run_experiments.sh         # Script to run all experiments
├── ffnnTestsOutput.json      # FFNN experimental results
├── rnnTestsOutput.json       # RNN experimental results
├── Data_Embedding.zip        # Dataset and word embeddings
└── README.md                 # This file
```

## Requirements
- Python 3.8.x
- PyTorch 1.10.1
- Additional dependencies listed in `requirements.txt`

### Setup
```bash
# Create virtual environment
conda create -n cs4375 python=3.8
conda activate cs4375

# Install dependencies
pip install -r requirements.txt
```

## Model Implementations

### Feedforward Neural Network (FFNN)
- Uses bag-of-words representation
- Multiple hidden layers with ReLU activation
- Dropout regularization
- Supports both SGD and Adam optimizers
- Learning rate scheduling

To run FFNN:
```bash
python ffnn.py --hidden_dim [dim] --epochs [num] --train_data [path] --val_data [path]
```

### Recurrent Neural Network (RNN)
- Processes word embeddings sequentially
- Handles variable-length sequences
- Uses packed sequences for efficiency
- Includes dropout and regularization
- Supports multiple RNN layers

To run RNN:
```bash
python rnn.py --hidden_dim [dim] --epochs [num] --train_data [path] --val_data [path]
```

## Experimental Results

### FFNN Performance
| Experiment | Hidden Dim | LR | Batch Size | Optimizer | Train Acc | Val Acc |
|------------|------------|-------|------------|-----------|------------|----------|
| Best Model | 50 | 0.001 | 16 | Adam | 0.66 | 0.62 |

### RNN Performance
| Experiment | Hidden Dim | LR | Batch Size | Num Layers | Train Acc | Val Acc |
|------------|------------|-------|------------|------------|------------|----------|
| Best Model | 100 | 0.01 | 16 | 1 | 0.56 | 0.57 |

## Key Findings
1. Adam optimizer generally outperformed SGD for FFNN
2. Increasing hidden dimensions improved model capacity but required careful regularization
3. RNN showed better handling of sequential patterns but was more computationally intensive
4. Dropout and learning rate scheduling were crucial for preventing overfitting

## Running Experiments
To reproduce all experiments:
```bash
# Make script executable
chmod +x run_experiments.sh

# Run experiments
./run_experiments.sh
```

## Results and Analysis
- Detailed results are stored in `ffnnTestsOutput.json` and `rnnTestsOutput.json`
- Learning curves and analysis are available in the project report
- Error analysis shows models struggle with:
  - Mixed sentiment reviews
  - Sarcasm detection
  - Complex language patterns

## Future Improvements
1. Implement attention mechanisms
2. Explore LSTM/GRU architectures
3. Utilize pre-trained language models
4. Enhance handling of mixed sentiments
5. Implement cross-validation

## Citation
If you use this code in your research, please cite:
```bibtex
@misc{kashif2024neural,
  author = {Kashif, Danyaal},
  title = {Neural Networks for Sentiment Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ceonutrify/CS4375-Lab2-DanyaalKashif}
}
```

## License
This project is licensed under Danyaal Kashif's personal license or something like that, anyone can use this, it's literally a UTD CS4375 project that was done by me, also proof that our prof's are lowkey finessing MIT curriculum. 