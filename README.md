# Affective EEG-based Person Identification with Continual Learning
This repository is the official implementation of the paper "Affective EEG-based Person Identification with Continual Learning"[[PDF]](https://ieeexplore.ieee.org/document/10540616). Our repository is primarily built upon PyCIL [[Github]](https://github.com/G-U-N/PyCIL), and we are grateful for there contributions!

![Framework](/images/framework.png)



## Datasets

The THU-EP dataset is not publicly shareable on the internet. You can contact the original authors to obtain the dataset. Additionally, the original authors' team has recently released a publicly accessible affective EEG dataset with over a hundred subjects, called FACED [[URL]](https://www.synapse.org/#!Synapse:syn50614194/wiki/620378). However, please note that the dataset processing code provided in this repository may not be directly applicable to FACED.




## How To Use

### Clone

Clone this GitHub repository:

```
git clone https://github.com/JerryKingQAQ/AEEG-PI-CL.git
cd AEEG-PI-CL
```

### Dependencies

Install the required dependencies:

```
pip install -r requirements.txt
```

### Dataset process

To preprocess the data, create a folder named `data` and run the preprocessing code.

```
mkdir data
cd dataset_process
python main.py --files_path 'path/to/THU-EP/'
```

### Run experiments

1. Edit the `[Experiment Name].json` file for global settings.
2. Edit the hyperparameters in the corresponding `[Model Name].py` file (e.g., `models/icarl.py`).
3. Run:

```
python main.py --config=./exps/[Experiment Name].json
```

For detailed explanations about the hyperparameters, you can refer to the descriptions in the PyCIL repository.



## Acknowledgments

We are very grateful to Haoyu Wang [[6jybuchiyu]](https://github.com/6jybuchiyu) for reviewing and correcting this repository! We also extend our gratitude to the PyCIL repository!



## Citation

If you find the paper or this repo useful, please cite:

```
@ARTICLE{10540616,
  author={Jin, Jiarui and Chen, Zongnan and Cai, Honghua and Pan, Jiahui},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Affective EEG-Based Person Identification With Continual Learning}, 
  year={2024},
  volume={73},
  number={},
  pages={1-16},
  keywords={Identification of persons;Electroencephalography;Continuing education;Brain modeling;Task analysis;Feature extraction;Transformers;Continual learning;electroencephalogram (EEG);multidomain coordinated attention mechanism;person identification;transformer},
  doi={10.1109/TIM.2024.3406836}}

```

