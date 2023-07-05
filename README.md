# Leveraging Large Language Models with Multiple Loss Learners for Few-Shot Author Profiling

This repository contains the code and data for the paper "Leveraging Large Language Models with Multiple Loss Learners for Few-Shot Author Profiling" by Hamed Babaei Giglou, Mostafa Rahgouy, Jennifer D’Souza, Milad Molazadeh Oskuee, Hadi Bayrami Asl Tekanlou, and Cheryl D Seals. The paper was presented at the Fourteenth International Conference of the CLEF Association (CLEF 2023).

## Abstract

The objective of author profiling (AP) is to study the characteristics of authors through the analysis of how language is exchanged among people. Studying these attributes sometimes is challenging due to the lack of annotated data. This indicates the significance of studying AP from a low-resource perspective. This year at AP@PAN 2023 the major interest raised in profiling cryptocurrency influencers with a few-shot learning technique to analyze the effectiveness of advanced approaches in dealing with new tasks from a low-resource perspective.

![File](images/main-diagram.png)

## Contents

- `data/`: Contains the datasets used in the paper.
- `models/`: Contains the code for the Bi-Encoder and Large Language Model used in the paper.
- `experiments/`: Contains the code for the experiments conducted in the paper.
- `results/`: Contains the results of the experiments.

## Requirements

- Python 3.8 or higher
- PyTorch 1.9.x or higher
- Transformers 4.3.x or higher

## Usage

1. Clone the repository:

```
https://github.com/HamedBabaei/author-profiling-pan2023
cd author-profiling-pan2023
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Run the experiments:
   1. Inference fsl
    ```bash
    bash inference_fsl.sh 
    ```
   2. Inference fsl
   ```bash
   bash inference_fsl_biencoder.sh 
   ``` 
   3. Baseline (random)
    ```bash
    bash random_baseline.sh 
    ```
   4. Baseline (Zero Shot)
   ```bash
   bash zero_shot_baseline.sh
   ``` 
   5. Train & Test SBERT
    ```bash
    bash train_test_runner_sbert.sh
    ```
   6. Train & Test flanT5
   ```bash
   bash train_test_runner_flan_t5.sh 
   ```  


## Citation

If you use this code or data in your research, please cite the following paper:



```bib
@InProceedings{giglou:2023,
  author =                   {Hamed Babaei Giglou, Mostafa Rahgouy, Jennifer D’Souza, Milad Molazadeh Oskuee , Hadi Bayrami Asl Tekanlou and Cheryl D Seals},
  booktitle =                {{CLEF 2023 Labs and Workshops, Notebook Papers}},
  crossref =                 {pan:2023},
  editor =                   {Hamed Babaei Giglou, Mostafa Rahgouy, Jennifer D’Souza, Milad Molazadeh Oskuee , Hadi Bayrami Asl Tekanlou and Cheryl D Seals},
  month =                    sep,
  publisher =                {CEUR-WS.org},
  title =                    {{Leveraging Large Language Models with Multiple Loss Learners for Few-Shot Author Profiling}},
  year =                     2023
}
```