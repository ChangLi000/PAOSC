# GDAO-A-Plug-and-Play-Generative-Discriminative-Attention-Optimization-Framework-for-LLMs

## Project Name
PAOSC: Plug-and-play Attention Optimization for Semantic Consistency in LLMs

<img width="1974" height="979" alt="Figure 2" src="https://github.com/user-attachments/assets/0f904847-e623-4a32-ac4c-87c2f6163f62" />


Attention mechanisms are essential to the success of Large Language Models (LLMs). In practice, models often overemphasize semantically low-value tokens, forming attention sinks while failing to capture truly informative tokens. Existing inference-time optimization methods mainly rely on static adjustments or attention redistribution, which often disrupt the correspondence between attention distribution and the actual semantics of the input, leading to a loss of semantic consistency and degraded performance. To address this problem, we propose PAOSC, a plug-and-play attention optimization model designed to maintain semantic consistency by dynamically adjusting attention. PAOSC employs a generator to identify informative tokens and a discriminator to optimize the generator via policy gradients based on confidence changes and loss fluctuations. Experiments on eight LLMs show up to a 9.68\% improvement in the F1 score. On our constructed HTTP-RL dataset with 21,996 samples, PAOSC eliminates 18\% of low-value tokens, improving inference efficiency while maintaining semantic consistency.

## 📂Code Structure
```bash
src/
├── README.md
├── train_script.py          # training script

├── model.py                      # model strcuture
│   ├── generator                 # Generator G
│   ├── discriminator             # Discriminator D
│   ├── classificationhead        # Classification Head
│
├── data.py                      # Prepare training data
├── train.py/                    # PAOSC training framework

```

## 🔧 Quick Start

### 1️⃣ Download Dataset
You can get public dataset from:
  > AG_news: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset  
  > HttpParam: https://www.kaggle.com/datasets/evg3n1j/httpparamsdataset  
  > Spam: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset  
  
Due to the need for privacy protection, if you need the **HTTP_RL** dataset, please contact us **cli@cnic.cn**.

### 2️⃣ Train
  ```bash
  python train_script.py \
    --modelname "Llama2-7B" \
    --modelpath "/path/to/llama2-7b" \
    --datapath "/path/to/dataset.csv" \
    --lr 1e-4 \
    --topk 3 \
    --max_steps 100 \
    --epoch 5 \
    --lambda_mask 0.3 \
    --savepath "./saved_models"
```
