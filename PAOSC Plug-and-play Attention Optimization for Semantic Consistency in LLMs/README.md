# GDAO-A-Plug-and-Play-Generative-Discriminative-Attention-Optimization-Framework-for-LLMs

## Project Name
PAOSC: Plug-and-play Attention Optimization for Semantic Consistency in LLMs

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


