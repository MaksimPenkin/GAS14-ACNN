# GAS14-Attention-CNN

## How to train
1. Select desired model from **checkpoints** folder. 
2. Each model contains its generator. Copy it into **models/model.py** from **checkpoints/YOUR_MODEL/model.py**.
3. python run_model.py 
        --phase train
        --checkpoints checkpoints/YOUR_MODEL
        --datalist PATH_TO_TRAIN_DATALIST
        --val_datalist PATH_TO_VALIDATIONs_DATALIST
        --restore_ckpt INITIAL CHECKPOINT NUMBER FOR TRANSFER LEARNING
        