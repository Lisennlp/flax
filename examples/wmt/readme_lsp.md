
### 安装环境
    git clone git@github.com:Lisennlp/flax.git
    cd flax
    pip install -e .
    pip install -r requirements.txt
### 下载训练集

    python -m tensorflow_datasets.scripts.download_and_prepare --datasets=wmt17_translate/de-en

### 下载测试集

    python -m tensorflow_datasets.scripts.download_and_prepare --datasets=wmt14_translate/de-en

### 训练
    # dcformer_compare_experiments包含wmt17_translate和wmt14_translate两个文件夹
    export TFDS_DATA_DIR=gs://jax_llm_data/dcformer_compare_experiments/
    FLAGS="--config.num_train_steps=100000 --config.warmup_steps=1000 --config.checkpoint_every_steps=1000"
    python3 main.py --workdir=logs/wmt_256/ --config=configs/default.py $FLAGS


