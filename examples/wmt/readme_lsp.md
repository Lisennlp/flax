
### 安装环境
    - 使用我们的脚本创建tpu并安装环境
    
    - 补充库
        pip uninstall flax
        git clone git@github.com:Lisennlp/flax.git
        cd flax
        pip install -e .
        cd examples/wmt/
        pip install -r requirements.txt
### 下载训练集

    python -m tensorflow_datasets.scripts.download_and_prepare --datasets=wmt17_translate/de-en

### 下载测试集

    python -m tensorflow_datasets.scripts.download_and_prepare --datasets=wmt14_translate/de-en

### 训练
    # dcformer_compare_experiments包含wmt17_translate和wmt14_translate两个文件夹
    export TFDS_DATA_DIR=gs://jax_llm_data/dcformer_compare_experiments/

    # per_device_batch_size表示每台机器的batch_size
    FLAGS="--config.num_train_steps=100000 --config.warmup_steps=1000 --config.checkpoint_every_steps=1000 --config.per_device_batch_size=32"
    WOKRDIR=gs://jax_llm_data/dcformer_compare_experiments/logs/wmt_256/

    # default.py为配置文件
    # v3-8
    TPU_NAME=llm-jax-v3-8-10
    ZONE=us-east1-d
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/python main.py --workdir=$WOKRDIR --config=configs/default.py $FLAGS"

    # v3-32
    TPU_NAME=llm-jax-v3-32-10
    ZONE=us-east1-d
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="/home/lishengping/miniconda3/bin/python main.py --workdir=$WOKRDIR --config=configs/default.py $FLAGS"

