#!/usr/bin/env bash
root=/mnt/cephfs_wj/common/lab/baoyu.nlp
cd ${root}

fairseq-train fairseq_data/wmt17_en_de/ --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000   --criterion label_smoothed_cross_entropy --label-smoothing 0.1   --lr-scheduler fixed --force-anneal 50  --arch transformer_vaswani_wmt_en_de_big --save-dir fairseq_experiments/checkpoints/transformer_vaswani_wmt_en_de_big --keep-last-epochs 10


fairseq-train /mnt/cephfs_wj/common/lab/baoyu.nlp/fairseq_data/iwslt16_en_de/ --user-dir /mnt/cephfs_wj/common/lab/baoyu.nlp/projects/fairseq/nonauto -a transformer_iwslt16_de_en --optimizer adam -s en -t de --dropout 0.079 --max-tokens 2048 --min-lr '1e-05' --lr-scheduler anneal_decay --criterion cross_entropy --max-update 500000 --warmup-updates 746 --adam-betas '(0.9, 0.98)' --save-dir /mnt/cephfs_wj/common/lab/baoyu.nlp/fairseq_experiments/checkpoints/iwslt16/en-de/small --keep-last-epochs 10 --tensorboard-logdir ${ARNOLD_OUTPUT}/small

# generate
fairseq-generate --path /mnt/cephfs_wj/common/lab/baoyu.nlp/fairseq_experiments/checkpoints/iwslt16/en-de/small/checkpoint_best.pt /mnt/cephfs_wj/common/lab/baoyu.nlp/fairseq_data/iwslt16_en_de --beam 1 --source-lang en --target-lang de --user-dir /mnt/cephfs_wj/common/lab/baoyu.nlp/projects/fairseq/nonauto

fairseq-generate --path /mnt/cephfs_wj/common/lab/baoyu.nlp/fairseq_experiments/checkpoints/iwslt16/basic_nat_small/checkpoint_best.pt /mnt/cephfs_wj/common/lab/baoyu.nlp/fairseq_data/iwslt16_en_de --beam 1 --source-lang en --target-lang de --task translation_fast --user-dir /mnt/cephfs_wj/common/lab/baoyu.nlp/projects/fairseq/nonauto

# explore the nat
fairseq-train /mnt/cephfs_wj/common/lab/baoyu.nlp/fairseq_data/iwslt16_en_de/ --user-dir /mnt/cephfs_wj/common/lab/baoyu.nlp/projects/fairseq/nonauto --arch small --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --criterion label_smoothed_cross_entropy --label-smoothing 0.1   --lr-scheduler fixed --force-anneal 50 --save-dir /mnt/cephfs_wj/common/lab/baoyu.nlp/fairseq_experiments/checkpoints/iwslt16/basic_nat_small --keep-last-epochs 10


# prepare the data set