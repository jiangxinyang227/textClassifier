#!/bin/sh

export BERT_BASE_DIR=../modelParams/uncased_L-12_H-768_A-12

export DATASET=../data/

python run_classifier.py \
  --data_dir=$DATASET \
  --task_name=imdb \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --output_dir=../output/ \
  --do_train=true \
  --do_eval=true \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=200 \
  --train_batch_size=16 \
  --learning_rate=5e-5\
  --num_train_epochs=5.0
