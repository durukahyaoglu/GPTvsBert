#!/bin/bash
data_dir="pretrained_spanbert"
model="tacred"
echo Downloading pre-trained SpanBERT 
wget -P $data_dir http://dl.fbaipublicfiles.com/fairseq/models/spanbert_$model.tar.gz
tar xvzf $data_dir/spanbert_$model.tar.gz -C $data_dir
rm $data_dir/spanbert_$model.tar.gz
