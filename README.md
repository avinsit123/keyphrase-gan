# Keyphrase-GAN
This repository contains the code to run generative adversarial networks for keyphrase extraction and generation.

Our Implementation is built on the starter code from <a href = "https://github.com/kenchan0226/keyphrase-generation-rl"> keyphrase-generation-rl </a> and <a href = "https://github.com/memray/seq2seq-keyphrase-pytorch"> seq2seq-keyphrase-pytorch </a> .

## Dependencies 

## Adverserial Training
First start by creating a virtual environment and install all required dependencies.
```terminal
pip install virtualenv
virtualenv mypython
pip install -r requirements.txt
source mypython/bin/activate
```

### Data 
The GAN model is trained on close to 500000 examples of the kp20k dataset and evaluated on the Inspec (Huth) , Krapivin , NUS , Semeval Datasets . After Downloading this repo , create a `Data` folder within it . Download all the required datasets from [this](https://drive.google.com/open?id=1DbXV1mZXm_o9bgfwPV9PV0ZPcNo1cnLp) and store it in the `Data` folder . The Folders with `_sorted` suffix contain present keyphrases which are sorted in the order of there occurence , and the ones with `_seperated` suffix contains present and absent keyphrases seperated by a `<peos>` token . In order to preprocess the kp20k dataset , run 
```terminal
python3 preprocess.py -data_dir data/kp20k_sorted -remove_eos -include_peos
```

If you cant preprocess and want to temporarily run the repository , to can download the datasets with 10000 examples [here](https://drive.google.com/drive/folders/1YIJOAAR8rK8oiAfPK-5aJwgwlmw0uie_?usp=sharing) .

### Training the MLE model 
The first step in GAN training involves training the MLE model as a baseline using maximum likelihood loss . The paper has used CatSeq model as a baseline . In order to train Catseq model without copy attention run
```terminal
python3 train.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -exp kp20k -epochs 25 -train_ml -one2many -one2many_mode 1 -batch_size 24
```
or with copy attention run
```terminal
python3 train.py -data data/kp20k_sorted/ -vocab data/kp20k_sorted/ -exp_path exp/%s.%s -exp kp20k -epochs 25 -train_ml -one2many -one2many_mode 1 -batch_size 24 -copy_attention
```

Note Down the Checkpoints Location while training .

### Training the Discriminator 

Now that the baseline MLE model is trained we need to train the Discriminator using the MLE model as Generator. The Discriminator is a hierarchal blstm which uses attention mechanism to calculate its 
