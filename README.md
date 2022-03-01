# Preference-Aware LID 

Implementation for the ACL2022 findings paper "Unsupervised Preference-Aware Language Identification"

### Requirements
- Python = 3.6
- TensorFlow = 1.12.0
- pyyaml
- nltk

### Benchmark
- U-LID:  in corpus21_ulid/test_ulid
The "U-LID" is collected from a real-world translation system AliTrans. This benchmark consists of 21 languages and 11,031 samples. The average word count in each sample is 2.08, and the average number with respect to character is 13.27.

- KB-21:  in corpus21_ulid/test_kb21
The "KB-21" is a publicly available test set from Kocmi and Bojar (2017)[1], using a subset of 21 languages. "KB-21" consists of 2,100 samples, the average amounts of words and characters in each sample are 4.47 and 34.90, respectively.

- Instruction
The "test.ulid" represents the user's language preference, and the format is "language abbreviation: probability value; language abbreviation: probability value". 
The "test.trg" is the language label with abbreviations. 
The "test.src" is the user's input text.

- Sample
test.ulid is ar:0.00,de:0.00,en:0.12,es:0.80,fr:0.00,he:0.00,hi:0.00,id:0.01,it:0.04,ja:0.00,ko:0.00,ms:0.00,nl:0.00,pl:0.00,pt:0.03,ru:0.00,th:0.00,tr:0.00,vi:0.00,zh:0.00
test.trg is pt
test.src is monitor cardiaco esportivo
- Meaning
The language label of text "monitor cardiaco esportivo" is "Portuguese", and the user language preference is "80% Spanish, 12% English, 4% Italian, 3% Portuguese, 1% Indonesian".


- Language label and abbreviations:
English (en), Chinese (zh), Russian (ru), Portuguese (pt), Spanish (es), French (fr), German (de), Italian (it), Dutch (nl), Japanese (ja), Korean (ko), Arabic (ar), Thai (th), Hindi (hi), Hebrew (he),  Vietnamese (vi), Turkish (tr),  Polish (pl),  Indonesian (id), Malay (ms), and Ukrainian (uk).


### Dataset

Sample 100,000 training dataset in corpus21_ulid.

### Instruction
"transformer_model_fn" is the main function of the model, which calls the training function "get_loss". Compared with the normal transformer based text classification model, our innovation is mainly focused on the "output_layer" function.
In the "output_layer" function, we define the variable "use_user_lang_map". When the variable value is "1", the Revision-Based Model is implemented, and when the variable value is "2", the Representation-Based Model is implemented.

### Train & Evaluation
#### Transformer-Based Model ( Baseline )
- For Train
```
python train_transformer.py --exp_name ulid_transformer_vbase --corpus_dir corpus21_ulid --vocab_size 15000 --class_num 21 --use_user_lang_map 0
```
- For Evaluation on U-LID Testset 
```
python eval.py --exp_name ulid_transformer_vbase --corpus_dir corpus21_ulid/test_ulid --vocab_dir corpus21_ulid --vocab_size 15000 --class_num 21 --use_user_lang_map 0 --postfix ".test_ulid
```
- For Evaluation on KB-21 Testset
```
python eval.py --exp_name ulid_transformer_vbase --corpus_dir corpus21_ulid/test_kb21 --vocab_dir corpus21_ulid --vocab_size 15000 --class_num 21 --use_user_lang_map 0 --postfix ".test_kb21
```
#### For Revision-Based Model ( Preference-Aware Model )
- For Train
```
python train_transformer.py --exp_name ulid_transformer_v1 --corpus_dir corpus21_ulid --vocab_size 15000 --class_num 21 --use_user_lang_map 1
```
- For Evaluation on U-LID Testset 
```
python eval.py --exp_name ulid_transformer_v1 --corpus_dir corpus21_ulid/test_ulid --vocab_dir corpus21_ulid --vocab_size 15000 --class_num 21 --use_user_lang_map 1 --postfix ".test_ulid
```
- For Evaluation on KB-21 Testset
```
python eval.py --exp_name ulid_transformer_v1 --corpus_dir corpus21_ulid/test_kb21 --vocab_dir corpus21_ulid --vocab_size 15000 --class_num 21 --use_user_lang_map 1 --postfix ".test_kb21
```
#### For Representation-Based Model ( Preference-Aware Model )
- For Train
```
python train_transformer.py --exp_name ulid_transformer_v2 --corpus_dir corpus21_ulid --vocab_size 15000 --class_num 21 --use_user_lang_map 2
```
- For Evaluation on U-LID Testset 
```
python eval.py --exp_name ulid_transformer_v2 --corpus_dir corpus21_ulid/test_ulid --vocab_dir corpus21_ulid --vocab_size 15000 --class_num 21 --use_user_lang_map 2 --postfix ".test_ulid
```
- For Evaluation on KB-21 Testset
```
python eval.py --exp_name ulid_transformer_v2 --corpus_dir corpus21_ulid/test_kb21 --vocab_dir corpus21_ulid --vocab_size 15000 --class_num 21 --use_user_lang_map 2 --postfix ".test_kb21
```

### References
\[1\] Tom Kocmi and Ondrej Bojar. 2017. Lanidenn: Multilingual language identification on character window. CoRR, abs/1701.03338.
