# Experiments on Variations of Neural Machine Translation System
### A Detailed Analysis for Vietnamese to English and Chinese to English Translation

## Tasks:
- Recurrent neural network based encoder-decoder without attention
- Recurrent neural network based encoder-decoder with attention
- Replace the recurrent encoder with either convolutional or self-attention based encoder
- Build either or both fully self-attention translation system

## To Run:
### RNN based encoder-decoder without attention
	python main.py --language zh --save_model_name zh_rnn --FT_emb_path ft_emb \\
		       --data_path MT_data --goal zh_rnn --decoder_type basic 
### RNN based encoder-decoder with attention
	python main.py --language zh --save_model_name zh_attn --FT_emb_path ft_emb \\
		       --data_path MT_data --goal zh_attn --decoder_type attn
### Self-attention based encoder and RNN based decoder with attention
	python main.py --language zh --save_model_name zh_attn --FT_emb_path ft_emb \\
		       --data_path MT_data -goal zh_selfattn --self_attn True
### Transformer
	python main.py --language zh --save_model_name zh_transformer --FT_emb_path ft_emb \\
		       --data_path MT_data -goal zh_transformer --transformer True
### Test
	python main.py --language zh --save_model_name zh_attn --FT_emb_path ft_emb \\
		       --data_path MT_data -goal zh_transformer --test_only True

## Experiments Results:
#### Sacre-BLEU scores of three models on test set
| Model				| Vietnamese(vi)	| Chinese(zh)	|
| :--- 				|:---:			|	---: 	|
| Baseline			| 8.19           	| 7.23          |
| RNN + Attention 		| 17.69 		| 12.83 	|
| Self-attention + Attention 	| 13.27 		| 7.97 		|

#### LSTM vs GRU
| Model				| Vietnamese(vi)	| Chinese(zh)	|
| :--- 				|:---:			|	---: 	|
| GRU				| 16.37          	| 11.86         |
| LSTM				| 17.06			| 11.94 	|

## Sample Tranaslation Results:
#### Chinese to English
> Source Sentence:  对 于 阿 富 汗 ， 我 意 识 到 — — 这 是 西 方 国 家 也 不 曾 意 识 到 的 — — 即 在 我 们 成 功 的 背 后 是 一 个 这 样 的 父 亲 — — 他 能 认 识 到 自 己 女 儿 的 价 值 ， 也 明 白 女 儿 的 成 功 ， 就 是 他 自 己 的 成 功 。

> Target Sentence:   what i ve come to realize about afghanistan , and this is something that is often dismissed in the west , that behind most of us who succeed is a father who recognizes the value in his daughter and who sees that her success is his success

> Translation:  and in fact , i realize that this is nothing in the west that is the father of the end of the world who can recognize his daughter s value , and it s also his daughter success , is his own
#### Vietnamese to English 
> Source Sentence: tôi đã khôngtưởng được những gì xảy đến với cuộcsống của một người tịnạn từ Bắc UNK thì sẽ như thếnào nhưng tôi sớm nhận ra rằng nó khôngnhững rất khókhăn màcòn vôcùng nguyhiểm vì những người tịnạn từ Bắc UNK vào trungQuốc đều bị coi là dân nhậpcư tráiphép 

> Target Sentence: i had no idea what life was going to be like as a north korean refugee , but i soon learned it s not only extremely difficult , it s also very dangerous , since north korean refugees are considered in china as illegal migrants

> Translation: i was longing for what happened to the life of refugees in northern ireland , and i would find out that it wasn t difficult to be very , but dangerous because refugees from north korea is not allowed by bullies

		      
## References:
- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
- [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
- [A decomposable attention model](https://arxiv.org/pdf/1606.01933.pdf)
- [Training Deeper Neural Machine Translation Models with Transparent Attention](http://aclweb.org/anthology/D18-1338)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)
- [Harvardnlp, The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
