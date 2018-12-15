# translate_machine_translation
Vietnamese and Chinese to English 

## Tasks:
- Recurrent neural network based encoder-decoder without attention
- Recurrent neural network based encoder-decoder with attention
- Replace the recurrent encoder with either convolutional or self-attention based encoder
- Build either or both fully self-attention translation system

## To Run:
### 
	python main.py --language zh --save_model_name zh_attn --FT_emb_path ft_emb --data_path MT_data --encoder_hidden_size 256 --decoder_hidden_size 256 --learning_rate 0.01


## References:
- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
- [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
- [A decomposable attention model](https://arxiv.org/pdf/1606.01933.pdf)
- [Training Deeper Neural Machine Translation Models with Transparent Attention](http://aclweb.org/anthology/D18-1338)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)
- [Harvardnlp, The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
