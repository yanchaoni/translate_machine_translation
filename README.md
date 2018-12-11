# translate_machine_translation
Vietnamese and Chinese to English 

## Task
- Recurrent neural network based encoder-decoder without attention
- Recurrent neural network based encoder-decoder with attention
- Replace the recurrent encoder with either convolutional or self-attention based encoder.
- [Optional] Build either or both fully self-attention translation system or/and multilingual translation system.

## Possible references:
- [BERT](https://arxiv.org/pdf/1810.04805.pdf)
- [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
- [A decomposable attention model](https://arxiv.org/pdf/1606.01933.pdf)
- [OpenAI Transformer](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Training Deeper Neural Machine Translation Models with Transparent Attention](http://aclweb.org/anthology/D18-1338)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025.pdf)
- [Harvardnlp, The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## To do:
- [x] need to batchify: sort, pack padded seq etc.
- [x] attention: need mask when doing attention
- [x] beam search
- [x] pretrained embedding + BLEU
- [x] preprocess
- [ ] mask out loss after EOS
- [x] save logs
- [x] argparser
- [x] debug vietnamese
- [x] learning rate annealing
- [x] add c, y(t-1) to linear layer
- [ ] debug self attention based encoder
- [x] try new Chinese char embd
- [ ] transformer
- [x] fit LSTM 

## Run training
	python main.py --language zh --save_model_name zh_attn --FT_emb_path ft_emb --data_path MT_data --encoder_hidden_size 256 --decoder_hidden_size 256 --learning_rate 0.01
