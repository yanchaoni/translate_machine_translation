# translate_machine_translation
Vietnamese and Chinese to English 
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
- [ ] attention: need mask when doing attention
- [x] beam search
- [x] pretrained embedding + BLEU
- [x] preprocess
- [ ] mask out loss after EOS
- [x] save logs
- [ ] argparser
- [ ] debug vietnamese
- [x] learning rate annealing
- [x] add c, y(t-1) to linear layer
