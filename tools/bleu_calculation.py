'''
Insturction:
bleu_cal = BLEUCalculator(smooth="floor", smooth_floor=0.01,
                 lowercase=False, use_effective_order=True,
                 tokenizer=DEFAULT_TOKENIZER)
bleu_cal.bleu(sys, ref)

sys: decoded words (list or string)
ref: reference sentence (list or string)

returns: bleu_score, correct, total, precisions, brevity_penalty, sys_len, ref_len
'''


from sacrebleu import corpus_bleu, TOKENIZERS, DEFAULT_TOKENIZER

class BLEUCalculator():

    def __init__(self,
                 smooth="floor", smooth_floor=0.01,
                 lowercase=False, use_effective_order=True,
                 tokenizer=DEFAULT_TOKENIZER):
        self.smooth = smooth
        self.smooth_floor = smooth_floor
        self.lowercase = lowercase
        self.use_effective_order = use_effective_order
        self.tokenizer = tokenizer

    def bleu(self, sys, ref, score_only=False):
#         if isinstance(sys, str):
#             _s = sys
#         else:
#             _s = ' '.join(sys)
#         if isinstance(ref, str):
#             _refs = ref
#         else:
#             _refs = ' '.join(ref)
        bleu = corpus_bleu(
                sys, ref,
                smooth=self.smooth, smooth_floor=self.smooth_floor,
                force=False, lowercase=self.lowercase,
                tokenize=self.tokenizer,
                use_effective_order=self.use_effective_order)

        if score_only:
            return bleu.score
        else:
            return bleu

