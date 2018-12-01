import torch
from tools.Constants import *
class Beam(object):
    """
    inspired by OpenNMT https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam.py
    """
    def __init__(self, beam_width, min_len, n_best, device):
        self.beam_width = beam_width
        self.scores = torch.zeros(beam_width).to(device)
        self.prev_ks = []
        self.next_ys = [torch.LongTensor(beam_width).fill_(PAD).to(device)]
        self.next_ys[0][0] = SOS
        # stop condition
        self.eos_top = False
        self.finished = []
        self.min_len = min_len
        self.n_best = n_best
    
    def get_current_state(self):
        return self.next_ys[-1]

    def get_current_origin(self):
        return self.prev_ks[-1]
    
    def advance(self, word_probs):
        """
        word_probs: (beam_width, vocab_size)
        """
        num_words = word_probs.size(1)
        cur_len = len(self.next_ys)
        if cur_len < self.min_len:
            for k in range(len(word_probs)):
                word_probs[k][EOS] = -1e20
        # Don't select PAD
        for k in range(len(word_probs)):
            word_probs[k][PAD] = -1e20
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + \
                self.scores.unsqueeze(1).expand_as(word_probs)
            # Don't expand EOS any more
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == EOS:
                    beam_scores[i] = -1e20
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(k=self.beam_width, dim=0,
                                                            largest=True, sorted=True)
        self.scores = best_scores
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))
        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == EOS:
                s = self.scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
        
        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0] == EOS:
            self.eos_top = True

        return self.done()
    
    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best
    
    def sort_finished(self):
        if not self.finished:
            i = torch.argmax(self.scores)
            self.finished.append((self.scores[i], len(self.next_ys)-1, i))
        self.finished = sorted(self.finished, key=lambda a: -a[0] / (a[1] ** 0))
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp = []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return hyp[::-1]


