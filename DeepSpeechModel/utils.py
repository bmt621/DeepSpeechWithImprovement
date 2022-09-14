import re
from torchmetrics import WordErrorRate,CharErrorRate
from pyctcdecode import build_ctcdecoder
import torchaudio
from torch import nn,tensor 
import numpy as np


def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


class TextTransform:
    """maps characters to integer and vice versa"""
    
    def __init__(self):
        chars = ["'",'<SPACE>','a','b','c','d','e','f','g','h','i','j','k',
                 'l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
        self.char_map = dict({v:i for i,v in enumerate(chars)})
        self.index_map = dict({i:v for i,v in enumerate(chars)})
        
       
    def text_to_int(self,texts):
        new_texts = self.remove_special_chars(texts)
        data = [self.char_map['<SPACE>'] if val is ' ' else self.char_map[val] for val in new_texts]
        
        return data
    
    
    def int_to_text(self,data):
        
        texts = [' ' if self.index_map[val]=='<SPACE>' else self.index_map[val] for val in data]
        
        return ''.join(texts)
        
    
    def remove_special_chars(self,texts):
        chars_not_to_ignore = r"[^a-zA-Z' ]"
        
        try:
            import re
            
            new_text = re.sub(chars_not_to_ignore,'',texts)
            return new_text
        
        except (ImportError,ModuleNotFoundError) as e:
            print(e)
            
    
    
def data_processor(data,text_transforms,train=True):
    spectrograms = []
    labels = []
    input_len = []
    label_len = []
    
    if train:
        transformers = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate = 16000,n_mels = 128),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35),
        )
        
    else:
        transformers = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate = 16000,n_mels = 128)
            
        )

    for (wave,text) in data:
        
        spec = transformers(wave).squeeze(0).transpose(0,1) # (time,n_feat)
        spectrograms.append(spec)
        label = tensor(text_transforms.text_to_int(text.lower()))
        labels.append(label)
        
        input_len.append(spec.shape[0]//2)
        label_len.append(len(labels))
        
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms,\
    batch_first=True).unsqueeze(1).transpose(2,3) #(batch,n_channels,n_feats,time)
    
    labels = nn.utils.rnn.pad_sequence(labels,batch_first=True) #(batch,len_sequence)
    
    return spectrograms,labels,input_len,label_len




def train(model,device,train_loader,criterion,optimizer,scheduler,epoch,iter_meter):
    
    model.train()
    data_len = len(train_loader.dataset)

    for batch_idx, data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = data 
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        out = model(spectrograms) # (batch, time, n_class)
        out= F.log_softmax(out,dim=2)
        output = out.transpose(0, 1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()
        iter_meter.step()

        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(spectrograms), data_len,
                    100. * batch_idx / len(train_loader), loss.item()))
            


            
def test(model,device,test_loader,criterion,epoch, iter_meter):
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    cer = CharErrorRate()
    wer = WordErrorRate()
    
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)
         
            out = model(spectrograms) # (batch, time,n_class)
            out = F.log_softmax(out,dim=2)
            output = out.transpose(0,1) # (time,batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)

            test_loss += loss.item() / len(test_loader)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths)

            for j in range(len(decoded_preds)):
                test_cer.append(cer(decoded_preds[j],decoded_targets[j]))
                test_wer.append(wer(decoded_preds[j],decoded_targets[j]))

    
    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)
    
    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))