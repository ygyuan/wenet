from typing import List

import torch
import torch.nn.functional as F
from typeguard import check_argument_types
#from wenet.utils.common import get_activation
#from wenet.transformer.convolution import ConvolutionModule
#from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from itertools import groupby
import numpy as np
# import ctcdecode

def remove_duplicates_and_blank(hyp: List[int]) -> List[int]:                   
    new_hyp: List[int] = []                                                     
    cur = 0                                                                     
    while cur < len(hyp):                                                       
        if hyp[cur].all() != 0:                                                       
            new_hyp.append(hyp[cur])                                            
        prev = cur                                                              
        while cur < len(hyp) and hyp[cur].all() == hyp[prev].all():                         
            cur += 1                                                            
    return new_hyp

class CTC(torch.nn.Module):
    """CTC module"""
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
        """
        assert check_argument_types()
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)
        #self.states = self.ctc_lo.transpose(0, 1)


    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        #self.states = ys_hat
        ys_hat = ys_hat.log_softmax(2)
        # self.states = ys_hat
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.size(1)
        return loss

    
    def kd_forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                predictions: torch.Tensor) -> torch.Tensor:

        # scores, predictions = torch.max(targets, dim=-1)
        # predictions = self.argmax(targets)
        device = torch.device("cuda")

        pred_list = []
        pred_len_list = []
        for j in range(predictions.shape[0]):
            # Getting current predictions
            current_pred = predictions[j]
            current_pred = remove_duplicates_and_blank(list(current_pred.cpu().numpy()))
            current_pred_len = len(current_pred)
            pred_list.append(current_pred)
            pred_len_list.append(current_pred_len)

        max_pred_len = max(pred_len_list)
        for j in range(predictions.shape[0]):
            diff = max_pred_len - pred_len_list[j]
            for n in range(diff):
                pred_list[j].append(0)

        # generate soft label of teacher model
        fake_lab = torch.from_numpy(np.array(pred_list)) 
        fake_lab.to(device)
        fake_lab = fake_lab.int()
        fake_lab_lengths = torch.from_numpy(np.array(pred_len_list)).int()
        fake_lab_lengths.to(device)

        #print("fake ", fake_lab[0], fake_lab_lengths[0])
        print("fake ", fake_lab.size(), fake_lab_lengths.size(), fake_lab_lengths)

        # input_lens = (input_lens * log_probs.shape[1]).round().int()
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        #self.states = ys_hat
        ys_hat = ys_hat.log_softmax(2)
        # log_probs = log_probs.transpose(0, 1)
        loss=  self.ctc_loss(
            ys_hat,
            fake_lab,
            hlens,
            fake_lab_lengths,
        )
        loss = loss / ys_hat.size(1)
        return loss

    def convert_to_string(self, tokens, vocab, seq_len):
        return "".join([vocab[x] for x in tokens[0:seq_len]])
    
    def nbest_kd_forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                predictions: torch.Tensor,vocab_list:list, nbest: int) -> torch.Tensor:

        device = torch.device("cuda")
        batch_size=hs_pad.shape[0]

        decoder = ctcdecode.CTCBeamDecoder(
            vocab_list, beam_width=5, blank_id=0, num_processes=70,log_probs_input=True)
        # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #     executor.map(download_one, sites_list)
        beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(predictions)
        # output_str1 = self.convert_to_string(beam_results[0][0], self.vocab_list, out_seq_len[0][0])
        # output_str2 = self.convert_to_string(beam_results[1][0], self.vocab_list, out_seq_len[1][0])
        del decoder
        losses=0
        for n in range(0, nbest):
            pred_list = []
            pred_len_list = []

            for b in range(0, batch_size):
                utput_str1 = self.convert_to_string(beam_results[b][n], vocab_list, out_seq_len[b][n])
                pred_list.append(list(beam_results[b][n][ :out_seq_len[b][n]].cpu().numpy()))
                pred_len_list.append(out_seq_len[b][n])

            max_pred_len = max(pred_len_list)
            for j in range(batch_size):
                diff = max_pred_len - pred_len_list[j]
                for n in range(diff):
                    pred_list[j].append(0)

            # generate soft label of teacher model
            fake_lab = torch.from_numpy(np.array(pred_list))
            fake_lab.to(device)
            fake_lab = fake_lab.int()
            fake_lab_lengths = torch.from_numpy(np.array(pred_len_list)).int()
            fake_lab_lengths.to(device)

            # input_lens = (input_lens * log_probs.shape[1]).round().int()
            ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
            
            # ys_hat: (B, L, D) -> (L, B, D)
            ys_hat = ys_hat.transpose(0, 1)
            #self.states = ys_hat
            ys_hat = ys_hat.log_softmax(2)
            # log_probs = log_probs.transpose(0, 1)
            loss=  self.ctc_loss(
                ys_hat,
                fake_lab,
                hlens,
                fake_lab_lengths,
            )
            loss = loss / ys_hat.size(1)
            losses=losses+loss

        # ctc_loss = torch.cat(losses, dim=0)
        # ctc_loss=torch.tensor(losses)
        losses =losses/ nbest
        # del losses
        return losses

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """        
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)
    

    def softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """        
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)
