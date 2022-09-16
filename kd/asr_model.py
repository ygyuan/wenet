# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import List, Optional, Tuple

import torch

import math
import random
# import kenlm
import yaml
import logging
import os


from torch.nn.utils.rnn import pad_sequence

from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import (TransformerDecoder,
                                       BiTransformerDecoder)
from wenet.transformer.encoder import ConformerEncoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.cmvn import load_cmvn
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)


from wenet.pretrain.mpc import MPC
from wenet.utils.checkpoint import load_checkpoint
from wenet.transformer.w2v_encoder import W2vConformerEncoder
from wenet.pretrain.w2v_loss import W2VLOSS
from wenet.transformer.grad_multiply import GradMultiply
from wenet.transformer.w2v_bert_encoder import W2vBertConformerEncoder
from wenet.transformer.hubert_encoder import HuBertConformerEncoder
from wenet.transformer.data2vec_encoder import Data2vecConformerEncoder
from wenet.transformer.w2v_hbert_encoder import W2vHbertConformerEncoder
from wenet.transformer.best_rq_encoder import BestRQConformerEncoder
from wenet.pretrain.hubert_loss import HubertLOSS
from wenet.pretrain.mpc import MaskedLanguageModel
from torch.nn import MSELoss
import numpy as np
# import ctcdecode
# from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        args: None,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        ctc_unit: CTC=None,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.ctc_unit=ctc_unit
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        self.kd_ctc=args.get('kd_ctc', False)
        self.kd_ctc_coef=args.get('kd_ctc_coef', 0.0)
        self.kd_temperature=args.get('kd_temperature',1.0)

        self.kd_dec=args.get('kd_dec', False)
        self.kd_dec_coef=args.get('kd_dec_coef', 0.0)

        self.kd_att=args.get('kd_att', False)
        self.kd_att_coef=args.get('kd_att_coef', 0.0)

        self.kd_reps=args.get('kd_reps', False)
        self.kd_reps_coef=args.get('kd_reps_coef', 0.0)

        self.kd_ctc_seq=args.get('kd_ctc_seq', False)
        self.kd_ctc_seq_coef=args.get('kd_ctc_seq_coef', 0.0)

        self.kd_ctc_nbestseq=args.get('kd_ctc_nbestseq', False)
        self.kd_ctc_nbestseq_coef=args.get('kd_ctc_nbestseq_coef', 0.0)
        self.kd_ctc_nbest=args.get('kd_ctc_nbest', 5)

        if self.kd_ctc_nbestseq:
            self.vocab_list= []
            # for j in range(vocab_size):
            #     self.vocab_list.append(str(i))
            dict=args.get('kd_ctc_dict', None)
            with open(dict, 'r',encoding='utf-8') as fin:
                for line in fin:
                    arr = line.strip().split()
                    assert len(arr) == 2
                    self.vocab_list.append(arr[0])

        self.kd_coef_plus=args.get('kd_coef_plus', True)
        self.kd_coef_independ =args.get('kd_coef_independ', False)
        self.kd_coef_pre=args.get('kd_coef_pre', False)

        self.teacherconfig=args.get('teacherconfig', None)
        if self.teacherconfig is not None:
            with open(self.teacherconfig, 'r') as fin:
                self.teacherconfigs = yaml.load(fin)

        self.teachercheckpoint=args.get('teachercheckpoint', None)
        if self.teachercheckpoint is not None:
            self.tec_model=init_asr_model(self.teacherconfigs)
            self.infos1 = load_checkpoint(self.tec_model, self.teachercheckpoint)
            if self.kd_reps and self.kd_reps_coef > 0.0:
                self.layer_dense=torch.nn.Linear(encoder._output_size,self.tec_model.encoder._output_size )
        else:
            self.infos1 = {}

        if self.kd_ctc:
            self.kd_kl_criterion = torch.nn.KLDivLoss(reduction="sum")
            # Prepare loss functions
            self.kd_mse = MSELoss()

        self.log_interval = args.get('log_interval', 100)


        
    def set_num_updates(self,num_updates):
        self.encoder.set_num_updates(num_updates)

    
    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        units:torch.Tensor,
        units_lengths: torch.Tensor,
        args,
        epoch,
        batch_idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """

        text2unit=args.get('text2unit',False)

        # assert text_lengths.dim() == 1, text_lengths.shape
        batch_size=speech.shape[0]
        encoder_out,encoder_mask=None,None
        sample_size=1

        # unlabel and label data
        if text !=None:
            if self.wav2ctc and text[0][0]!=2622:
                label_data=True
                self.w2v_coef=args.get('w2v_coef',0.07)
            else:
                label_data=False
                self.w2v_coef=args.get('w2v_coef',0.07)
        else:
            label_data=False
            self.w2v_coef=args.get('w2v_coef',1.0)

            if self.encoder_grad_mult>0 and args["step"]>=self.freeze_finetune_updates:
                if self.intermediate_ctc_layers!=None or self.w2v_layer != None or self.kd_ctc:
                    encoder_out, encoder_mask ,ext_res= self.encoder(speech, speech_lengths)
                else:
                    encoder_out, encoder_mask,_ = self.encoder(speech, speech_lengths)
                if self.encoder_grad_mult !=1.0:
                    encoder_out=GradMultiply.apply(encoder_out,self.encoder_grad_mult)
                    encoder_mask=GradMultiply.apply(encoder_mask,self.encoder_grad_mult)
            else:
                # logging.info("freeze updates %d"% args["step"])
                with torch.no_grad():
                    encoder_out, encoder_mask,_ = self.encoder(speech, speech_lengths)

            encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        loss_att,acc_att,loss_ctc,loss_kd_ctc,loss_kd_dec,loss_kd_ctcseq=0,0,0,0,0,0
        loss_intermediate_ctc=0
        loss_w2v=0
        loss_c=0
        losses_w2v = []
        loss_hubert=0
        loss_data2vec=0
        loss_w2vseq=0
        loss_bestrq=0
        


########################### distiall #####################################3

        if self.kd_ctc and self.kd_ctc_coef!=0:
            self.tec_model.eval()
            with torch.no_grad():
                kd_hs_pad, kd_hs_mask ,kd_ext_res= self.tec_model.encoder(speech, speech_lengths)
                kd_hs_len = kd_hs_mask.squeeze(1).sum(1)
                tmp_loss_kd_ctc = self.tec_model.ctc(kd_hs_pad,kd_hs_len,text,text_lengths)
                
            if self.kd_ctc_coef != 0:
                loss_kd_ctc = self.kd_kl_criterion(torch.log_softmax(self.ctc.states / self.kd_temperature, dim=-1),
                                                   torch.softmax(self.tec_model.ctc.states / self.kd_temperature,
                                                                 dim=-1)) / batch_size * math.pow(self.kd_temperature,
                                                                                        2)
                if batch_idx % self.log_interval == 0:
                    logging.info("Student model ctc loss: %f" % loss_ctc)
                    logging.info("Teacher model ctc loss: %f" % tmp_loss_kd_ctc)
                    logging.info("KD ctc loss: %f" % loss_kd_ctc)
        
        if self.kd_ctc_seq and self.kd_ctc_seq_coef!=0:
            self.tec_model.eval()
            with torch.no_grad():
                if not self.kd_ctc:
                    kd_hs_pad, kd_hs_mask ,kd_ext_res= self.tec_model.encoder(speech, speech_lengths)
                    kd_hs_len = kd_hs_mask.squeeze(1).sum(1)
                kd_targets=self.tec_model.ctc.argmax(kd_hs_pad)
                
            if self.kd_ctc_coef != 0:
                #self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
                loss_kd_ctcseq=self.ctc.kd_forward(encoder_out,encoder_out_lens,kd_targets)

                if batch_idx % self.log_interval == 0:
                    logging.info("Student model ctc seq loss: %f" % loss_kd_ctcseq)
        

        if self.kd_ctc_nbestseq and self.kd_ctc_nbestseq_coef!=0:
            self.tec_model.eval()
            with torch.no_grad():
                if not self.kd_ctc:
                    kd_hs_pad, kd_hs_mask ,kd_ext_res= self.tec_model.encoder(speech, speech_lengths)
                    kd_hs_len = kd_hs_mask.squeeze(1).sum(1)
                kd_targets=self.tec_model.ctc.log_softmax(kd_hs_pad)
                
            if self.kd_ctc_coef != 0:
                #self.ctc(encoder_out, encoder_out_lens, text, text_lengths)
                loss_kd_ctcseq=self.ctc.nbest_kd_forward(encoder_out,encoder_out_lens,kd_targets,self.vocab_list,self.kd_ctc_nbest)

                if batch_idx % self.log_interval == 0:
                    logging.info("Student model nbest ctc seq loss: %f" % loss_kd_ctcseq)
        
        
        loss_kd_att=0.0
        if self.kd_att and self.kd_att_coef!=0:
            layers_per_block = int(self.tec_model.encoder.num_blocks / self.encoder.num_blocks)
            new_teacher_atts = [
                         kd_ext_res["att_outputs"][i * layers_per_block + layers_per_block - 1]
                        for i in range(self.encoder.num_blocks)
                    ]
            
            for student_att, teacher_att in zip(ext_res["att_outputs"], new_teacher_atts):
                student_att = torch.where(student_att <= -1e2,
                                            torch.zeros_like(student_att),
                                            student_att)
                teacher_att = torch.where(teacher_att <= -1e2,
                                            torch.zeros_like(teacher_att),
                                            teacher_att)

                tmp_loss = self.kd_mse(student_att, teacher_att)
                loss_kd_att += tmp_loss
            if batch_idx % self.log_interval == 0:
                    logging.info("KD att loss: %f" % loss_kd_att)
        
        loss_kd_resp=0.0
        if self.kd_reps and self.kd_reps_coef !=0:
            layers_per_block = int(self.tec_model.encoder.num_blocks / self.encoder.num_blocks)
            new_teacher_reps = [
                         kd_ext_res["layer_outputs"][i * layers_per_block + layers_per_block - 1]
                        for i in range(self.encoder.num_blocks)
                    ]
            # new_teacher_reps = [
            #     teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)
            # ]

            tmp = []
            for sid,layer_output in enumerate(ext_res["layer_outputs"]):
                tmp.append(self.layer_dense(layer_output))
            new_student_reps = tmp
            for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                tmp_loss = self.kd_mse(student_rep, teacher_rep)
                loss_kd_resp += tmp_loss
            if batch_idx % self.log_interval == 0:
                logging.info("KD resp loss: %f" % loss_kd_resp)

        if self.kd_dec and self.kd_dec_coef!=0:
            with torch.no_grad():
                mt_pred_pad, r_mt_pred_pad,_ = self.tec_model.decoder(kd_hs_pad,kd_hs_mask, ys_in_pad, ys_in_lens,r_ys_in_pad,
                                                            self.reverse_weight)
                mt_acc = th_accuracy(
                    mt_pred_pad.view(-1, self.vocab_size), ys_out_pad, ignore_label=self.ignore_id
                )
                if batch_idx % self.log_interval == 0:
                    logging.info("Student model accuracy: %f" % acc_att)
                    logging.info("Teacher mt model accuracy: %f" % mt_acc)
            
            if self.kd_dec_coef != 0:
                loss_kd_rdec = torch.tensor(0.0)
                loss_kd_ldec = self.kd_kl_criterion(torch.log_softmax(decoder_out / self.kd_temperature, dim=-1),
                                                  torch.softmax(mt_pred_pad / self.kd_temperature,
                                                                dim=-1)) / batch_size * math.pow(self.kd_temperature, 2)
                if self.reverse_weight >0:
                    loss_kd_rdec = self.kd_kl_criterion(torch.log_softmax(r_decoder_out / self.kd_temperature, dim=-1),
                                                    torch.softmax(r_mt_pred_pad / self.kd_temperature,
                                                                    dim=-1)) / batch_size * math.pow(self.kd_temperature, 2)
                loss_kd_dec = loss_kd_ldec * (
                1 - self.reverse_weight) + loss_kd_rdec * self.reverse_weight
                if batch_idx % self.log_interval == 0:
                    logging.info("KD dec probability loss: %f" % float(loss_kd_dec))
                    logging.info("dec var: %f" % float(torch.var(torch.softmax(decoder_out, dim=-1), dim=-1).mean()))
                    logging.info("KD dec var: %f" % float(torch.var(torch.softmax(mt_pred_pad, dim=-1), dim=-1).mean()))
                    logging.info("kd_ctc_cf: %f kd_dec_cf: %f"%(self.kd_ctc_coef,self.kd_dec_coef))

########################### distiall ##################################################

                if self.training:
                        loss=self.ctc_weight * loss_ctc + \
                        self.kd_ctc_coef * loss_kd_ctc + self.kd_ctc_seq_coef*loss_kd_ctcseq + +self.kd_ctc_nbestseq_coef*loss_kd_ctcseq +\
                        self.kd_att_coef*loss_kd_att+self.kd_reps_coef*loss_kd_resp + \
                        (1 -self.ctc_weight-self.kd_ctc_coef-self.kd_ctc_seq_coef- self.kd_ctc_nbestseq_coef-self.kd_att_coef-self.kd_reps_coef) * \
                        (loss_att + loss_kd_dec * self.kd_dec_coef)


        return loss, loss_att, loss_ctc,loss_w2v,sample_size

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     r_ys_in_pad,
                                                     self.reverse_weight)
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att,decoder_out,r_decoder_out
    
      
    def forward_encoder_argmax(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, L, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk, it's
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
        Returns:
            encoder output tensor, lens and mask
        """

        # encoder_out, encoder_mask ,_ = self.encoder(xs,xs_lens)
        encoder_out, encoder_mask,_ = self.encoder(xs,xs_lens)

        batch_size = encoder_out.size(0)
        maxlen = encoder_out.size(1)
        # encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        ctc_probs = self.ctc.log_softmax(encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)

        return topk_index,encoder_out, encoder_mask
        # return ctc_probs,encoder_out, encoder_mask
      
  
    def forward_encoder(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, L, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk, it's
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
        Returns:
            encoder output tensor, lens and mask
        """

        # encoder_out, encoder_mask ,_ = self.encoder(xs,xs_lens)
        encoder_out, encoder_mask,_ = self.encoder(xs,xs_lens)

        batch_size = encoder_out.size(0)
        maxlen = encoder_out.size(1)
        # encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        ctc_probs = self.ctc.log_softmax(encoder_out)  # (B, maxlen, vocab_size)
        # topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        # topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)

        # return topk_index,encoder_out, encoder_mask
        return ctc_probs,encoder_out, encoder_mask
    
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        subsampling_cache: Optional[torch.Tensor] = None,
        elayers_output_cache: Optional[List[torch.Tensor]] = None,
        conformer_cnn_cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, L, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk, it's
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
        Returns:
            encoder output tensor, lens and mask
        """
        (encoder_out,subsampling_cache,elayers_output_cache,conformer_cnn_cache)=self.encoder.forward_chunk(xs,0,0,
                                    subsampling_cache,
                                    elayers_output_cache,
                                    conformer_cnn_cache)
        
        batch_size = encoder_out.size(0)
        maxlen = encoder_out.size(1)
        # encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        ctc_probs = self.ctc.log_softmax(encoder_out)  # (B, maxlen, vocab_size)
        # topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        # topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)

        # return topk_index,encoder_out, encoder_mask
        return ctc_probs,encoder_out,encoder_mask

    def set_onnx_mode(self,
    onnx:bool):
        self.encoder.set_onnx_mode(onnx)


    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            # encoder_out, encoder_mask = self.encoder(
            #     speech,
            #     speech_lengths,
            #     decoding_chunk_size=decoding_chunk_size,
            #     num_decoding_left_chunks=num_decoding_left_chunks
            # )  # (B, maxlen, encoder_dim)
            if self.intermediate_ctc_layers or self.w2v_layer:
                encoder_out, encoder_mask ,ext_res= self.encoder(
                    speech,
                    speech_lengths
                )  # (B, maxlen, encoder_dim)
                # encoder_out=ext_res['inter_outputs'][0]
            else:
                encoder_out, encoder_mask,_ = self.encoder(
                    speech,
                    speech_lengths
                )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask
        
    def forward_encoder_mask(
        self,
        xs: torch.Tensor,
        masks: torch.Tensor,
        decoding_chunk_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        encoder_out, encoder_mask = self.encoder.forward_mask(xs,masks)

        ctc_probs = self.ctc.log_softmax(encoder_out)  # (B, maxlen, vocab_size)

        return ctc_probs,encoder_out, encoder_mask

    @torch.jit.export
    def forward_attention_decoder(
    # def forward(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(num_hyps,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=encoder_out.device)
        decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps,
            hyps_lens)  # (num_hyps, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        return decoder_out


    @torch.jit.export
    def recognize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 10,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> torch.Tensor:
        """ Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (B*N, 1, max_len)

        hyps = torch.ones([running_size, 1], dtype=torch.long,
                          device=device).fill_(self.sos)  # (B*N, 1)
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(
            device)  # (B*N, 1)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        cache: Optional[List[torch.Tensor]] = None
        # 2. Decoder forward step by step
        for i in range(1, maxlen + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1).to(device)  # (B*N, i, i)
            # logp: (B*N, vocab)
            logp, cache = self.decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache)
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)
            # 2.3 Seconde beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (B*N)

            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  # (B*N)
            best_hyps_index = best_k_index // beam_size
            last_best_k_hyps = torch.index_select(
                hyps, dim=0, index=best_hyps_index)  # (B*N, i)
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                             dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], self.eos).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_index = torch.argmax(scores, dim=-1).long()
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        return best_hyps

    def ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[List[int]]:
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.eos)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps

    def _ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """ CTC prefix beam search inner implementation

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[List[int]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # For CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out

    def ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[int]:
        """ Apply CTC prefix beam search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[int]: CTC prefix beam search nbest results
        """
        hyps, _ = self._ctc_prefix_beam_search(speech, speech_lengths,
                                               beam_size, decoding_chunk_size,
                                               num_decoding_left_chunks,
                                               simulate_streaming)
        confid=hyps[0][1]*100/(speech_lengths[0].float()/4)
        token_len=len(hyps[0][0])
        logging.info("confidence: %f %f %d" % (hyps[0][1],confid,speech_lengths[0]))
        return hyps[0][0],hyps[0][1],token_len

    def convert_to_string(self, tokens, vocab, seq_len):
        return "".join([vocab[x] for x in tokens[0:seq_len]])

    def _batch_ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        vocab_list: list,
        decoding_chunk_size: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """ CTC prefix beam search inner implementation

        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        num_decoding_left_chunks=-1
        # For CTC prefix beam search, we only support batch_size=1
        # assert batch_size == 1
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
 
        batch_encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        batch_ctc_probs = self.ctc.log_softmax(
                batch_encoder_out)  # (1, maxlen, vocab_size)
        batch_hyps=[]

        decoder = ctcdecode.CTCBeamDecoder(
            vocab_list, beam_width=beam_size, blank_id=0, num_processes=10,log_probs_input=True)
        # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #     executor.map(download_one, sites_list)
        beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(batch_ctc_probs)
        # output_str1 = self.convert_to_string(beam_results[0][0], self.vocab_list, out_seq_len[0][0])
        # output_str2 = self.convert_to_string(beam_results[1][0], self.vocab_list, out_seq_len[1][0])
        del decoder

        for b in range(0, batch_size):
            batch_hyps.append(beam_results[b][0][ :out_seq_len[b][0]].cpu().numpy())
        return batch_hyps, batch_encoder_out 

    def batch_ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        vocab_list: list,
        decoding_chunk_size: int = -1,
        simulate_streaming: bool = False,
    ) -> List[int]:
        """ Apply CTC prefix beam search
        """
        batch_hyps, _ = self._batch_ctc_prefix_beam_search(speech, speech_lengths,
                                               beam_size,vocab_list,decoding_chunk_size,
                                               simulate_streaming)
        return batch_hyps

    def ngram_rescoring(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        ngram_model =None,
        dict =None,
        ngram_weight: float=0.0,
        decoder_weight: float=0.0
    ) -> List[int]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out
        """
        reverse_weight=0.0
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        hyps, encoder_out = self._ctc_prefix_beam_search(
            speech, speech_lengths, beam_size, decoding_chunk_size,
            simulate_streaming)
        
        assert len(hyps) == beam_size

        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)
        ori_hyps_pad = hyps_pad
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        # decoder_out, _ = self.decoder(
        #     encoder_out, encoder_mask, hyps_pad,
        #     hyps_lens)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
 
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            hyp_string=""
            for j, w in enumerate(hyp[0]):
                hyp_string += dict[w]
                hyp_string +=" "
                score += decoder_out[i][j][w]
            # for j, w in enumerate(hyp[0]):
                # score += decoder_out[i][j][w]

            score += decoder_out[i][len(hyp[0])][self.eos]
            score =score*decoder_weight
            score += ngram_model.score(hyp_string, bos = True, eos = True)*ngram_weight
            # add ctc score
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        return hyps[best_index][0]

    def ctc_greedy_search_prob(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1, 
        simulate_streaming: bool = False,
    ) -> torch.Tensor:
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # Let's assume B = batch_size
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech, decoding_chunk_size=decoding_chunk_size
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size
            )  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        return ctc_probs


    def attention_rescoring(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
    ) -> List[int]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns:
            List[int]: Attention rescoring result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.decoder, 'right_decoder')
        device = speech.device
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        hyps, encoder_out = self._ctc_prefix_beam_search(
            speech, speech_lengths, beam_size, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)

        assert len(hyps) == beam_size
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.eos]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][self.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        return hyps[best_index][0]

    @torch.jit.export
    def subsampling_rate(self) -> int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self.encoder.embed.subsampling_rate

    @torch.jit.export
    def right_context(self) -> int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self.encoder.embed.right_context

    @torch.jit.export
    def sos_symbol(self) -> int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self.sos

    @torch.jit.export
    def eos_symbol(self) -> int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self.eos

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        subsampling_cache: Optional[torch.Tensor] = None,
        elayers_output_cache: Optional[List[torch.Tensor]] = None,
        conformer_cnn_cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor],
               List[torch.Tensor]]:
        """ Export interface for c++ call, give input chunk xs, and return
            output from time 0 to current chunk.

        Args:
            xs (torch.Tensor): chunk input
            subsampling_cache (Optional[torch.Tensor]): subsampling cache
            elayers_output_cache (Optional[List[torch.Tensor]]):
                transformer/conformer encoder layers output cache
            conformer_cnn_cache (Optional[List[torch.Tensor]]): conformer
                cnn cache

        Returns:
            torch.Tensor: output, it ranges from time 0 to current chunk.
            torch.Tensor: subsampling cache
            List[torch.Tensor]: attention cache
            List[torch.Tensor]: conformer cnn cache

        """
        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          subsampling_cache,
                                          elayers_output_cache,
                                          conformer_cnn_cache)

    @torch.jit.export
    def ctc_activation(self, xs: torch.Tensor) -> torch.Tensor:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (torch.Tensor): encoder output

        Returns:
            torch.Tensor: activation before ctc

        """
        return self.ctc.log_softmax(xs)


    @torch.jit.export
    def is_bidirectional_decoder(self) -> bool:
        """
        Returns:
            torch.Tensor: decoder output
        """
        if hasattr(self.decoder, 'right_decoder'):
            return True
        else:
            return False

    @torch.jit.export
    def forward_attention_decoder(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            r_hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad eos at the begining which is used fo right to left decoder
            reverse_weight: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(num_hyps,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=encoder_out.device)
        # input for right to left decoder
        # this hyps_lens has count <sos> token, we need minus it.
        r_hyps_lens = hyps_lens - 1
        # this hyps has included <sos> token, so it should be
        # convert the original hyps.
        r_hyps = hyps[:, 1:]
        r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(self.ignore_id))
        r_hyps, _ = add_sos_eos(r_hyps, self.sos, self.eos, self.ignore_id)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
            reverse_weight)  # (num_hyps, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)

        # right to left decoder may be not used during decoding process,
        # which depends on reverse_weight param.
        # r_dccoder_out will be 0.0, if reverse_weight is 0.0
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        return decoder_out, r_decoder_out


    def forward_chunk_init(
        self,
        xs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:

        batch_size = xs.shape[0];
        (ys, subsampling_cache, elayers_output_cache,
             conformer_cnn_cache) = self.encoder.forward_chunk_init(xs)

        ctc_probs = self.ctc.log_softmax(ys)  # (B, maxlen, vocab_size)

        return ctc_probs, ys, subsampling_cache, elayers_output_cache, conformer_cnn_cache

    def forward_chunk_noinit(
        self,
        xs: torch.Tensor,
        subsampling_cache: torch.Tensor,
        elayers_output_cache: List[torch.Tensor],
        conformer_cnn_cache: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor],
               List[torch.Tensor]]:
 
        batch_size = xs.shape[0];
        (ys, subsampling_cache, elayers_output_cache,
             conformer_cnn_cache) = self.encoder.forward_chunk_noinit(xs,
                                                       subsampling_cache,
                                                       elayers_output_cache,
                                                       conformer_cnn_cache)

        ctc_probs = self.ctc.log_softmax(ys)  # (B, maxlen, vocab_size)

        return ctc_probs, ys, subsampling_cache, elayers_output_cache, conformer_cnn_cache


def init_asr_model(configs):
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']
    unit_size=configs.get('output_unit_dim',0)

    if configs.get('wav2seq',False):
        vocab_size=configs.get('latent_vars',1024)
        logging.info("wav2seq model vocabsize change: %f" % vocab_size)

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'transformer')

    text2unit=configs.get('text2unit',False)
    wav2vec_train=configs.get('w2v_encoder', False)
    wav2bert_train=configs.get('w2v_bert_encoder', False)
    hubert_train=configs.get('hubert_encoder', False)
    data2vec_train=configs.get('data2vec_encoder',False)
    wav2hbert_train=configs.get('w2v_hubert_encoder',False)
    bestrq_train=configs.get('best_rq_encoder',False)
    self_condition=configs.get('self_condition',False)
    if text2unit:
        ctc_unit = CTC(unit_size, configs['encoder_conf']['output_size'])
    else:
        ctc_unit=None
    ctc = CTC(vocab_size, configs['encoder_conf']['output_size'])
    if wav2vec_train:
        if encoder_type == 'conformer':
            encoder = W2vConformerEncoder(configs,input_dim,
                                    global_cmvn=global_cmvn,
                                    **configs['encoder_conf'],
                                    )
        else:
            encoder = TransformerEncoder(configs,input_dim,
                                        global_cmvn=global_cmvn,
                                        **configs['encoder_conf'])
    elif wav2bert_train:
        if encoder_type == 'conformer':
            if self_condition:
                encoder = W2vBertConformerEncoder(configs,input_dim,
                                    global_cmvn=global_cmvn,
                                    **configs['encoder_conf'],
                                    ctc_softmax=ctc.softmax,
                                    conditioning_layer_dim=vocab_size)
            else:
                encoder = W2vBertConformerEncoder(configs,input_dim,
                                    global_cmvn=global_cmvn,
                                    **configs['encoder_conf'])
    elif hubert_train:
        if encoder_type == 'conformer':
            encoder = HuBertConformerEncoder(configs,input_dim,
                                    global_cmvn=global_cmvn,
                                    **configs['encoder_conf'])
    elif data2vec_train:
        encoder = Data2vecConformerEncoder(configs,input_dim,
                        global_cmvn=global_cmvn,
                        **configs['encoder_conf'])
    elif wav2hbert_train:
        if encoder_type == 'conformer':
            encoder = W2vHbertConformerEncoder(configs,input_dim,
                                    global_cmvn=global_cmvn,
                                    **configs['encoder_conf'])
    elif bestrq_train:
        if encoder_type == 'conformer':
            encoder = BestRQConformerEncoder(configs,input_dim,
                                    global_cmvn=global_cmvn,
                                    **configs['encoder_conf'])
    else:
        if encoder_type == 'conformer':
            if self_condition:
                encoder = ConformerEncoder(configs,input_dim,
                                    global_cmvn=global_cmvn,
                                    **configs['encoder_conf'],
                                    ctc_softmax=ctc.softmax,
                                    conditioning_layer_dim=vocab_size)
            else:
                encoder = ConformerEncoder(configs,input_dim,
                                    global_cmvn=global_cmvn,
                                    **configs['encoder_conf'])
    
    ctc_weight=configs.get('ctc_weight',0.3)
    logging.info("model ctc weight: %f" % ctc_weight)
    if ctc_weight==1.0:
        decoder=None
    else:
        if decoder_type == 'transformer':
            decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                        **configs['decoder_conf'])
        else:
            # assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
            # assert configs['decoder_conf']['r_num_blocks'] > 0
            decoder = BiTransformerDecoder(vocab_size, encoder.output_size(),
                                        **configs['decoder_conf'])
   
    model = ASRModel(
        configs,
        vocab_size=vocab_size,
        encoder=encoder,
        decoder=decoder,
        ctc=ctc,
        ctc_unit=ctc_unit,
        **configs['model_conf'],
    )
    return model
	
