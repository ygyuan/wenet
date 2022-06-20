import torch
import torch.nn.functional as F
from typeguard import check_argument_types


class CTCBranch(torch.nn.Module):
    """CTC module"""
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0
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
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.dropout_rate = dropout_rate

    #def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor) -> torch.Tensor:
    def forward(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        ys_hat = ys_hat.log_softmax(2)
        return ys_hat

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)
