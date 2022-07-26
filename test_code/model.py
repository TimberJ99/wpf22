# -*-Encoding: utf-8 -*-
import paddle
import paddle.nn as nn


class BaselineGruModel(nn.Layer):
    """
    Desc:
        step predict GRU model
    """

    def __init__(self, settings):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(BaselineGruModel, self).__init__()
        self.output_len = settings["output_len"]
        self.hidC = settings["in_var"]
        self.hidR = 48
        self.out = settings["out_var"]
        self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=settings["lstm_layer"],
                           time_major=False)
        self.projection = nn.Linear(self.hidR, self.out)

    def forward(self, x_enc_group):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc:
        Returns:
            A tensor
        """
        ## time_major=False
        samples = paddle.zeros([x_enc_group.shape[0], self.output_len, self.out])
        for seq_len in [144, 72, 36, 18, 12, 6, 3]:
            x_enc = x_enc_group[:, -seq_len:, :]
            out_one = int(seq_len * 2)
            x = paddle.zeros([x_enc.shape[0], out_one, x_enc.shape[2]])
            x_enc = paddle.concat((x_enc, x), 1)
            dec, _ = self.lstm(x_enc)
            sample = self.projection(self.dropout(dec))
            samples[:, :out_one, -self.out:] = sample[:, -out_one:, -self.out:]
        return samples