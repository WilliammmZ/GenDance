
from unicodedata import bidirectional
from matplotlib import style
import  torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) 
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
    

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # b x t x mic_embed
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # b x t x n_head x 64
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn  # b x lq x d_q
    

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) 

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)
   
  
class MultiHeadStyleAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        
        querys = self.W_query(query)  # [bsz, 1, d_model]
        keys = self.W_key(key)  # [bsz, len, d_model]
        values = self.W_value(key) # [bsz, len, d_model]

        
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h,bsz,1,d_model/n_head]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h,bsz,len,d_model/n_head]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  
        scores = scores / (self.key_dim ** 0.5)
        scores = torch.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values) 
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0) 
        
        return out, scores


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            batch_size, _, _ = attn.size()
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class STY_Generator(nn.Module):
    def __init__(self, 
                 kps_size, 
                 lstm_hidden_dim, 
                 lstm_n_layer, 
                 n_token, 
                 s_n_head, 
                 d_model):
        super(STY_Generator, self).__init__()
        
        self.lstm = nn.LSTM(kps_size, lstm_hidden_dim, lstm_n_layer, batch_first = True)
        
        self.global_memory = nn.Parameter(torch.FloatTensor(n_token, lstm_hidden_dim // s_n_head))
        torch.nn.init.normal_(self.global_memory, mean=0, std=0.5)
        
        self.attn = MultiHeadStyleAttention(lstm_hidden_dim, lstm_hidden_dim // s_n_head, d_model, s_n_head)
    

    def forward(self, kps_seq):
        bsz = kps_seq.size(0) 
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(kps_seq)
        dance_hidden = h_n[-1].squeeze(0) # bsz x hidden size
        query = dance_hidden.unsqueeze(1) # bsz x 1 x lstm_hidden
        key = torch.tanh(self.global_memory).unsqueeze(0).expand(bsz, -1, -1)
        style_embed, style_attn = self.attn(query, key)
      
        return style_embed, style_attn
    
    
class STY_Generator_shift(nn.Module):
    def __init__(self, 
                 kps_size, 
                 lstm_hidden_dim, 
                 lstm_n_layer, 
                 n_token, 
                 s_n_head, 
                 d_model):
        super(STY_Generator_shift, self).__init__()
        self.kps_size =kps_size
        
        self.num_conv = 3 #kps_feat = 41 only can div2^3
        en_fliter = [1, 32, 64, 128]
        convs = [nn.Conv2d(
            in_channels=en_fliter[i],
            out_channels=en_fliter[i+1],
            kernel_size=(3,3),
            stride=(2,2),
            padding=(1,1)
        ) for i in range((self.num_conv))]
        
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=en_fliter[i]) for i in range(1, self.num_conv + 1)])
        self.last_feature_dim = self.cal_dim(kps_size)
        
        self.lstm = nn.LSTM(self.last_feature_dim * en_fliter[-1], lstm_hidden_dim, lstm_n_layer, batch_first = True)
    
    
        self.global_memory = nn.Parameter(torch.FloatTensor(n_token, lstm_hidden_dim // s_n_head))
        torch.nn.init.normal_(self.global_memory, mean=0, std=0.5)
        
        self.attn = MultiHeadStyleAttention(lstm_hidden_dim, lstm_hidden_dim // s_n_head, d_model, s_n_head)
    
    
    def forward(self, kps_seq):
        bsz, _, n_feat = kps_seq.size()
        shift_map = kps_seq[:,1:,:] - kps_seq[:,:-1,:]
        out = shift_map.view(bsz, 1, -1, n_feat)
        
        for conv, bn in zip(self.convs, self.bns): 
            out = conv(out)
            out = bn(out)
            out = F.leaky_relu(out, 0.2) # bsz x 128 x time_step//8 x n_feat //8
        
        out = out.transpose(1,2) # bsz x time_step//8 x 128 x n_feat //8
        T = out.size(1)
        bsz = out.size(0)
        out = out.contiguous().view(bsz, T, -1)  # bsz x time_step//8 x (128 x n_feat //8)
        
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(out)
        dance_hidden = h_n[-1] # bsz x hidden size
        query = dance_hidden.unsqueeze(1) # bsz x 1 x lstm_hidden
        key = torch.tanh(self.global_memory).unsqueeze(0).expand(bsz, -1, -1)
        style_embed, style_attn = self.attn(query, key)
      
        return style_embed, style_attn
             
    def cal_dim(self, dim):
        for _ in range(self.num_conv):
            dim = (dim - 3 + 2 * 1)//2 + 1
        
        return dim

    
class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None, non_pad_mask=None):

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
 
    
class Encoder(nn.Module):
    def __init__(self, 
                 mic_size, 
                 d_mic_embed, 
                 d_model, 
                 d_inner, 
                 n_head, 
                 d_k, 
                 d_v, 
                 n_layers, 
                 dropout, 
                 max_len=1800):
        super(Encoder, self).__init__()
        
        self.mic_embed = nn.Linear(mic_size, d_mic_embed)
        
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_len + 1, d_mic_embed, padding_idx=0),
            freeze=True)
        #d_model = d_mic_embed
        
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.d_model = d_model
        
    def forward(self, mic_seq, mic_pos, mask=None):
        '''
        input -->
            mic_seq  [b x t x mic_size]
            mic_pos  [b x t x 1]
        output -->
            enc_output [b x t x d_k]
        '''
        #b x t x d_mic_embed  = b x t x d_mic_embed  + b x t x d_mic_embed
        enc_output =  self.mic_embed(mic_seq) + self.position_enc(mic_pos)
        
        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output, slf_attn_mask=mask) # out  & attn
        
        return enc_output,

class NoAR_Decoder(nn.Module):
    def __init__(self, kps_size, d_model, d_inner, dropout=0.1):
        super(NoAR_Decoder, self).__init__()
        self.hidden_size = d_inner        
        self.tgt_emb = nn.Linear(kps_size + d_model, kps_size + d_model)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(kps_size + d_model, self.hidden_size, n_layers=3, bidirectional=True)        
    def forward(self, hidden):       
        hidden = self.tgt_emb(hidden)
        in_frame = self.dropout(hidden)
        out, _ = self.bilstm(in_frame) # bxlx(2xh)
        return out
          

class AR_Decoder(nn.Module):
    def __init__(self, kps_size, d_inner, dropout=0.1):
        super(AR_Decoder, self).__init__()
        self.hidden_size = d_inner      
        self.tgt_emb = nn.Linear(kps_size, kps_size)
        self.dropout = nn.Dropout(dropout)             
        self.lstm1 = nn.LSTMCell(kps_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)      
    def init_state(self, bsz):
        c0 = torch.randn(bsz, self.hidden_size).cuda()
        c1 = torch.randn(bsz, self.hidden_size).cuda()
        c2 = torch.randn(bsz, self.hidden_size).cuda()
        h0 = torch.randn(bsz, self.hidden_size).cuda()
        h1 = torch.randn(bsz, self.hidden_size).cuda()
        h2 = torch.randn(bsz, self.hidden_size).cuda()
        vec_h = [h0, h1, h2]
        vec_c = [c0, c1, c2]
        return vec_h, vec_c
        
        
    def forward(self, kps, vec_h, vec_c):  
        kps = self.tgt_emb(kps)
        in_frame = self.dropout(kps)
        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h[0], (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h[1], (vec_h[2], vec_c[2]))
        vec_h_new = [vec_h0, vec_h1, vec_h2]
        vec_c_new = [vec_c0, vec_c1, vec_c2]      
        return vec_h2, vec_h_new, vec_c_new
         


# The whole model
class GST_Model(nn.Module):
    def __init__(self, mic_size,
                 kps_size, 
                 d_mic_embed, 
                 d_model, 
                 d_inner, 
                 n_head, 
                 d_k, 
                 d_v, 
                 n_layers, 
                 dropout, 
                 max_len, 
                 lstm_hidden_dim, 
                 lstm_n_layer, 
                 n_token, 
                 s_n_head,
                 ar = True,
                 shift=False):
        super(GST_Model, self).__init__()
             
        self.music_encoder = Encoder(mic_size, d_mic_embed, d_model, d_inner, n_head, d_k, d_v, n_layers, dropout, max_len)
        
        if shift:
            self.g_style = STY_Generator_shift(kps_size, lstm_hidden_dim, lstm_n_layer, n_token, s_n_head, d_model)
        else:
            self.g_style = STY_Generator(kps_size, lstm_hidden_dim, lstm_n_layer, n_token, s_n_head, d_model) 
        
        if ar:
            self.all_decoder = AR_Decoder(kps_size, d_inner, dropout)
            self.linear = nn.Linear(self.all_decoder.hidden_size + self.music_encoder.d_model, kps_size)
        else:
            self.all_decoder = NoAR_Decoder(kps_size, d_model, d_inner, dropout)
            self.linear = nn.Linear(self.all_decoder.hidden_size * 2, kps_size)
            
        
    
    def forward(self, mic_seq, mic_pos, kps_seq, init_kps, vec_h=None, vec_c=None):

        _, time_step, _ = kps_seq.size() 

        enc_outputs, *_ = self.music_encoder(mic_seq, mic_pos, mask=None)
        style_embedding, style_attn = self.g_style(kps_seq)
        
        style_attn = style_attn.squeeze(2)
        style_attn = style_attn.transpose(1,0)
    
        repeat_style_embedding = style_embedding.repeat(1, time_step, 1)   
        all_hidden = enc_outputs + repeat_style_embedding
        out_style_embedding = style_embedding.squeeze(1)
        
        if vec_h:
            preds = []
            dec_outputs = init_kps
            dec_input = dec_outputs.detach()
            for i in range(time_step):      
                dec_outputs, vec_h, vec_c = self.all_decoder(dec_input, vec_h, vec_c)
                dec_outputs = torch.cat([dec_outputs, all_hidden[:, i]], 1)
                dec_outputs = self.linear(dec_outputs)
                preds.append(dec_outputs)
                dec_input = dec_outputs.detach()  # dec_output
        
            outputs = [z.unsqueeze(1) for z in preds]
            outputs = torch.cat(outputs, dim=1)
        else:
            init_hidden = init_kps.repeat(1, time_step, 1)
            all_hidden = enc_outputs + repeat_style_embedding
            all_hidden = torch.concat([all_hidden, init_hidden], axis=-1)
            outputs = self.linear(self.all_decoder(all_hidden))
            
            
        return  outputs, out_style_embedding, style_attn
