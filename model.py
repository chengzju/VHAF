import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertTokenizer, BertModel, BertConfig





class MLP(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_uniform_(self.fc_1.weight)
        nn.init.xavier_uniform_(self.fc_2.weight)

    def forward(self, x):
        h1 = F.tanh(self.fc_1(x))
        out = self.fc_2(h1)
        return out




class VHAF(nn.Module):
    def __init__(self, args, device, N, K, Q):
        super(VHAF, self).__init__()
        self.args = args
        self.device = device
        self.N = N
        self.K = K
        self.Q = Q
        bert_config = BertConfig.from_pretrained(args.bert_path)
        self.bert_encoder = BertModel.from_pretrained(args.bert_path, config=bert_config)
        self.bert_init()
        self.f1 = nn.Linear(bert_config.hidden_size, 256)
        self.f2 = nn.Linear(256, 1)
        self.o_emb_dim = 1024
        self.o_lat_dim = 768
        self.encoder = nn.Linear(bert_config.hidden_size * 2, self.o_emb_dim)
        self.mu = nn.Linear(self.o_emb_dim, self.o_lat_dim)
        self.logvar = nn.Linear(self.o_emb_dim, self.o_lat_dim)
        self.threshold_MLP = MLP(args, self.o_lat_dim, self.o_lat_dim // 2, 1)
        self.bce_wo_sigmoid_loss = nn.BCELoss()

    def bert_init(self):
        all_layers = ['embeddings', 'layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6',
                      'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11', 'pooler']
        if self.args.freeze_bert:
            unfreeze_layers = all_layers[self.args.freeze_layer_num + 1:]
            for name, param in self.bert_encoder.named_parameters():
                param.requires_grad = False
                for ele in unfreeze_layers:
                    if ele in name:
                        param.requires_grad = True
                        break

    def get_pos_view(self, t, mask):
        t_pos = t * mask.unsqueeze(1).unsqueeze(0).unsqueeze(-1)
        t_pos = torch.sum(t_pos, dim=-2)
        return t_pos

    def get_mean_mu_and_var(self, logvar, mu, cnt):
        reciprocal_var = 1 / torch.exp(logvar)
        var_denominator = torch.sum(reciprocal_var, dim=-2)
        var_mean = cnt / var_denominator
        mu_mean = torch.sum(reciprocal_var * mu, dim=-2) / var_denominator
        return var_mean, mu_mean


    def get_kl_pred(self, s_mu, s_var, q_mu, q_logvar):
        rate_denominator = s_var.unsqueeze(1).unsqueeze(1)
        var_rate = torch.exp(q_logvar) / rate_denominator
        q_s_kl = torch.log(var_rate) - var_rate - torch.pow(s_mu.unsqueeze(1).unsqueeze(1) - q_mu,
                                                                   2) / rate_denominator + 1
        q_s_kl = - 0.5 * q_s_kl
        q_s_kl = torch.sum(q_s_kl, dim=-1)
        kl_pred = torch.softmax(-q_s_kl, dim=-1)
        return kl_pred


    def get_kl(self, s_mu, s_var, n_mu, n_var):
        var_rate = n_var / s_var
        q_s_kl = torch.log(var_rate) - var_rate - torch.pow(s_mu - n_mu, 2) / s_var + 1
        q_s_kl = - 0.5 * q_s_kl
        return q_s_kl


    def forward(self, support_inputs, query_inputs, class_input_list, training=False):
        s_ids, s_type_ids, s_masks, s_meta_labels = support_inputs
        q_ids, q_type_ids, q_masks, q_meta_labels = query_inputs
        aspect_ids, aspect_type_ids, aspect_mask = class_input_list

        # support feature extracting
        s_shape = s_ids.shape # B, N, K, Tt
        bs = s_shape[0]
        s_neg_meta_labels = (1 - s_meta_labels).view(bs, self.N * self.K, self.N).transpose(1, 2)  # B, N, N * K
        s_neg_cnt = torch.sum(s_neg_meta_labels, dim=-1)

        eye_mask = torch.eye(self.N).to(self.device)

        s_ids = s_ids.view(bs * self.N * self.K, -1)
        s_type_ids = s_type_ids.view(bs * self.N * self.K, -1)
        s_masks = s_masks.view(bs * self.N * self.K, -1)
        s_text_hidden_state = self.bert_encoder(input_ids=s_ids,
                                     token_type_ids=s_type_ids,
                                     attention_mask=s_masks)[0]
        s_text_hidden_state = s_text_hidden_state.view(bs, self.N, self.K, s_shape[3], -1)  # B, N, K, Tt, d
        s_masks = s_masks.view(bs, self.N, self.K, -1)  # B, N, K, Tt
        s_A = self.f2(torch.tanh(self.f1(s_text_hidden_state)))
        s_A = s_A.masked_fill((1 - s_masks).unsqueeze(-1).type(torch.bool), -1e10)
        s_A = torch.softmax(s_A, dim=-2)
        s_O = s_text_hidden_state.transpose(-1, -2) @ s_A
        s_O = s_O.view(bs, self.N, self.K, -1) # B, N, K, d

        # aspect feature extracting
        as_shape = aspect_ids.shape
        aspect_ids = aspect_ids.view(bs * self.N, -1)
        aspect_type_ids = aspect_type_ids.view(bs * self.N, -1)
        aspect_mask = aspect_mask.view(bs * self.N, -1)
        as_hidden_state = self.bert_encoder(input_ids=aspect_ids,
                                      token_type_ids=aspect_type_ids,
                                      attention_mask=aspect_mask
                                      )[0]
        as_hidden_state = as_hidden_state.view(bs, self.N, as_shape[2], -1)
        as_A = torch.softmax(self.f2(torch.tanh(self.f1(as_hidden_state))), dim=-2)
        as_E = as_hidden_state.transpose(-1, -2) @ as_A
        as_E = as_E.view(bs, self.N, -1) # B, N, d

        # query feature extracting
        q_shape = q_ids.shape
        q_ids = q_ids.view(bs * self.N * self.Q, -1)
        q_type_ids = q_type_ids.view(bs * self.N * self.Q, -1)
        q_masks = q_masks.view(bs * self.N * self.Q, -1)
        q_text_hidden_state = self.bert_encoder(input_ids=q_ids,
                                     token_type_ids=q_type_ids,
                                     attention_mask=q_masks)[0]
        q_text_hidden_state = q_text_hidden_state.view(bs, self.N, self.Q, q_shape[3],
                                                       -1)  # B, N, Q, Tt, d
        q_masks = q_masks.view(bs, self.N, self.Q, -1)  # B, N, Q, Tt

        s_as_atten = s_text_hidden_state @ as_E.unsqueeze(1).unsqueeze(1).transpose(-1, -2)
        s_as_atten = s_as_atten.masked_fill((1 - s_masks).unsqueeze(-1).type(torch.bool), -1e10)
        s_as_atten = torch.softmax(s_as_atten, dim=-2)
        s_u = s_as_atten.transpose(-2, -1) @ s_text_hidden_state  # B, N, K, N, d

        s_intra_atten = s_text_hidden_state @ s_O.unsqueeze(2).transpose(-1, -2)
        s_intra_atten = s_intra_atten.masked_fill((1 - s_masks).unsqueeze(-1).type(torch.bool), -1e10)
        s_intra_atten = torch.softmax(s_intra_atten, dim=-2)
        s_v = s_intra_atten.transpose(-2, -1) @ s_text_hidden_state  # B, N, K, K, d
        s_v = torch.mean(s_v, dim=-2)  # B, N, K, d

        s_O = s_O.view(bs, self.N * self.K, -1).transpose(-1, -2).unsqueeze(1).unsqueeze(1)
        q_as_atten = q_text_hidden_state @ as_E.unsqueeze(1).unsqueeze(1).transpose(-1, -2)
        q_as_atten = q_as_atten.masked_fill((1 - q_masks).unsqueeze(-1).type(torch.bool), -1e10)
        q_as_atten = torch.softmax(q_as_atten, dim=-2)
        q_u = q_as_atten.transpose(-2, -1) @ q_text_hidden_state  # B, N, Q, N, d

        q_s_atten = q_text_hidden_state @ s_O  # B, N, Q, Tt, N * K
        q_s_atten = q_s_atten.masked_fill((1 - q_masks).unsqueeze(-1).type(torch.bool), -1e10)
        q_s_atten = torch.softmax(q_s_atten, dim=-2)
        q_v = q_s_atten.transpose(-2, -1) @ q_text_hidden_state  # B, N, Q, N * K, d
        q_v = q_v.view(bs, self.N, self.Q, self.N, self.K,
                                       -1)  # B, N, Q, N, K, d
        q_v = torch.mean(q_v, dim=-2)  # B, N, Q, N, d



        s_m = torch.cat([s_u, s_v.unsqueeze(-2).expand_as(s_u)], dim=-1)
        q_m = torch.cat([q_u, q_v], dim=-1)


        s_m_pos = self.get_pos_view(s_m, eye_mask)
        s_emb_pos = self.encoder(s_m_pos)
        s_mu_pos = self.mu(s_emb_pos)
        s_logvar_pos = self.logvar(s_emb_pos)

        s_emb = self.encoder(s_m)
        s_mu = self.mu(s_emb)
        s_logvar = self.logvar(s_emb)

        s_mu_neg = s_mu * (1 - s_meta_labels).unsqueeze(-1)
        s_mu_neg = s_mu_neg.view(bs, self.N * self.K, self.N, -1).transpose(1, 2)  # B, N, N * K, d_lat
        s_logvar_neg = s_logvar * (1 - s_meta_labels).unsqueeze(-1)
        s_logvar_neg = s_logvar_neg.view(bs, self.N * self.K, self.N, -1).transpose(1, 2)  # B, N, N * K, d_lat
        s_var_mean_neg, s_mu_mean_neg = self.get_mean_mu_and_var(s_logvar_neg, s_mu_neg, s_neg_cnt.unsqueeze(-1))
        s_var_mean_pos, s_mu_mean_pos = self.get_mean_mu_and_var(s_logvar_pos, s_mu_pos, self.K)  # B, N, d_lat

        q_emb = self.encoder(q_m)
        q_mu = self.mu(q_emb)
        q_logvar = self.logvar(q_emb)

        kl_pred = self.get_kl_pred(s_mu_mean_pos, s_var_mean_pos, q_mu, q_logvar)
        p_n_kl = self.get_kl(s_mu_mean_pos, s_var_mean_pos, s_mu_mean_neg, s_var_mean_neg).detach()
        threshold_pred = torch.sigmoid(self.threshold_MLP(p_n_kl).squeeze(-1))



        if training:
            kl_pred_comp = kl_pred.detach()
            threshold_pred_sub = kl_pred_comp - threshold_pred.unsqueeze(1).unsqueeze(1)
            threshold_pred_sub_sig = torch.sigmoid(threshold_pred_sub)
            pt = (1 - threshold_pred_sub_sig) * q_meta_labels + threshold_pred_sub_sig * (1 - q_meta_labels)
            alpha = 1 - 1 / self.N
            gamma = self.K
            focal_weight = (alpha * q_meta_labels + (1 - alpha) * (1 - q_meta_labels)) * torch.pow(pt, gamma)
            kl_threshold_bce_loss = F.binary_cross_entropy_with_logits(threshold_pred_sub, q_meta_labels,
                                                                       reduction='none') * focal_weight
            kl_threshold_bce_loss = torch.mean(kl_threshold_bce_loss)
            kl_pred_loss = self.bce_wo_sigmoid_loss(kl_pred, q_meta_labels)
            loss = kl_pred_loss + kl_threshold_bce_loss
            return loss
        return kl_pred, threshold_pred

