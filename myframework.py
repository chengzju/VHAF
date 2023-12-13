import dataloader
from transformers import BertTokenizer, AdamW
import torch
import numpy as np
from sklearn import metrics
from collections import OrderedDict
from future.utils import iteritems
from tqdm import tqdm
from utils import make_argmax_with_mask
import model

def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)

def convert_weights(state_dict):
    tmp_weights = OrderedDict()
    for name, params in iteritems(state_dict):
        tmp_weights[name.replace('module.', '')] = params
    return tmp_weights

def load_model(model, model_path):
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        model.load_state_dict(convert_weights(torch.load(model_path)))
    return model

class MyFramework(object):
    def __init__(self, args, B, N, K, Q, model_name, dataset, training_epoch, early_stop=True, patience=3,
                 max_len=None, shuffle=True, device=None, log_file=None):
        self.args = args
        self.B = B
        self.N = N
        self.K = K
        self.Q = Q
        self.epoch = training_epoch
        self.dataset = dataset
        self.patience = patience
        self.early_stop = early_stop
        self.device = device
        self.log_file = log_file
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        self.dataloader = dataloader.MyJSONFileDataLoader(args, dataset=dataset, tokenizer=self.tokenizer,
                                                              max_len=max_len, shuffle=shuffle)
        self.model = model.VHAF(args, device, N, K, Q).to(self.device)
        self.save_path = model_name


    def train(self):
        optimizer = AdamW(self.model.parameters(), lr=self.args.lr, eps=self.args.adam_epsilon)
        best_AUC = 0.0
        best_macro_f1 = 0.0
        eval_epoch = 0
        epoch = 1
        patience = self.patience

        while epoch <= self.epoch:
            print('epoch {}: '.format(epoch))
            loss_train = self.train_one_epoch(tasks=self.args.train_task_num, optimizer=optimizer)
            AUC_eval, macro_f1_eval = self.eval(tasks=self.args.eval_task_num)
            log_str = 'ep:{}. loss:{:.4f}. AUC,f1: {:.4f}\t{:.4f}'.format(epoch, loss_train, AUC_eval, macro_f1_eval)
            print(log_str)
            with open(self.log_file, 'a') as w:
                w.write('\n' + log_str)
            if AUC_eval > best_AUC:
                print('saving...')
                save_model(self.model, self.save_path + '/BEST_checkpoint.pt')
                best_AUC = AUC_eval
                best_macro_f1 = macro_f1_eval
                eval_epoch = epoch
                patience = self.patience
            elif self.early_stop:
                patience -= 1
            else:
                patience = self.patience
            if patience == 0:
                break
            print('patience now: {}'.format(patience))
            epoch += 1
        print('best macro f1: {} and AUC: {} at epoch {} during evaluating'.format(best_macro_f1, best_AUC, eval_epoch))
        return

    def train_one_epoch(self, tasks, optimizer):
        self.model.train()
        loss_train = 0.0
        for step in tqdm(range(int(tasks / self.B))):
            optimizer.zero_grad()
            support_set, query_set = self.dataloader.next_batch(B=self.B, N=self.N, K=self.K,
                                                                Q=self.Q, phrase='train')
            s_ids = support_set['text_ids'].to(self.device)
            s_type_ids = support_set['text_type_ids'].to(self.device)
            s_masks = support_set['text_masks'].to(self.device)
            s_meta_label = support_set['meta_label'].to(self.device)
            q_ids = query_set['text_ids'].to(self.device)
            q_type_ids = query_set['text_type_ids'].to(self.device)
            q_masks = query_set['text_masks'].to(self.device)
            q_meta_label = query_set['meta_label'].to(self.device)  # B, N, Q, N
            s_input_list = [s_ids, s_type_ids, s_masks, s_meta_label]
            q_input_list = [q_ids, q_type_ids, q_masks, q_meta_label]
            aspect_ids = support_set['aspect_ids'].to(self.device)
            aspect_type_ids = support_set['aspect_type_ids'].to(self.device)
            aspect_mask = support_set['aspect_mask'].to(self.device)
            class_input_list = [aspect_ids, aspect_type_ids, aspect_mask]
            loss = self.model(s_input_list, q_input_list, class_input_list, training=True)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_train /= int(tasks / self.B)
        return loss_train

    def eval(self, tasks):
        self.model.eval()
        AUC_eval = 0.
        macro_f1_eval = 0.
        for step in tqdm(range(int(tasks / self.B))):
            support_set, query_set = self.dataloader.next_batch(B=self.B, N=self.N, K=self.K, Q=self.Q, phrase='eval')
            s_ids = support_set['text_ids'].to(self.device)
            s_type_ids = support_set['text_type_ids'].to(self.device)
            s_masks = support_set['text_masks'].to(self.device)
            s_meta_label = support_set['meta_label'].to(self.device)
            q_ids = query_set['text_ids'].to(self.device)
            q_type_ids = query_set['text_type_ids'].to(self.device)
            q_masks = query_set['text_masks'].to(self.device)
            q_meta_label = query_set['meta_label'].to(self.device)  # B, N, Q, N
            s_input_list = [s_ids, s_type_ids, s_masks, s_meta_label]
            q_input_list = [q_ids, q_type_ids, q_masks, q_meta_label]
            aspect_ids = support_set['aspect_ids'].to(self.device)
            aspect_type_ids = support_set['aspect_type_ids'].to(self.device)
            aspect_mask = support_set['aspect_mask'].to(self.device)
            class_input_list = [aspect_ids, aspect_type_ids, aspect_mask]
            pred, threshold_mask = self.model(s_input_list, q_input_list, class_input_list)

            auc_sklearn = metrics.roc_auc_score(np.array(q_meta_label.cpu().numpy()).reshape((-1, self.N)),
                                                pred.detach().cpu().numpy().reshape((-1, self.N)),
                                                multi_class='ovo')
            threshold_pred_with_mask = make_argmax_with_mask(pred, threshold_mask)
            f1_score_with_mask = metrics.f1_score(q_meta_label.cpu().numpy().reshape((-1, self.N)),
                                        threshold_pred_with_mask.cpu().numpy().reshape((-1, self.N)),
                                        average='macro')
            AUC_eval += auc_sklearn
            macro_f1_eval += f1_score_with_mask
        r = int(tasks / self.B)
        AUC_eval /= r
        macro_f1_eval /= r
        return AUC_eval, macro_f1_eval

    def test(self, tasks):
        print('testing...')
        load_model(self.model, self.save_path + '/BEST_checkpoint.pt')
        self.model.eval()
        AUC_test = 0.
        macro_f1_test = 0.
        with torch.no_grad():
            for step in tqdm(range(int(tasks / self.B))):
                support_set, query_set = self.dataloader.next_batch(B=self.B, N=self.N, K=self.K, Q=self.Q,
                                                                    phrase='test')
                s_ids = support_set['text_ids'].to(self.device)
                s_type_ids = support_set['text_type_ids'].to(self.device)
                s_masks = support_set['text_masks'].to(self.device)
                s_meta_label = support_set['meta_label'].to(self.device)
                q_ids = query_set['text_ids'].to(self.device)
                q_type_ids = query_set['text_type_ids'].to(self.device)
                q_masks = query_set['text_masks'].to(self.device)
                q_meta_label = query_set['meta_label'].to(self.device)  # B, N, Q, N
                s_input_list = [s_ids, s_type_ids, s_masks, s_meta_label]
                q_input_list = [q_ids, q_type_ids, q_masks, q_meta_label]
                aspect_ids = support_set['aspect_ids'].to(self.device)
                aspect_type_ids = support_set['aspect_type_ids'].to(self.device)
                aspect_mask = support_set['aspect_mask'].to(self.device)
                class_input_list = [aspect_ids, aspect_type_ids, aspect_mask]
                pred, threshold_mask = self.model(s_input_list, q_input_list, class_input_list)

                auc_sklearn = metrics.roc_auc_score(np.array(q_meta_label.cpu().numpy()).reshape((-1, self.N)),
                                                    pred.detach().cpu().numpy().reshape((-1, self.N)),
                                                    multi_class='ovo')
                threshold_pred_with_mask = make_argmax_with_mask(pred, threshold_mask)
                f1_score_with_mask = metrics.f1_score(q_meta_label.cpu().numpy().reshape((-1, self.N)),
                                                      threshold_pred_with_mask.cpu().numpy().reshape((-1, self.N)),
                                                      average='macro')
                AUC_test += auc_sklearn
                macro_f1_test += f1_score_with_mask
        r = int(tasks / self.B)
        AUC_test /= r
        macro_f1_test /= r

        log_str = '\ntest, AUC, f1: \t{:.4f}\t{:.4f}'.format(AUC_test, macro_f1_test)
        print(log_str)
        with open(self.log_file, 'a') as w:
            w.write(log_str)
        return
