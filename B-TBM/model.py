# -*- coding:utf-8 -*-


from transformers import BertModel
import torch.nn as nn
from config import parsers
import torch
import torch.nn.functional as F





class BiLSTMModel(nn.Module):
    def __init__(self,hidden_dim, output_size,n_layers,bidirectional=True, drop_prob=0.5):
        super(BiLSTMModel, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)

        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)


    def forward(self, x, hidden):                     #x需要bert最后一层隐藏层
        batch_size = x.size(0)

        # 生成bert字向量
        #x = self.bert(x)[0]  # bert 字向量

        # lstm_out
        # x = x.float()
        lstm_out, (hidden_last, cn_last) = self.lstm(x, hidden)
        # print(lstm_out.shape)   #[32,100,768]
        #print(hidden_last.shape)   #[4, 32, 384]
        # print(cn_last.shape)    #[4, 32, 384]

        # 修改 双向的需要单独处理
        if self.bidirectional:
            # 正向最后一层，最后一个时刻
            hidden_last_L = hidden_last[-2]
            #print(hidden_last_L.shape)  #[32, 384]
            # 反向最后一层，最后一个时刻
            hidden_last_R = hidden_last[-1]
            #print("tttttttttttttt" + str(hidden_last_R.shape))  # [batch_size,隐藏层]
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
            #print("qqqqqqqqqqqq" + str(hidden_last_out.shape))
            # print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out = hidden_last[-1]  # [32, 384]
        #print(hidden_last_out.shape)    #[beach_size,hidd*2]
        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        # print(out.shape)    #[32,768]
        #out = self.fc(out)

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        number = 1
        if self.bidirectional:
            number = 2


        hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda()
                      )


        return hidden




class TextCnnModel(nn.Module):
    def __init__(self):
        super(TextCnnModel, self).__init__()
        self.num_filter_total = parsers().num_filters * len(parsers().filter_sizes)
        self.Weight = nn.Linear(self.num_filter_total, parsers().class_num, bias=False)
        self.bias = nn.Parameter(torch.ones([parsers().class_num]))
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, parsers().num_filters, kernel_size=(size, parsers().hidden_size)) for size in parsers().filter_sizes
        ])

    def forward(self, x):
        # x: [batch_size, 12, hidden]
        x = x.unsqueeze(1)  # [batch_size, channel=1, 12, hidden]

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            out = F.relu(conv(x))  # [batch_size, channel=2, 12-kernel_size[0]+1, 1]
            maxPool = nn.MaxPool2d(                                                #最大池化
                kernel_size=(parsers().encode_layer - parsers().filter_sizes[i] + 1, 1)
            )
            # maxPool: [batch_size, channel=2, 1, 1]
            pooled = maxPool(out).permute(0, 3, 2, 1)  # [batch_size, h=1, w=1, channel=2]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(parsers().filter_sizes))  # [batch_size, h=1, w=1, channel=2 * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filter_total])  # [batch_size, 6]
        #print("eeeeeeeeeeeeee"+str(h_pool_flat.shape))

        #output = self.Weight(h_pool_flat) + self.bias  # [batch_size, class_num]

        return h_pool_flat


class BertTextModel_encode_layer(nn.Module):
    def __init__(self):
        super(BertTextModel_encode_layer, self).__init__()
        self.bert = BertModel.from_pretrained(parsers().bert_pred)

        for param in self.bert.parameters():    #允许参数回传
            param.requires_grad = True

        self.linear = nn.Linear(parsers().hidden_size, parsers().class_num)
        self.Bilstm = BiLSTMModel(parsers().hidden_dim,parsers().class_num,parsers().bilstm_layers)
        self.textCnn = TextCnnModel()
        self.linear_sum = nn.Linear(774,parsers().class_num)
    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x[0], x[1], x[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True  # 确保 hidden_states 的输出有值
                            )
        h = self.Bilstm.init_hidden(parsers().batch_size)
        h = tuple([each.data for each in h])
        pred_bilstm = self.Bilstm(outputs[0],h)
        #print("pred_bilstm = " + str(pred_bilstm))
        #print("pred_bilstm.shape = " + str(pred_bilstm.shape))


        # 取每一层encode出来的向量
        hidden_states = outputs.hidden_states  # 13 * [batch_size, seq_len, hidden] 第一层是 embedding 层不需要
        #print("wwwwwwwwwwww"+str(len(hidden_states)))
        cls_embeddings = parsers().layer_sizes[0]*hidden_states[1][:, 0, :].unsqueeze(1)  # [batch_size, 1, hidden]
        #print("cls_embeddings" + str(cls_embeddings))
        #print("2*cls_embeddings" + str(2*cls_embeddings))
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textCnn的输入
        for i in range(2, 13):
            cls_embeddings = torch.cat((cls_embeddings, parsers().layer_sizes[i]*hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        #print("cls_embeddings  "+str(cls_embeddings.shape))
        # cls_embeddings: [batch_size, 12, hidden]
        pred_textcnn = self.textCnn(cls_embeddings)

        pred = self.linear_sum(torch.cat([pred_bilstm,pred_textcnn],dim = 1))
        #print("pred_textcnn = " + str(pred_textcnn))
        #print("pred_textcnn.shape = " + str(pred_textcnn.shape))

        #pred = parsers().bilstm_or_textcnn[0] * pred_bilstm + parsers().bilstm_or_textcnn[0] * pred_textcnn
        #print(pred)
        if parsers().patten:
            return pred
        else:
            return  pred_textcnn,pred_bilstm


class BertTextModel_last_layer(nn.Module):
    def __init__(self):
        super(BertTextModel_last_layer, self).__init__()
        self.bert = BertModel.from_pretrained(parsers().bert_pred)
        for param in self.bert.parameters():
            param.requires_grad = True

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=parsers().num_filters, kernel_size=(k, parsers().hidden_size),) for k in parsers().filter_sizes]
        )
        self.dropout = nn.Dropout(parsers().dropout)
        self.fc = nn.Linear(parsers().num_filters * len(parsers().filter_sizes), parsers().class_num)

    def conv_pool(self, x, conv):
        x = conv(x)  # shape [batch_size, out_channels, x.shape[1] - conv.kernel_size[0] + 1, 1]
        x = F.relu(x)
        x = x.squeeze(3)  # shape [batch_size, out_channels, x.shape[1] - conv.kernel_size[0] + 1]
        size = x.size(2)
        x = F.max_pool1d(x, size)   # shape[batch+size, out_channels, 1]
        x = x.squeeze(2)  # shape[batch+size, out_channels]
        return x

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x[0], x[1], x[2]
        hidden_out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               output_hidden_states=False)
        out = hidden_out.last_hidden_state.unsqueeze(1)   # shape [batch_size, 1, max_len, hidden_size]
        out = torch.cat([self.conv_pool(out, conv) for conv in self.convs], 1)  # shape  [batch_size, parsers().num_filters * len(parsers().filter_sizes]
        out = self.dropout(out)
        out = self.fc(out)
        return out


