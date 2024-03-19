# -*- coding:utf-8 -*-


import argparse
import os.path


def parsers():
    parser = argparse.ArgumentParser(description="Bert model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join("./data", "train.txt"))
    parser.add_argument("--dev_file", type=str, default=os.path.join("./data", "dev.txt"))
    parser.add_argument("--test_file", type=str, default=os.path.join("./data", "test.txt"))
    parser.add_argument("--classification", type=str, default=os.path.join("./data", "class.txt"))
    parser.add_argument("--bert_pred", type=str, default="./bert-base-chinese", help="bert 预训练模型")
    parser.add_argument("--select_model", type=bool, default=False, help="选择模型")
    parser.add_argument("--class_num", type=int, default=5, help="分类数")
    parser.add_argument("--max_len", type=int, default=150, help="句子的最大长度")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--learn_rate", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2, help="失活率")
    parser.add_argument("--filter_sizes", type=list, default=[2, 3, 4], help="TextCnn 的卷积核大小")
    parser.add_argument("--layer_sizes", type=list, default=[0.3,"填充",0.4,0.3,0.3,0.5,0.5,0.5,0.6,0.6,0.7,0.9,1], help="每一层比例")
    parser.add_argument("--num_filters", type=int, default=2, help="TextCnn 的卷积输出")
    parser.add_argument("--encode_layer", type=int, default=12, help="chinese bert 层数")
    parser.add_argument("--hidden_size", type=int, default=768, help="bert 层输出维度")
    parser.add_argument("--save_model_best", type=str, default=os.path.join("model", "best_model.pth"))
    parser.add_argument("--save_model_severity", type=str, default=os.path.join("model", "last_model.pth"))
    parser.add_argument("--save_model_possibility", type=str, default=os.path.join("model", "model_possi.pth"))
    parser.add_argument("--save_model_Risk_Level", type=str, default=os.path.join("model", "model_Risk_Level.pth"))
    parser.add_argument("--modle_severity", type=bool, default=True, help="是否选择严重性")
    parser.add_argument("--modle_possibility", type=bool, default=True, help="是否选择可能性")
    parser.add_argument("--modle_Risk_Level", type=bool, default=False, help="是否选择风险等级")

    parser.add_argument("--Types_Of_LOSS", type=bool, default= True, help="选择LOSS种类  -- True 是旧的   False 是新的")

    parser.add_argument("--hidden_dim", type=int, default=384, help="Bilstm隐藏层数量")
    parser.add_argument("--bilstm_layers", type=int, default=2, help="Bilstm隐藏层层数")
    parser.add_argument("--bidirectional", type=bool, default=True, help="是否为双向LSTM")

    parser.add_argument("--bilstm_or_textcnn", type=list, default=[0.8,0.2], help="最终结果影响")

    parser.add_argument("--patten", type=bool, default=True, help="选择什么结合方式")
    args = parser.parse_args()
    return args
