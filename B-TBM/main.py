# -*- coding:utf-8 -*-


import time

from sklearn.svm._libsvm import predict
from tqdm import tqdm
from config import parsers
from utils import read_data, MyDataset
from torch.utils.data import DataLoader
from model import BertTextModel_encode_layer, BertTextModel_last_layer,BiLSTMModel
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import recall_score,f1_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
import torch.nn.functional as F

import numpy as np

proportion = 0.3

now_loss_severity = []
now_loss_possibility = []
now_loss_Risk_Level = []

now_acc_severity = []
now_acc_possibility = []


acc_sum_severity = []   #求ACC平均数
acc_average_severity = 0
acc_sum_possibility = []   #求ACC平均数
acc_average_possibility = 0
acc_sum_Risk_Level = []   #求ACC平均数
acc_average_Risk_Level = 0

e_severity = []
e_possibility = []


severity_loss = []
possibility_loss = []
Risk_Level_loss = []

severity_roc = []
possibility_roc = []

severity_true_roc = []
possibility_true_roc = []

"""
class loss_new():
    def __init__(self,pred, batch_con):
        self.m = 1

    def forward(self, pred, batch_con):
        sum_list = [0,0,0,0,0,0,0,0,0,0]
        late = 0
        loss_list = [ [1] * parsers().class_num] * parsers().batch_size
        loss_thro = torch.tensor(loss_list)
        soft = F.softmax(pred, 1)
        log_soft = F.log_softmax(pred, 1)

        print(soft)
        loss_softmax = loss_thro - soft
        print(loss_softmax)
        loss_log = torch.log(loss_softmax)
        print(loss_log)


        batch = list(batch_con)
        loss_log_list = list(loss_log)
        for i in range(parsers().batch_size):
            for j in range(parsers().class_num):
                if batch[i] != j:
                    sum_list[i] = sum_list[i] + loss_log_list[i][j]
            sum_list[i] = sum_list[i] / (parsers().class_num -1)
        for i in range(parsers().batch_size):
            late = sum_list[i] + late
        return torch.as_tensor(-late / parsers().batch_size, dtype=torch.float32)


    #new = torch.tensor(soft)
    def backward(self):
        pass

"""



def loss_new(pred, batch_con):
    lable_one_hot = F.one_hot(batch_con,5)
    soft = F.softmax(pred, 1)
    #soft_bilstm = F.softmax(pred_bilstm, 1)

    #soft =   (soft_textcnn *  parsers().bilstm_or_textcnn[1]) + (soft_bilstm * parsers().bilstm_or_textcnn[0])

    loss_BCE = torch.nn.BCELoss()
    lable_one_hot = lable_one_hot.to(torch.float)
    loss = loss_BCE(soft,lable_one_hot)*3
    return loss








def train(model, device, trainLoader, opt, epoch):

    model.train()
    loss_sum, count = 0, 0
    if (model == model_severity):
        print("严重性 第 " + str(epoch))
    else:
        print("可能性 第 " + str(epoch))
    for batch_index, batch_con in enumerate(trainLoader):
        batch_con = tuple(p.to(device) for p in batch_con)

        pred = model(batch_con)


        #true_e = F.one_hot(batch_con[-1], 5)
        mmu = F.softmax(pred, 1)
        #print("mmu" + str(batch_con[-1]))
        #pred_textcnn,pred_bilstm = model(batch_con)

        opt.zero_grad()
        #print("输出结果 =" + str(pred), str(batch_con[-1]))
        #print("pred  =  "+str(pred))
        #print("batch_con[-1]   =  " + str(batch_con[-1]))

        #print("batch_con[-1] " + str(batch_con[-1]))
        #print(F.softmax(pred, 1))
        #print(F.log_softmax(pred, 1))
        #loss_fn_2 = loss_new(pred, batch_con[-1])
        if parsers().Types_Of_LOSS:
            loss =loss_fn_1(pred, batch_con[-1])
        else:

            loss = loss_new(pred, batch_con[-1])

        #print("损失函数 = "+str(loss) +"训练次数 = "+ str(epoch))
        loss.backward()

        opt.step()

        loss_sum += loss.item()
        count += 1

        if count != 0:
            msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
            if batch_index % 100 == 0:
                print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
            if model ==model_severity:
                severity_loss.append(loss_sum / count)
                now_loss_severity.append(loss_sum / count)
            if model ==model_possibility:
                possibility_loss.append(loss_sum / count)
                now_loss_possibility.append(loss_sum / count)
            if model == model_Risk_Level:
                Risk_Level_loss.append(loss_sum / count)
                now_loss_Risk_Level.append(loss_sum / count)
            loss_sum, count = 0.0, 0


# 显示混淆矩阵
def plot_confuse_data(model,x_val, y_val):

    if(model == model_severity):
        classes = range(0, 5)
    if (model == model_possibility):
        classes = range(0, 4)




    # 将one-hot转化为label
    confusion = confusion_matrix(y_true=x_val, y_pred=y_val)
    #颜色风格为绿。。。。
    plt.imshow(confusion, cmap=plt.cm.Greens)
# ticks 坐标轴的坐标点
# label 坐标轴标签说明

    indices = range(len(confusion))
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix')
    # plt.rcParams两行是用于解决标签不能显示汉字的问题
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 显示数据
    for first_index in range(len(confusion)):    #第几行
        for second_index in range(len(confusion[first_index])):    #第几列
            plt.text(first_index, second_index, confusion[first_index][second_index])
# 显示
    plt.show()

def dev(model, device, devLoader,accuracy,epoch,acc_sum):
    global acc_min

    global e_severity
    global e_possibility

    epoch_roc = 2



    """
    if model == model_severity:
        model.load_state_dict(torch.load(args.save_model_severity))

    if model == model_possibility:
        model.load_state_dict(torch.load(args.save_model_possibility))
    if model == model_Risk_Level:
        model.load_state_dict(torch.load(args.save_Risk_Level))
    """


    model.eval()
    all_true, all_pred = [], []
    for batch_con in tqdm(devLoader):
        batch_con = tuple(p.to(device) for p in batch_con)
        soft = model(batch_con)





        #soft_textcnn = F.softmax(pred_textcnn, 1)
        #soft_bilstm = F.softmax(pred_bilstm, 1)
        #soft = (soft_textcnn * parsers().bilstm_or_textcnn[1]) + (soft_bilstm * parsers().bilstm_or_textcnn[0])

        pred = torch.argmax(soft, dim=1)

        pred_label = pred.cpu().detach().numpy().tolist()  # .detach()不带梯度信息
        true_label = batch_con[-1].cpu().numpy().tolist()


        # 计算roc曲线
        if model == model_severity and epoch ==  epoch_roc:
            sous = F.softmax(soft, 1)
            sous_list = sous.tolist()

            batch_list = batch_con[-1].tolist()

            for i in range(4):
                severity_true_roc.append(batch_list[i])
                severity_roc.append(sous_list[i])

        if model == model_possibility and epoch ==  epoch_roc:
            sous_possibility = F.softmax(soft, 1)
            sous_list_possibility = sous_possibility.tolist()

            batch_list_possibility = batch_con[-1].tolist()

            for i in range(4):
                possibility_true_roc.append(batch_list_possibility[i])
                possibility_roc.append(sous_list_possibility[i])

        if model == model_severity:
            pred_read = F.softmax(soft, 1)

            #print("pred_read  " + str(pred_read))
            #print("batch_con[-1]  " + str(batch_con[-1]))
            for i in range (4):

                #计算loss改进图像
                e_severity.append(np.append(pred_read.detach().cpu().numpy()[i],batch_con[-1].detach().cpu().numpy()[i]))
        if model == model_possibility:
            pred_read = F.softmax(soft, 1)
            #print("pred_read  " + str(pred_read))
            #print("batch_con[-1]  " + str(batch_con[-1]))
            for i in range(4):
                # 计算loss改进图像
                e_possibility.append(np.append(pred_read.detach().cpu().numpy()[i], batch_con[-1].detach().cpu().numpy()[i]))
        #print("lll "+str(e))


        #print("pred   " + str(pred))

        #出现错误信息 Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead. 加上.detach()解决  问题是提示带有梯度信息 不推荐转换


        #print("pred_label   true_label  "+str(pred_label)+str(true_label) )
        all_true.extend(true_label)
        all_pred.extend(pred_label)



    """
    
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-calss ROC')
    plt.legend(loc="lower right")
    plt.show()
    """

    acc = accuracy_score(all_true, all_pred)

    # 计算roc曲线
    if epoch == epoch_roc:

        if model == model_severity:
            ##严重性画混淆矩阵以及ROC曲线
            severity_true_array = np.array(severity_true_roc)  # 列表转化程array
            severity_array = np.array(severity_roc)
            classes = max(severity_true_array) + 1  ##类别数为最大数加1
            one_hot_label = np.zeros(shape=(severity_true_array.shape[0], classes))  ##生成全0矩阵
            one_hot_label[np.arange(0, severity_true_array.shape[0]), severity_true_array] = 1  ##相应标签位置置1

            # 清空

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            fpr["micro"], tpr["micro"], _ = roc_curve(one_hot_label.ravel(), severity_array.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


            e = pd.DataFrame(zip(fpr["micro"],tpr["micro"]))
            print(e)
            with pd.ExcelWriter('D:/BERT_BILSTM_TEXTCNN.xlsx', mode='a', engine="openpyxl") as writer:
                e.to_excel(writer, sheet_name='severity', index=True)



        if model == model_possibility:
            possibility_true_array = np.array(possibility_true_roc)  # 列表转化程array
            possibility_array = np.array(possibility_roc)
            classes = max(possibility_true_array) + 1  ##类别数为最大数加1
            one_hot_label_possibility = np.zeros(shape=(possibility_true_array.shape[0], classes))  ##生成全0矩阵
            one_hot_label_possibility[np.arange(0, possibility_true_array.shape[0]), possibility_true_array] = 1  ##相应标签位置置1

            # 清空

            fpr_possibility = dict()
            tpr_possibility = dict()
            roc_auc = dict()
            fpr_possibility["micro"], tpr_possibility["micro"], _ = roc_curve(one_hot_label_possibility.ravel(), possibility_array.ravel())
            roc_auc["micro"] = auc(fpr_possibility["micro"], tpr_possibility["micro"])

            q = pd.DataFrame(zip(fpr_possibility["micro"] ,tpr_possibility["micro"]))
            print(q)
            with pd.ExcelWriter('D:/BERT_BILSTM_TEXTCNN.xlsx', mode='a', engine="openpyxl") as writer:
                q.to_excel(writer, sheet_name='possibility', index=False)

        if model == model_severity:
            plot_confuse_data(model_severity, all_pred, all_true)  ##画混淆矩阵
        if model == model_possibility:
            plot_confuse_data(model_possibility, all_pred, all_true)  # 画混淆矩阵


    if model == model_severity:
        now_acc_severity.append(acc)

    if model == model_possibility:
        now_acc_possibility.append(acc)






    recall_0 = recall_score(all_true, all_pred, labels=[0], average='macro')
    recall_1 = recall_score(all_true, all_pred, labels=[1], average='macro')
    recall_2 = recall_score(all_true, all_pred, labels=[2], average='macro')
    recall_3 = recall_score(all_true, all_pred, labels=[3], average='macro')
    recall_4 = recall_score(all_true, all_pred, labels=[4], average='macro')

    print("标签0的recall  " + str(recall_0))
    print("标签1的recall  " + str(recall_1))
    print("标签2的recall  " + str(recall_2))
    print("标签3的recall  " + str(recall_3))
    print("标签4的recall  " + str(recall_4))

    f1_0 = f1_score(all_true, all_pred, labels=[0], average='macro')
    f1_1 = f1_score(all_true, all_pred, labels=[1], average='macro')
    f1_2 = f1_score(all_true, all_pred, labels=[2], average='macro')
    f1_3 = f1_score(all_true, all_pred, labels=[3], average='macro')
    f1_4 = f1_score(all_true, all_pred, labels=[4], average='macro')

    print("标签0的f1  " + str(f1_0))
    print("标签1的f1  " + str(f1_1))
    print("标签2的f1  " + str(f1_2))
    print("标签3的f1  " + str(f1_3))
    print("标签4的f1  " + str(f1_4))
    if (epoch > 50 and epoch <60):
        acc_sum.append(acc)

    #recall = recall_score(all_true, all_pred,average='weighted')  # acerage f1分数

    print(f"准确率:{acc:.4f}")

    #print(recall)
    #print(metrics.f1_score(all_true, all_pred,average='weighted'))
    #print(classification_report(all_true, all_pred))
        #accuracy = acc


    return acc
"""
    if acc > accuracy:
        print(f"以保存最佳模型")
        return acc
    else:
        return accuracy
"""

def data(proportion_test):
    train_severity_text = []
    train_severity_labels = []
    text_severity_text = []
    text_severity_labels = []

    train_possibility_labels = []
    text_possibility_labels = []

    train_Risk_Level_labels = []
    text_Risk_Level_labels = []

    with open('D:/辽阳石化脱硫及硫磺.csv','r') as f:
        reader = csv.reader(f)
        reader_now = list(reader)
        train_1, text_1, train_2, text_2 = train_test_split(reader_now, reader_now, test_size=proportion_test, random_state=0)
        for i in train_1:
            if i[1] != '严重性' or i[2] != '可能性':
                train_severity_text.append(i[0])
                train_severity_labels.append((int)(i[1]) - 1)
                train_possibility_labels.append(5 - (int)(i[2]))
                train_Risk_Level_labels.append(i[3])
        for j in text_1:
            if j[1] != '严重性' or i[2] != '可能性':
                text_severity_text.append(j[0])
                text_severity_labels.append((int)(j[1]) - 1)
                text_possibility_labels.append(5 - (int)(j[2]))
                text_Risk_Level_labels.append(j[3])
    print( "训练集长度 = " + str(len(train_severity_text)),str(len(train_severity_labels)))
    print("验证集长度 = " + str(len(text_severity_text)),str(len(text_severity_labels)))
    return train_severity_text, train_severity_labels,train_possibility_labels,train_Risk_Level_labels, text_severity_text, text_severity_labels,text_possibility_labels,text_Risk_Level_labels
    # print(sentences_text,labels_text)

def drawing(loss_m,B = [],m = False):
    x_e = [0]
    x_e[0] = 0
    x_f = [0]
    x_f[0] = 0
    for i in range(1, len(loss_m)):
        x_e.append(x_e[i - 1] + 1)

    plt.plot(x_e, loss_m)
    if m :
        for i in range(1, len(B)):
            x_f.append(x_f[i - 1] + 1)

        plt.plot(x_f, B)


    plt.show()



if __name__ == "__main__":
    """
    a = torch.tensor([[ 1.6844, -0.7355,  1.0149, -1.4294,  1.0342],
        [ 1.6152, -0.5980,  0.9453, -1.2410,  1.0546],
        [ 1.6003, -0.5140,  0.9640, -1.2454,  1.0176],
        [ 1.6278, -0.6312,  0.9922, -1.1767,  1.0692]])
    b = torch.tensor([3,0,2,0])
    

    print(loss_new(a,b))
    """

    #loss求和
    sumloss_severity = 0
    sumloss_possibility = 0
    sumloss_Risk_Level = 0

    #平均loss
    severityloss = []
    possibilityloss = []
    Risk_Levelloss = []


    train_text = []
    train_severity_labels = []
    text_text = []
    text_severity_labels = []

    train_possibility_labels = []

    text_possibility_labels = []
    train_Risk_Level_labels = []
    text_Risk_Level_labels = []

    accuracy_severity = float(0)
    accuracy_possibility = float(0)
    accuracy_Risk_Level = float(0)
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_text, train_severity_labels,train_possibility_labels,train_Risk_Level_labels, text_text, text_severity_labels,text_possibility_labels,text_Risk_Level_labels = data(proportion)

    loss_fn_1 = CrossEntropyLoss()

    #train_text, train_label = read_data(args.train_file)
    #dev_text, dev_label = read_data(args.dev_file)

    #for i in range(len(dev_label)):
    #    dev_label[i] = dev_label[i] - 1
    # 数据处理
    #严重性
    trainData_severity = MyDataset( train_text, train_severity_labels, with_labels=True)
    trainLoader_severity = DataLoader(trainData_severity, batch_size=args.batch_size, shuffle=True)

    devData_severity = MyDataset(text_text, text_severity_labels, with_labels=True)
    devLoader_severity = DataLoader( devData_severity, batch_size=args.batch_size, shuffle=True)

    #可能性
    trainData_possibility = MyDataset(train_text, train_possibility_labels, with_labels=True)
    trainLoader_possibility = DataLoader(trainData_possibility, batch_size=args.batch_size, shuffle=True)

    devData_possibility = MyDataset(text_text, text_possibility_labels, with_labels=True)
    devLoader_possibility = DataLoader(devData_possibility, batch_size=args.batch_size, shuffle=True)

    #风险等级
    trainData_Risk_Level = MyDataset(train_text, train_Risk_Level_labels, with_labels=True)
    trainLoader_Risk_Level = DataLoader(trainData_Risk_Level, batch_size=args.batch_size, shuffle=True)

    devData_Risk_Level = MyDataset(text_text, text_Risk_Level_labels, with_labels=True)
    devLoader_Risk_Level = DataLoader(devData_Risk_Level, batch_size=args.batch_size, shuffle=True)

    # 选择模型
    if args.select_model:
        model_severity = BertTextModel_last_layer().to(device)
        model_possibility = BertTextModel_last_layer().to(device)
        model_Risk_Level = BertTextModel_last_layer().to(device)
    else:

        model_severity = BertTextModel_encode_layer().to(device)
        model_possibility = BertTextModel_encode_layer().to(device)
        model_Risk_Level = BertTextModel_encode_layer().to(device)
     #   model = BertTextModel_encode_layer().to(device)

    #设置学习跌带
    opt_severity = AdamW(model_severity.parameters(), lr=args.learn_rate, weight_decay=1e-2)
    opt_possibility = AdamW(model_possibility.parameters(), lr=args.learn_rate, weight_decay=1e-2)
    opt_Risk_Level = AdamW(model_Risk_Level.parameters(), lr=args.learn_rate, weight_decay=1e-2)



    acc_min = float("-inf")
    for epoch in range(args.epochs):
        if args.modle_severity:
            train(model_severity, device, trainLoader_severity, opt_severity, epoch)
            for i in range(len(now_loss_severity)):
                sumloss_severity = now_loss_severity[i] + sumloss_severity

            severityloss.append(sumloss_severity/len(now_loss_severity))
            sumloss_severity = 0
            now_loss_severity = []
            #torch.save(model_severity.state_dict(), args.save_model_severity)
            accuracy_severity = dev(model_severity, device, devLoader_severity, accuracy_severity, epoch,
                                    acc_sum_severity)


        if args.modle_possibility:
            train(model_possibility, device, trainLoader_possibility, opt_possibility, epoch)
            #torch.save(model_possibility.state_dict(), args.save_model_possibility)
            for i in range(len(now_loss_possibility)):
                sumloss_possibility = now_loss_possibility[i] + sumloss_possibility

            possibilityloss.append(sumloss_possibility/len(now_loss_possibility))
            sumloss_possibility = 0
            now_loss_possibility = []
            accuracy_possibility = dev(model_possibility, device, devLoader_possibility, accuracy_possibility, epoch,
                                            acc_sum_possibility)
        if args.modle_Risk_Level:
            train(model_Risk_Level, device, trainLoader_Risk_Level, opt_Risk_Level, epoch)
            #torch.save(model_Risk_Level.state_dict(), args.save_model_Risk_Level)

            for i in range(len(now_loss_Risk_Level)):
                sumloss_Risk_Level = now_loss_Risk_Level[i] + sumloss_Risk_Level

            Risk_Levelloss.append(sumloss_Risk_Level/len(now_loss_Risk_Level))
            sumloss_Risk_Level = 0
            now_loss_Risk_Level = []

            accuracy_Risk_Level = dev(model_Risk_Level, device, devLoader_Risk_Level, accuracy_Risk_Level, epoch,
                                       acc_sum_Risk_Level)

    a = pd.DataFrame(e_severity)
    with pd.ExcelWriter('D:/辽阳石化加氢和加氢裂化-最终excel.xlsx', mode='a', engine="openpyxl") as writer:
        a.to_excel(writer, sheet_name='severity',index=False)
    a = pd.DataFrame(e_possibility)
    with pd.ExcelWriter('D:/辽阳石化加氢和加氢裂化-最终excel.xlsx', mode='a', engine="openpyxl") as writer:
        a.to_excel(writer,sheet_name='possibility', index=False)
    if args.modle_severity:
        for m in range(len(acc_sum_severity)):
            acc_average_severity += acc_sum_severity[m]
        print("严重性平均acc = " + str(acc_average_severity / len(acc_sum_severity)))
        #drawing(severity_loss)
        drawing(severityloss)
        drawing(now_acc_severity)
        model_severity.eval()
        torch.save(model_severity.state_dict(), args.save_model_severity)

    if args.modle_possibility:
        for m in range(len(acc_sum_possibility)):
            acc_average_possibility += acc_sum_possibility[m]
        print("可能性平均acc = " + str(acc_average_possibility / len(acc_sum_possibility)))
        #drawing(possibility_loss)
        drawing(possibilityloss)
        drawing(now_acc_possibility)

        model_possibility.eval()
        torch.save(model_possibility.state_dict(), args.save_model_possibility)

    if args.modle_Risk_Level:
        for m in range(len(acc_sum_Risk_Level)):
            acc_average_Risk_Level += acc_sum_Risk_Level[m]
        print("可能性平均acc = " + str( acc_average_Risk_Level / len(acc_sum_Risk_Level)))
        #drawing( Risk_Level_loss)
        drawing(Risk_Levelloss)
        model_Risk_Level.eval()
        torch.save(model_Risk_Level.state_dict(), args.save_model_Risk_Level)












    end = time.time()
    print(f"运行时间：{(end-start)/60%60:.4f} min")
