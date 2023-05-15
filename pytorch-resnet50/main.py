import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1、跳跃时，a[l]与a[l+2]维度一致
#残差块
class IdentityBlock(nn.Module):
    def __init__(self, f, filters, stage, block):
        super(IdentityBlock, self).__init__()
        # conv_name_base = 'res' + str(stage) + block + '_branch'
        # bn_name_base = 'bn' + str(stage) + block + '_branch'
        F1, F2, F3 = filters
        self.X_shortcut = nn.Identity()

        
        if stage == 2:
            flag = 256
        elif stage == 3:
            flag = 512
        elif stage == 4:
            flag = 1024
        elif stage == 5:
            flag = 2048


        self.conv2a = nn.Conv2d(in_channels=flag, out_channels=F1, kernel_size=1, stride=1, padding=0)
        self.bn2a = nn.BatchNorm2d(F1)
        self.relu2a = nn.ReLU()
        self.dropout2a = nn.Dropout(p=0.4)

        self.conv2b = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=f, stride=1, padding=(f-1)//2)
        self.bn2b = nn.BatchNorm2d(F2)
        self.relu2b = nn.ReLU()
        self.dropout2b = nn.Dropout(p=0.4)

        self.conv2c = nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1, stride=1, padding=0)
        self.bn2c = nn.BatchNorm2d(F3)
        self.relu = nn.ReLU()

    def forward(self, X):
        X_shortcut = self.X_shortcut(X)

        X = self.conv2a(X)
        X = self.bn2a(X)
        X = self.relu2a(X)
        X = self.dropout2a(X)

        X = self.conv2b(X)
        X = self.bn2b(X)
        X = self.relu2b(X)
        X = self.dropout2b(X)

        X = self.conv2c(X)
        X = self.bn2c(X)

        X += X_shortcut
        X = self.relu(X)

        return X
    
#2、跳跃时，a[l]与a[l+2]维度不一致，在小路上加上卷积层改变a[l]的维度
#残差块
class ConvolutionalBlock(nn.Module):
    def __init__(self, f, filters, stage, block, stride=2):
        super(ConvolutionalBlock, self).__init__()
        # conv_name_base = 'res' + str(stage) + block + '_branch'
        # bn_name_base = 'bn' + str(stage) + block + '_branch'
        F1, F2, F3 = filters
        self.X_shortcut = nn.Identity()

        if stage == 2:
            flag = 64
        elif stage == 3:
            flag = 256
        elif stage == 4:
            flag = 512
        elif stage == 5:
            flag = 1024

        self.conv2a = nn.Conv2d(in_channels=flag, out_channels=F1, kernel_size=1, stride=stride, padding=0)
        self.bn2a = nn.BatchNorm2d(F1)
        self.relu2a = nn.ReLU()
        self.dropout2a = nn.Dropout(p=0.4)

        self.conv2b = nn.Conv2d(in_channels=F1, out_channels=F2, kernel_size=f, stride=1, padding=(f-1)//2)
        self.bn2b = nn.BatchNorm2d(F2)
        self.relu2b = nn.ReLU()
        self.dropout2b = nn.Dropout(p=0.4)

        self.conv3 = nn.Conv2d(in_channels=F2, out_channels=F3, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(F3)
        self.conv_shortcut = nn.Conv2d(in_channels=flag, out_channels=F3, kernel_size=1, stride=stride, padding=0)
        self.bn_shortcut = nn.BatchNorm2d(F3)

    def forward(self, X):
        shortcut = X

        X = self.conv2a(X)
        X = self.bn2a(X)
        X = self.relu2a(X)
        X = self.dropout2a(X)

        X = self.conv2b(X)
        X = self.bn2b(X)
        X = self.relu2b(X)
        X = self.dropout2b(X)

        X = self.conv3(X)
        X = self.bn3(X)

        shortcut = self.conv_shortcut(shortcut)
        shortcut = self.bn_shortcut(shortcut)

        X += shortcut
        X = F.relu(X)

        return X
    
# 定义ResNet50模型
class ResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet50, self).__init__()
        self.padding = nn.ZeroPad2d((3, 3))
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = nn.Sequential(
            ConvolutionalBlock(f=3, filters=[64, 64, 256], stage=2, block='a', stride=1),
            IdentityBlock(3, [64, 64, 256], stage=2, block='b'),
            IdentityBlock(3, [64, 64, 256], stage=2, block='c')
        )
        self.layer2 = nn.Sequential(
            ConvolutionalBlock(f=3, filters=[128, 128, 512], stage=3, block='a', stride=2),
            IdentityBlock(3, [128, 128, 512], stage=3, block='b'),
            IdentityBlock(3, [128, 128, 512], stage=3, block='c'),
            IdentityBlock(3, [128, 128, 512], stage=3, block='d')
        )
        self.layer3 = nn.Sequential(
            ConvolutionalBlock(f=3, filters=[256, 256, 1024], stage=4, block='a', stride=2),
            IdentityBlock(3, [256, 256, 1024], stage=4, block='b'),
            IdentityBlock(3, [256, 256, 1024], stage=4, block='c'),
            IdentityBlock(3, [256, 256, 1024], stage=4, block='d'),
            IdentityBlock(3, [256, 256, 1024], stage=4, block='e'),
            IdentityBlock(3, [256, 256, 1024], stage=4, block='f')
        )
        self.layer4 = nn.Sequential(
            ConvolutionalBlock(f=3, filters=[512, 512, 2048], stage=5, block='a', stride=2),
            IdentityBlock(3, [512, 512, 2048], stage=5, block='b'),
            IdentityBlock(3, [512, 512, 2048], stage=5, block='c')
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, X):

        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.maxpool(X)
        
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.fc(X)
        X = F.softmax(X, dim=1)
        
        return X

model = ResNet50()

# ------------------------------------准备数据阶段---------------------------------------

# 加载数据集
train_data = np.load('datasets/train_merged_signal.npy')
train_label = np.load('datasets/train_labels.npy')

test_data = np.load('datasets/test_merged_signal.npy')
test_label = np.load('datasets/test_labels.npy')
print("test_label: ", test_label)

# 将标签中的-1转换为0
ys_train_orig = np.zeros((train_label.shape[0],), dtype=int)
for i in range((len(train_label))):
    if train_label[i] == -1:
        train_label[i] = 0
    ys_train_orig[i] = train_label[i][0].astype(int)

ys_test_orig = np.zeros((test_label.shape[0],), dtype=int)

for i in range((len(test_label))):
    if test_label[i] == -1:
        test_label[i] = 0
    ys_test_orig[i] = test_label[i][0].astype(int)
print("ys_test_orig: ", ys_test_orig)

# 将 data，label 换个名字
X_train, Y_train = train_data, ys_train_orig
X_test, Y_test = test_data, ys_test_orig
print("Y_test: ", Y_test)

# 将标签转化为 one-hot 编码
enc = OneHotEncoder(categories='auto')
Y_train = enc.fit_transform(Y_train.reshape(-1, 1)).toarray()
Y_test = enc.transform(Y_test.reshape(-1, 1)).toarray()

# 扩充 X 的维度
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# 转换数据为张量
X_train_tensor = torch.from_numpy(X_train).float()
Y_train_tensor = torch.from_numpy(Y_train).long()
X_test_tensor = torch.from_numpy(X_test).float()
Y_test_tensor = torch.from_numpy(Y_test).long()

print()
print("number of training examples = " + str(X_train_tensor.shape[0]))
print("number of test examples = " + str(X_test_tensor.shape[0]))
print("X_train_tensor shape: " + str(X_train_tensor.shape))
print("Y_train_tensor shape: " + str(Y_train_tensor.shape))
print("X_test_tensor shape: " + str(X_test_tensor.shape))
print("Y_test_tensor shape: " + str(Y_test_tensor.shape))

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# ------------------------------------训练和测试阶段---------------------------------------

#编译模型
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.03)

#最后一招
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.03)

criterion = nn.CrossEntropyLoss()

# 将模型和损失函数移动到GPU上（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion.to(device)

num_epochs = 250

# 训练
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print("outputs: ", outputs)
        print("torch.argmax(labels, dim=1)", torch.argmax(labels, dim=1))

        _, predicted = outputs.max(1)
        _, truths = labels.max(1)
        
        total += labels.size(0)

        correct += predicted.eq(truths).sum().item()

        # 每个 batch 打印 loss 和精度
        print(f"Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}, Accuracy: {100 * correct / total}%")

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1} loss: {epoch_loss}, Accuracy: {epoch_accuracy}%")

# 测试
with torch.no_grad():
    for inputs, labels in test_loader:

        inputs = inputs.to(device)  # 将输入数据移动到 CUDA 设备上
        labels = labels.to(device)  # 将标签数据移动到 CUDA 设备上

        outputs = model(inputs)
        test_loss = criterion(outputs, labels.float())

        _, predicted = outputs.max(1)
        _, truths = labels.max(1)
        
        test_accuracy = (predicted == truths).sum().item() / len(labels)

print("Loss =", test_loss.item())
print("Test Accuracy =", test_accuracy)