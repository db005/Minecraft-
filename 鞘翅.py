import torch as t
import matplotlib.pyplot as plt
import torch.utils.data as Data
# t.cuda.set_device(1)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = t.device('cuda')
def tick(alpha, vx, vy):
    lx = t.cos(alpha)
    ly = t.sin(alpha)
    speed_proj = vx
    vy += 0.08 * (-1.0 + lx * lx * 0.75)
    if vy < 0 and lx > 0:
        d = vy * -0.1 * lx * lx
        vx += d
        vy += d
    if ly > 0 and lx > 0:
        d = speed_proj * ly * 0.04
        vx -= d
        vy += d * 3.2
    if lx > 0:
        vx += (speed_proj - vx) * 0.1
    return vx * 0.9900000095367432, vy * 0.9800000190734863




# t.autograd.set_detect_anomaly(True)  
import torch.nn as nn

import os 


MaxModelX = 0
fo = open("MaxModelX", "r")
MaxModelX = fo.readline()
MaxModelX = int(0 if MaxModelX=="" else MaxModelX)
fo.close()
# Defining input size, hidden layer size, output size and batch size respectively
n1_in, n1_h1, n1_h2 , n1_out= 3, 20, 20, 2

actor = nn.Sequential(
   nn.Linear(n1_in, n1_h1),
   nn.ELU(),
   nn.Linear(n1_h1, n1_h2),
   nn.ELU(),
   nn.Linear(n1_h2, n1_out),
   nn.Tanh())

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.l = nn.Sequential(nn.Linear(n1_in, n1_h1),
        nn.ELU(),
        nn.Linear(n1_h1, n1_h2),
        nn.ELU(),
        nn.Linear(n1_h2, n1_out)) 
        self.t = nn.Tanh()
        self.s = nn.Sigmoid()
    
    def forward(self,x):
        o1 = self.l(x)
        # print(o1[:,0],o1[:,1])
        mean = self.t(o1[:,0])
        std = self.s(o1[:,1])
        return mean,std

actor = Actor()

if MaxModelX == 0:
    前缀 = ""
else:
    前缀 = str(MaxModelX)
print(前缀)
if os.path.exists(前缀+"actor.pkl"):
    actor.load_state_dict(t.load(前缀+"actor.pkl"))

n2_in, n2_h1, n2_h2 , n2_out = 5, 20, 20, 1
critic = nn.Sequential(
   nn.Linear(n2_in, n2_h1),
   nn.ELU(),
   nn.Linear(n2_h1, n2_h2),
   nn.ELU(),
   nn.Linear(n2_h2, n2_out),
   nn.Sigmoid())

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = critic
        self.distence = t.tensor(6000)
        self.distence.requires_grad = False
    def forward(self,x):
        o = self.l(x)
        return self.distence*o
critic = Critic()
if os.path.exists(前缀+"critic.pkl"):
    critic.load_state_dict(t.load(前缀+"critic.pkl"))

Lstm = nn.LSTM(4,10,3,dropout=0.5)
Linear = nn.Linear(10,1)
Sigmoid = nn.Sigmoid()

class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.Lstm = Lstm
        self.Linear = Linear
        self.Sigmoid = Sigmoid
        self.hidden = self.init_hidden()
        
    def forward(self, x):
        x = t.reshape(x,(x.shape[0],1,x.shape[1]))
        o2,self.hidden = self.Lstm(x,self.hidden)
        o2 = self.Linear(o2)
        o2 = self.Sigmoid(o2)
        return o2
    
    def init_hidden(self):
        return (t.zeros(3,1,10),
            t.zeros(3,1,10))

# Controller_ = Controller()

# if os.path.exists("controller.pkl"):
#     Controller_.load_state_dict(t.load("controller.pkl"))
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, x):
        mean,std = self.actor(x)
        output = self.critic(t.cat((mean.view(mean.shape[0],1),std.view(std.shape[0],1),x),1))
        return output

class MyLoss0(nn.Module):
    # 不要忘记继承Module
    def __init__(self):
        super(MyLoss0, self).__init__()

    def forward(self, output,Y):
        # 不要忘记返回scalar
        return t.sum(t.pow((output - Y), 2))

class MyLoss(nn.Module):
    # 不要忘记继承Module
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output):
        # 不要忘记返回scalar
        return t.sum(-1.0*output)
net = model()


# Construct the loss function
criterion1 = MyLoss0()
criterion2 = MyLoss()
# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer1 = t.optim.AdamW(critic.parameters(), lr = 0.01, weight_decay=0.001)
optimizer2 = t.optim.AdamW(net.parameters(), lr = 0.1, weight_decay=0.001)


sa = []
r = []
Hlist = []
IniInputlist = []
alphalist = []
Xlist = []
R = []
k=3
BestLine = 0
for i_episode in range(1000):
    del(Hlist)
    del(alphalist)
    del(Xlist)
    del(sa)
    del(r)
    del(IniInputlist)
    # del(net.hidden)
    # net.hidden = net.init_hidden()
    sa = []
    r = []
    Hlist = []
    alphalist = []
    Xlist = []
    IniInputlist = []
    ModelX = 0
    for i in range(20):
        # H = t.rand(1)*50+280
        H = t.tensor([300.0])
        H_ = H.tolist()[0]
        X = 0
        lastX = 0
        vx = t.tensor([0.0])
        vy = t.tensor([0.0])
        while H>0:
            IniInpu = t.cat((H,vx,vy),0)
            mean, std = actor(IniInpu.reshape(1,3))
            action = t.normal(mean,std)
            action = 3.1415/2*action

            IniInputlist.append(IniInpu.tolist())
            alpha = action
            
            # if  t.rand(1)>0.95:
            #     if t.rand(1)>0.5:
            #         alpha = alpha*(1.1)
            #     else:
            #         alpha = alpha*(-1.1)
            if t.isnan(alpha[0]) or alpha>3.1415/2 or alpha<-3.1415/2:
                if alpha >= 0:
                    alpha = t.tensor([3.1415/2])
                if alpha < 0:
                    alpha = t.tensor([-3.1415/2])

            vx, vy = tick(alpha,vx,vy)
            # print(alpha,end="")
            # print(H,alpha,vx,vy)
            # print(H)
            H = H + vy
            X = X + vx
            Hlist.append(H.tolist()[0])
            alphalist.append(alpha.tolist()[0])
            Xlist = Xlist+X.tolist()
            action = [mean.tolist()[0],std.tolist()[0]]
            action= action + IniInpu.tolist()
            # print(action)
            sa.append(action)
            r.append([lastX])
            lastX = X
        # IniInpu_ =  t.tensor(sa)
        # M = Controller_(IniInpu_)
        # M = int(M.tolist()[M.shape[0]-1][0][0])*30+1
        ModelX+=X.tolist()[0]
        print("行进",X.tolist()[0],"高度",H_,"最远",BestLine,"i",i)#,'epoch ',M)
        for oner in r:
            oner[0] = (X - oner[0])/4000
        # print(X)
        R.append(X.tolist()[0]/H_)
    sa_ = t.tensor(sa).to(device)
    r_ = t.tensor(r).to(device)
    IniInputlist_ = t.tensor(IniInputlist).to(device)
    # print(sa.shape,r.shape)
    # Gradient Descent

    # if BestLine<X:
    #     BestLine = X
    #     M=1000
    # else:
    #     M=50
    # if X + 100 > BestLine:
    #     M = 200
    
    plt.plot(Xlist,Hlist)
    plt.savefig("飞行曲线episode"+str(i_episode))
    plt.cla()
    plt.plot(Xlist,alphalist)
    plt.savefig("角度曲线episode"+str(i_episode))
    plt.cla()

    loss1 = t.tensor(0)
    loss2 = t.tensor(0)
    actor.cuda()
    net.cuda()    
    criticdata = Data.TensorDataset(sa_ , r_)
    netdata = Data.TensorDataset(IniInputlist_ , r_)
    criticloader = Data.DataLoader(
    dataset = criticdata,
    batch_size = 10,
    shuffle=True,
    num_workers = 0,  #采用两个进程来提取
)
    netloader = Data.DataLoader(
    dataset = netdata,
    batch_size = 10,
    shuffle=True,
    num_workers = 0,  #采用两个进程来提取
)
    ModelX=ModelX/20        
    if MaxModelX<ModelX:
        MaxModelX=ModelX
        fo = open("MaxModelX", "w")
        fo.writelines(str(int(MaxModelX)))
        print("写入"+str(int(MaxModelX))+"成功")
        fo.close()
    print("当前平均距离",ModelX,"最大平均距离",MaxModelX)
    for epoch in range(10):
        for step , (batch_x,batch_y) in enumerate(criticloader):
            y_pred = critic(batch_x)
            loss1 = criterion1(y_pred,batch_y)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            if epoch%2==0 and step%1000==0:
                print('i_episode: ',i_episode, 'epoch',epoch,'step',step,' loss1: ', loss1.tolist())
    for epoch in range(10):
        # epochloss2 = 0
        for step , (batch_x,batch_y) in enumerate(netloader):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = net(batch_x)
            # Compute and print loss
            loss2 = criterion2(y_pred)
            # epochloss2 += loss2
            # if epoch%5==0:
            # print('i_episode: ',i_episode, 'epoch: ', epoch,' loss1: ', loss1.tolist(),' loss2: ', loss2.tolist())
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer2.zero_grad()
            # perform a backward pass (backpropagation)
            loss2.backward()
            # nn.utils.clip_grad_value_(net.linear.weight, clip_value=1.1)
            
        # Update the parameters
        # optimizer2.step()
        # if -1*loss2/X*4000/X*4000>X*2+300:
        #     k*=1.5
        # elif -1*loss2/X*4000/X*4000<=(X-300):
        #     k/=2
        # print(k,' ',end="")
        # k = int(k)
        # if k<2:
        #     k=2    
        # if k>500:
            # k=500
        # print("训练中")
            if epoch%2==0 and step%1000==0:
                print('i_episode: ',i_episode, 'epoch',epoch,'step',step,' loss2: ', loss2.tolist())
    # if i_episode%20==0:
    # for name, parms in net.named_parameters():	
    #     print('-->name:', name, parms,'-->grad_requirs:',parms.requires_grad, \
    #         ' -->grad_value:',parms.grad)
    
    t.save(actor.state_dict(), str(int(ModelX))+"actor.pkl")
    t.save(critic.state_dict(), str(int(ModelX))+"critic.pkl")
    actor.cpu()
    net.cpu()
    
        # t.save(Controller_.state_dict(), "controller.pkl")


plt.plot(R)
plt.show()