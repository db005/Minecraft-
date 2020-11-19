import torch as t

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

# Defining input size, hidden layer size, output size and batch size respectively
n1_in, n1_h1, n1_h2 , n1_out= 3, 10, 10, 1

actor = nn.Sequential(nn.Linear(n1_in, n1_h1),
   nn.ReLU(),
   nn.Linear(n1_h1, n1_h2),
   nn.ReLU(),
   nn.Linear(n1_h1, n1_h2),
   nn.ReLU(),
   nn.Linear(n1_h1, n1_h2),
   nn.ReLU(),
   nn.Linear(n1_h2, n1_out))
if os.path.exists("actor.pkl"):
    actor.load_state_dict(t.load("actor.pkl"))

n2_in, n2_h1, n2_h2 , n2_out = 4, 10, 10, 1
critic = nn.Sequential(nn.Linear(n2_in, n2_h1),
   nn.ReLU(),
   nn.Linear(n2_h1, n2_h2),
   nn.ReLU(),
   nn.Linear(n2_h1, n2_h2),
   nn.ReLU(),
   nn.Linear(n2_h1, n2_h2),
   nn.ReLU(),
   nn.Linear(n2_h2, n2_out))

if os.path.exists("critic.pkl"):
    critic.load_state_dict(t.load("critic.pkl"))

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

if os.path.exists("controller.pkl"):
    Controller_.load_state_dict(t.load("controller.pkl"))
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def forward(self, x):
        output = self.actor(x)
        output = self.critic(t.cat((output,x),1))
        return output

class MyLoss0(nn.Module):
    # 不要忘记继承Module
    def __init__(self):
        super(MyLoss0, self).__init__()

    def forward(self, output,Y,X):
        # 不要忘记返回scalar
        return X.tolist()[0]/4000*X.tolist()[0]/4000*t.mean(t.pow((output - Y), 2))

class MyLoss(nn.Module):
    # 不要忘记继承Module
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output,X):
        # 不要忘记返回scalar
        return X.tolist()[0]/4000*X.tolist()[0]/4000*t.mean(-output)
net = model()


# Construct the loss function
criterion1 = MyLoss0()
criterion2 = MyLoss()
# Construct the optimizer (Stochastic Gradient Descent in this case)
optimizer1 = t.optim.Adam(critic.parameters(), lr = 0.01, weight_decay=0.01)
optimizer2 = t.optim.Adam(net.parameters(), lr = 0.001, weight_decay=0.1)

sa = []
r = []
Hlist = []
IniInputlist = []
alphalist = []
Xlist = []
R = []
k=8
BestLine = 0
for i_episode in range(100):
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
    H = t.rand(1)*50+280
    H_ = (H).tolist()[0]
    IniInputlist = []
    X = 0
    lastX = 0
    vx = t.tensor([0.0])
    vy = t.tensor([0.0])
    alphalist = []
    Xlist = []
    while H>0:
        IniInpu = t.cat((H,vx,vy),0)
        action = actor(IniInpu)

        IniInputlist.append(IniInpu.tolist())
        alpha = action
        
        if alpha>3.1415/2 or alpha<-3.1415/2 or t.isnan(alpha[0]) or t.rand(1)>0.95:
            alpha = t.tensor([-0.7])+t.rand(1)*(1.4)
        vx, vy = tick(alpha,vx,vy)
        # print(alpha,end="")
        # print(H,alpha,vx,vy)
        # print(H)
        H = H + vy
        X = X + vx
        Hlist.append(H.tolist()[0])
        alphalist.append(alpha.tolist()[0])
        Xlist = Xlist+X.tolist()
        action = alpha.tolist()
        action= action + IniInpu.tolist()
        # print(action)
        sa.append(action)
        r.append([lastX])
        lastX = X
    # IniInpu_ =  t.tensor(sa)
    # M = Controller_(IniInpu_)
    # M = int(M.tolist()[M.shape[0]-1][0][0])*30+1
    
    print("行进",X.tolist()[0],"高度",H_,"最远",BestLine)#,'epoch ',M)
    for oner in r:
        oner[0] = X - oner[0]
    # print(X)
    R.append(X.tolist()[0]/H_)
    sa = t.tensor(sa)
    r = t.tensor(r)
    # print(sa.shape,r.shape)
    # Gradient Descent

    if BestLine<X:
        BestLine = X
        M=10
    else:
        M=0
    if X + 300 > BestLine:
        M = 10
    loss1 = t.tensor(0)
    loss2 = t.tensor(0)
    for epoch in range(M):

        
        for i in range(k):
            
            y_pred = critic(sa)
            loss1 = criterion1(y_pred,r,X)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
        
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = net(t.tensor(IniInputlist))

        # Compute and print loss
        loss2 = criterion2(y_pred,X)
        # if epoch%5==0:
        # print('i_episode: ',i_episode, 'epoch: ', epoch,' loss1: ', loss1.tolist(),' loss2: ', loss2.tolist())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer2.zero_grad()

        # perform a backward pass (backpropagation)
        loss2.backward()

        # Update the parameters
        optimizer2.step()
        if -1*loss2/X*4000/X*4000>X*1.5+300:
            k*=1.5
        elif -1*loss2/X*4000/X*4000<=(X-300)*0.8:
            k/=2
        print(k,' ',end="")
        k = int(k)
        if k<2:
            k=2    
    print()
    print('i_episode: ',i_episode, ' loss1: ', loss1.tolist(),' loss2: ', loss2.tolist())
    if i_episode%20==0:
        t.save(actor.state_dict(), "actor.pkl")
        t.save(critic.state_dict(), "critic.pkl")
        # t.save(Controller_.state_dict(), "controller.pkl")
import matplotlib.pyplot as plt

plt.plot(R)
plt.show()
plt.plot(Xlist,Hlist)
plt.show()
plt.plot(Xlist,alphalist)
plt.show()