import deepSI
from deepSI import System_data, System_data_list
from deepSI.fit_systems import SS_encoder, SS_encoder_general, System_torch
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import optim, nn
from tqdm.auto import tqdm
import matplotlib

class LPV_with_schedul_net(nn.Module):
    def __init__(self, nx, nu, ny, Np, pnet, include_u_in_p=False, F=10):
        super(LPV_with_schedul_net,self).__init__()
        #without the 0.1 it sometimes was unstable? Might need to be adressed in the future
        make_mat = lambda n_in, n_out: 1/F*(torch.rand(n_out,n_in)*2-1)/n_in**0.5#returns a matrix of size (n_out, n_in) with uniform 
        
        self.nx = nx
        self.nu = nu
        self.ny = ny #=nx if it is the state equation
        self.Np = Np #np=0 is linear
        nu = 1 if nu==None else nu
        ny = 1 if ny==None else ny
        
        self.A = nn.Parameter(make_mat(nx, ny))
        self.B = nn.Parameter(make_mat(nu, ny))
        
        self.include_u_in_p = include_u_in_p
        self.pnet = pnet
        self.As = nn.Parameter(torch.stack([make_mat(nx, ny) for _ in range(Np)]))
        self.Bs = nn.Parameter(torch.stack([make_mat(nu, ny) for _ in range(Np)]))
    
    def forward(self, x, u=None, xenc=None):
        if self.nu==None:
            u = u[:,None] #(Nb,1)
        xp = x if xenc is None else xenc
        pin = torch.cat([xp,u],dim=1) if self.include_u_in_p else xp
        pout = self.pnet(pin)
        
        ylin = torch.einsum('ij,bj->bi',self.A,x) + torch.einsum('ij,bj->bi',self.B,u)
        ynonlin = torch.einsum('pij,bp,bj->bi',self.As,pout,x) + torch.einsum('pij,bp,bj->bi',self.Bs,pout,u)
        yout = ylin + ynonlin
        return yout[:,0] if self.ny==None else yout

    def parameters(self): #exclude pnet from the parameters
        return nn.ParameterList([self.A, self.B, self.As, self.Bs])


from deepSI.fit_systems.encoders import default_encoder_net
from deepSI.utils import simple_res_net
class LPV_single_encoder(SS_encoder_general):
    def __init__(self, nx=10, na=20, nb=20, Np=2,           feedthrough=True,  include_u_in_p=True, \
        e_net=default_encoder_net, f_net=LPV_with_schedul_net,           h_net=LPV_with_schedul_net,   p_net=simple_res_net, \
        e_net_kwargs={},           f_net_kwargs={},         h_net_kwargs={}, p_net_kwargs={}):
        assert feedthrough==True, 'non-feedthrough has not been implemented for this system yet'
        super(LPV_single_encoder, self).__init__(nx=nx,na=na,nb=nb, feedthrough=feedthrough, \
            e_net=e_net,f_net=LPV_with_schedul_net, h_net=LPV_with_schedul_net, \
            e_net_kwargs=e_net_kwargs, f_net_kwargs=f_net_kwargs, \
            h_net_kwargs=h_net_kwargs)
        self.p_net = p_net
        self.p_net_kwargs = p_net_kwargs
        self.include_u_in_p = include_u_in_p
        self.Np = Np
    
    def init_nets(self, nu, ny):
        nuval = 1 if nu==None else nu
        nyval = 1 if ny==None else ny
        self.pnet = self.p_net(n_in=self.nx + (nuval if self.include_u_in_p else 0), n_out = self.Np, **self.p_net_kwargs)
        self.encoder = self.e_net(nb=self.nb, nu=nu, na=self.na, ny=ny, nx=self.nx, **self.e_net_kwargs)
        #nx, nu, ny, np, pnet
        self.fn = self.f_net(nx=self.nx, nu=nu, ny=self.nx, Np=self.Np, pnet=self.pnet, include_u_in_p=self.include_u_in_p, **self.f_net_kwargs)
        self.hn = self.h_net(nx=self.nx, nu=nu, ny=ny,      Np=self.Np, pnet=self.pnet, include_u_in_p=self.include_u_in_p, **self.h_net_kwargs)

    def loss(self, uhist, yhist, ufuture, yfuture, loss_nf_cutoff=None, **Loss_kwargs):
        x = self.encoder(uhist, yhist) #initialize Nbatch number of states
        errors = []
        for y, u in zip(torch.transpose(yfuture, 0, 1), torch.transpose(ufuture, 0, 1)): #iterate over time
            xenc = self.encoder(uhist, yhist)
            yhat = self.hn(x=x, u=u, xenc=xenc) if self.feedthrough else self.hn(x, u=None, xenc=xenc)
            error = nn.functional.mse_loss(y, yhat)
            errors.append(error)
            if loss_nf_cutoff is not None and error.item() > loss_nf_cutoff:
                print(len(errors), end=' ')
                break

            uhist = torch.cat([uhist[:, 1:], u[:, None]], dim=1)
            yhist = torch.cat([yhist[:, 1:], y[:, None]], dim=1)
            x = self.fn(x, u)

        return torch.mean(torch.stack(errors))

class LPV_multi_encoder(LPV_single_encoder):
    def __init__(self, nx=10, na=20, nb=20, Np=2,           feedthrough=True,  include_u_in_p=True, yhat_to_enc=False, \
        e_net=default_encoder_net, f_net=LPV_with_schedul_net,           h_net=LPV_with_schedul_net,   p_net=simple_res_net, \
        e_net_kwargs={},           f_net_kwargs={},         h_net_kwargs={}, p_net_kwargs={}, y_to_enc=True):
        
        super(LPV_multi_encoder, self).__init__(nx=nx, na=na, nb=nb, Np=Np, feedthrough=feedthrough,  include_u_in_p=include_u_in_p, \
        e_net=e_net, f_net=f_net,           h_net=h_net,   p_net=p_net, \
        e_net_kwargs=e_net_kwargs,           f_net_kwargs=f_net_kwargs,         h_net_kwargs=h_net_kwargs, p_net_kwargs=p_net_kwargs)
        
        self.yhat_to_enc = yhat_to_enc
    
    def loss(self, uhist, yhist, ufuture, yfuture, loss_nf_cutoff=None, **Loss_kwargs):
        x = self.encoder(uhist, yhist) #initialize Nbatch number of states
        errors = []
        for y, u in zip(torch.transpose(yfuture,0,1), torch.transpose(ufuture,0,1)): #iterate over time
            xenc = self.encoder(uhist, yhist)
            yhat = self.hn(x=x,u=u,xenc=xenc) if self.feedthrough else self.hn(x,u=None,xenc=xenc)
            error = nn.functional.mse_loss(y, yhat)
            errors.append(error) #calculate error after taking n-steps
            if loss_nf_cutoff is not None and error.item()>loss_nf_cutoff:
                print(len(errors), end=' ')
                break
            
            #uhist = (Nb, nb, nu)
            #yhist = (Nb, na, ny)
            uhist = torch.cat([uhist[:,1:],u[:,None]],dim=1)
            if self.yhat_to_enc:
                yhist = torch.cat([yhist[:,1:],yhat[:,None]],dim=1)
            else:
                yhist = torch.cat([yhist[:,1:],y[:,None]],dim=1)
            
            x = self.fn(x,u) #advance state. 
        return torch.mean(torch.stack(errors))
    
    def apply_experiment(self, data, filt=True):
        if filt:
            return self.filter_experiment(data)
        else:
            return super().apply_experiment(data)
    
    def filter_experiment(self, sys_data): 
        if isinstance(sys_data, System_data_list):
            return System_data_list([self.filter_experiment(s) for s in sys_data])
        
        sys_data_norm = self.norm.transform(sys_data)
        
        A = sys_data_norm.to_hist_future_data(na=self.na, nb=self.nb, nf=len(sys_data)-self.k0)
        uhist, yhist, ufuture, yfuture = [torch.as_tensor(a,dtype=torch.float32) for a in A]
        yhats = []
        
        with torch.no_grad():
            x = self.encoder(uhist, yhist)
            for y, u in zip(torch.transpose(yfuture,0,1), torch.transpose(ufuture,0,1)): #iterate over time
                xenc = self.encoder(uhist, yhist)
                yhat = self.hn(x=x,u=u,xenc=xenc) if self.feedthrough else self.hn(x,u=None,xenc=xenc)
                yhats.append(yhat.numpy()[0]) #calculate error after taking n-steps

                #uhist = (Nb, nb, nu)
                #yhist = (Nb, na, ny)
                uhist = torch.cat([uhist[:,1:],u[:,None]],dim=1)
                if self.yhat_to_enc:
                    yhist = torch.cat([yhist[:,1:],yhat[:,None]],dim=1)
                else:
                    yhist = torch.cat([yhist[:,1:],y[:,None]],dim=1)
                x = self.fn(x,u) #advance state. 
            yhats = np.concatenate([sys_data_norm.y[:self.k0], np.array(yhats)],axis=0)
            return self.norm.inverse_transform(System_data(u=sys_data_norm.u, y=yhats,cheat_n=self.k0,normed=True))