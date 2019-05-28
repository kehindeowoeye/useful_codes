import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import random
from scipy import stats



cv1 = np.array(pd.read_csv('alldata0401.csv',header=None));
cv2 = np.array(pd.read_excel('alldata0404.xls',header=None))
cv3 = np.array(pd.read_csv('alldata0405.csv',header=None));
cv4 =np.array(pd.read_csv('big0405.csv',header=None))
ep = np.finfo(float).eps


def density_estimate(m1,m2):
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    print(values.shape)
    kernel = stats.gaussian_kde(values)
    Z = kernel(positions)
    return Z



###########################################################################################
def myap1(ala, data, who):
    ala = ala;data = data;le = 3600
    df = [];lm = range(5,65,5);lm = np.hstack((1,lm));
    print(lm)
    #KL divergence for my approach
    for i in range(0,13):
        pa = data;
        #pa = xlsread('yuyu_newestfeb06');
        #pa = pa(1:36001,:);%select just ten hours
        pa = pa[0:le,:];#select just ten hours
        ld = range(0,len(pa),lm[i]);
        print(len(ld))
        pa = pa[ld,:];
        pa = np.diff(pa,axis = 0);#compute_velocity
    
        pa_gd = cv2
        print(pa_gd.shape)
        pa_gd = pa_gd[ala:pa_gd.shape[0],:];pa_gd = pa_gd[0:le,0:72];
        print(pa_gd.shape)
        pa_gd = pa_gd[ld,:];
    
        #print(pa_gd.shape)
        print("This is ")
        pa_gd = np.diff(pa_gd,axis = 0);
        print(pa_gd.shape)
        #pa_gd = pa_gd(1:le,:);
        we = pa_gd
        #rdn = random.randint(0,35)
        rdn = who
        pa_gd = (we[:,rdn]).reshape(len(we[:,rdn]),1)
        pa_gd = np.concatenate((pa_gd,(we[:,rdn+36]).reshape(len(we[:,rdn+36]),1)),axis = 1);
        #pa_gd = np.vstack((pa_gd,we[:,37]))
        #pa_gd = pa_gd(ld,:); 
        print(pa_gd.shape)
        a1 = pa;
        a1 =  density_estimate(a1[:,0],a1[:,1]);
        a2 = pa_gd;
        a2 =  density_estimate(a2[:,0],a2[:,1]);
    
        #print(a1)
        kl = (a1+ep)* ((a1+ep)-np.log(a2+ep));
        kl1 = (a2+ep)* ((a2+ep)-np.log(a1+ep));
        #kl = (kl(~isinf(kl)));
        #kl = sum(kl(~isnan(kl)));
        #kl1 = (kl1(~isinf(kl1)));
        #kl1 = sum(kl1(~isnan(kl1)));
        dkl = 0.5*(kl+kl1);
        df = np.hstack((df,dkl));
        print(df)
    return(df)


def myap2(ala, data,who):
    ala = ala;data = data;le = 7200
    df = [];lm = range(5,65,5);lm = np.hstack((1,lm));
    print(lm)
    #KL divergence for my approach
    for i in range(0,13):
        pa = data;
        #pa = xlsread('yuyu_newestfeb06');
        #pa = pa(1:36001,:);%select just ten hours
        pa = pa[0:le,:];#select just ten hours
        ld = range(0,len(pa),lm[i]);
        print(len(ld))
        pa = pa[ld,:];
        pa = np.diff(pa,axis = 0);#compute_velocity
        
        pa_gd = cv2
        print(pa_gd.shape)
        pa_gd = pa_gd[ala:pa_gd.shape[0],:];pa_gd = pa_gd[0:le,0:72];
        print(pa_gd.shape)
        pa_gd = pa_gd[ld,:];
        
        #print(pa_gd.shape)
        print("This is ")
        pa_gd = np.diff(pa_gd,axis = 0);
        print(pa_gd.shape)
        #pa_gd = pa_gd(1:le,:);
        we = pa_gd
        #rdn = random.randint(0,35)
        rdn = who
        pa_gd = (we[:,rdn]).reshape(len(we[:,rdn]),1)
        pa_gd = np.concatenate((pa_gd,(we[:,rdn+36]).reshape(len(we[:,rdn+36]),1)),axis = 1);
        #pa_gd = np.vstack((pa_gd,we[:,37]))
        #pa_gd = pa_gd(ld,:);
        print(pa_gd.shape)
        a1 = pa;
        a1 =  density_estimate(a1[:,0],a1[:,1]);
        a2 = pa_gd;
        a2 =  density_estimate(a2[:,0],a2[:,1]);
        
        #print(a1)
        kl = (a1+ep)* ((a1+ep)-np.log(a2+ep));
        kl1 = (a2+ep)* ((a2+ep)-np.log(a1+ep));
        #kl = (kl(~isinf(kl)));
        #kl = sum(kl(~isnan(kl)));
        #kl1 = (kl1(~isinf(kl1)));
        #kl1 = sum(kl1(~isnan(kl1)));
        dkl = 0.5*(kl+kl1);
        df = np.hstack((df,dkl));
        print(df)
    return(df)



def myap3(ala, data,who):
    ala = ala;data = data;le = 10800
    df = [];lm = range(5,65,5);lm = np.hstack((1,lm));
    print(lm)
    #KL divergence for my approach
    for i in range(0,13):
        pa = data;
        #pa = xlsread('yuyu_newestfeb06');
        #pa = pa(1:36001,:);%select just ten hours
        pa = pa[0:le,:];#select just ten hours
        ld = range(0,len(pa),lm[i]);
        print(len(ld))
        pa = pa[ld,:];
        pa = np.diff(pa,axis = 0);#compute_velocity
        
        pa_gd = cv2
        print(pa_gd.shape)
        pa_gd = pa_gd[ala:pa_gd.shape[0],:];pa_gd = pa_gd[0:le,0:72];
        print(pa_gd.shape)
        pa_gd = pa_gd[ld,:];
        
        #print(pa_gd.shape)
        print("This is ")
        pa_gd = np.diff(pa_gd,axis = 0);
        print(pa_gd.shape)
        #pa_gd = pa_gd(1:le,:);
        we = pa_gd
        #rdn = random.randint(0,35)
        rdn = who
        pa_gd = (we[:,rdn]).reshape(len(we[:,rdn]),1)
        pa_gd = np.concatenate((pa_gd,(we[:,rdn+36]).reshape(len(we[:,rdn+36]),1)),axis = 1);
        #pa_gd = np.vstack((pa_gd,we[:,37]))
        #pa_gd = pa_gd(ld,:);
        print(pa_gd.shape)
        a1 = pa;
        a1 =  density_estimate(a1[:,0],a1[:,1]);
        a2 = pa_gd;
        a2 =  density_estimate(a2[:,0],a2[:,1]);
        
        #print(a1)
        kl = (a1+ep)* ((a1+ep)-np.log(a2+ep));
        kl1 = (a2+ep)* ((a2+ep)-np.log(a1+ep));
        #kl = (kl(~isinf(kl)));
        #kl = sum(kl(~isnan(kl)));
        #kl1 = (kl1(~isinf(kl1)));
        #kl1 = sum(kl1(~isnan(kl1)));
        dkl = 0.5*(kl+kl1);
        df = np.hstack((df,dkl));
        print(df)
    return(df)
##########################################################################################



#KL divergence for boids
def boids(start,who):
    ala = start;le = 10800
    df = [];
    lm = range(5,65,5);lm = np.hstack((1,lm));
    lm = np.hstack(( 1,lm ));
    #lm  = 1

    for fa in range (0,len(lm)):
        bv =   cv1;
        bv = bv[:,0:72];
        #ala = 657;
        #le = 62001;
   
  
        bv = bv[512:,:];
        bvx = bv[:,0:36];bvy = bv[:,36:73];
        #rdn = random.randint(0,35)
        rdn = who
        
        bg = np.vstack((bvx[:,rdn], bvy[:,rdn]));
        bg = bg.T
   
        bvx =np.delete(bvx,rdn,axis=1)
        bvy =np.delete(bvy,rdn,axis=1)
    
        
        ld = range(0,len(bvx),lm[fa]);

        bg = bg[ld,:]


        bx = bvx;by = bvy;
        bvx = bvx[0:,:];bvx1 = bvx[ld,:]; #x input
        bvy = bvy[0:,:];bvy1 = bvy[ld,:]; #y input
        bgv = np.diff(bg,axis = 0);
        avg_velox = (np.diff(bvx,axis = 0));avg_veloy = (np.diff(bvy,axis = 0));
        bg = bg[0:len(bg)-1,:];
        bvx1 = bvx1[0:len(bvx1)-1,:];
        bvy1 = bvy1[0:len(bvy1)-1,:];
        avg_x1 = avg_velox;avg_y1 = avg_veloy;


        centrox = bvx;centroy = bvy;
        dd = np.sqrt( np.power( ( (bg[:,0]).reshape(len(bg[:,0]),1) - bvx1), 2 )  +  np.power( ((bg[:,1]).reshape(len(bg[:,1]),1)- bvy1 ), 2) );
        ddg = np.sort(dd,1);
        ka = [];kad=[];


        for i in range (0, len(ddg)):
            ka = np.argsort(dd[i,:]);
            ka = ka.reshape(1,len(ka))
            if i ==0:
                kad = ka[:,0:15] 
            else:
                kad = np.vstack((  kad  ,ka[:,0:15]  ));ka=[];
       

        ccx=[];ccy=[];avx=[];avy=[];
        for i in range (0, len(ddg)):
            ucx = centrox[i,:];   ucy = centroy[i,:];
            if i == 0:
                ccx = ucx[kad[i,:]]
                ccy = ucy[kad[i,:]]
            else:
                ccx = np.vstack(( ccx, ucx[kad[i,:]]  ));
                ccy = np.vstack(( ccy, ucy[kad[i,:]]  ));
            ucx = avg_x1[i,:];   ucy = avg_y1[i,:];
            if i == 0:
                avx = ucx[kad[i,:]]
                avy = ucy[kad[i,:]]
            else:
                avx = np.vstack(( avx,ucx[kad[i,:]] ));
                avy = np.vstack(( avy,ucy[kad[i,:]]  ));
      
        lax = ccx;lay=ccy;
        ccx = np.mean(ccx,axis = 1);ccy = np.mean(ccy,axis = 1);avx = np.mean(avx,axis = 1);avy = np.mean(avy,axis = 1);
      
        lada =  (bgv[:,0] + bgv[:,1]).reshape(len(bgv),1);
        lada = np.hstack( ( lada, (bg[:,0]-ccx + bg[:,1]-ccy).reshape(len(bg[:,0]),1), (avx+avy).reshape(len(avx),1), (-(bg[:,0]-lax[:,0])-(bg[:,1]-lay[:,0]) ).reshape(len(bg[:,0]),1)  ));
        #weight = lada[:,1:]\lada[:,0]
        print((lada[:,1:]).shape)
        print((lada[:,0]).shape)

       
        #weight = np.linalg.solve(np.array(lada[:,1:]), np.array(lada[:,0]) )
        weight = np.linalg.lstsq(np.array(lada[:,1:]), np.array(lada[:,0]) )[0]
        print(weight)





        w2 = weight;
        pa_gd  =   cv2   #loads data data data 
        pa_gd = pa_gd[ala-1:,:];pa_gd = pa_gd[:,0:72];
        pa_gd = pa_gd[0:le,:];
        ld = range(0,len(pa_gd),lm[fa]);
        #ld = 1:lm[fa]:len(pa_gd)
        pa_gd = pa_gd[ld,:];
    
        ag = np.hstack(( pa_gd[0,rdn],pa_gd[0,rdn+36] ));
        print(ag)
        print("This is ag")
        pa_gd = np.delete(pa_gd,rdn,  axis=1)
        pa_gd = np.delete(pa_gd,rdn+35,  axis=1)#35 because you have deleted one 
        #pa_gd(:,rdn)=[];pa_gd(:,rdn+36)=[];
    
        vv = np.diff(pa_gd, axis = 0);
        dc=[];d5=[];ag1=[];
        for i in range(0,len(pa_gd)-1):
            if i == 0:
                ag1 = ag
            else:
                ag1 = np.vstack(( ag1,ag ));
            #print((pa_gd[i,0:35]).shape)
            #print((pa_gd[i,35:71]).shape)
            dc1 = np.sqrt(  np.power((( ag[0]- pa_gd[i,0:35])),2) + np.power(((ag[1]-pa_gd[i,35:70])), 2))
           
            dc2 = np.argsort(dc1)
          
            dc2 = dc2[0:15];
            
            for j in range (0,15):
                if j == 0:
                    dc = dc2[j]
                else:
                    dc = np.hstack(( dc, dc2[j] ));
        
        
            d5 = dc[0];
            gux = pa_gd[i,0:35];guy = pa_gd[i,35:70];
            fux = vv[i,0:35];fuy = vv[i,35:70];
            l1 =  ag[0]-np.mean(gux[dc]) 
            l1 =  np.hstack(( l1, (ag[1] - np.mean(guy[dc] )) ));
            l2 = np.hstack(( np.mean(fux[dc]),      np.mean(fuy[dc])   ));
            l3 = np.hstack((  (ag[0]- gux[d5]), (ag[1] - guy[d5] )  ));
            
            ga = l1
            ga = np.hstack((ga,l2,l3));
            
      
            ga_ = w2[0]*ga[0:2]
            
            ga_ = np.hstack((ga_ ,  w2[1]*ga[2:4] , w2[2]*ga[4:6] ));
            #print(ga_)
            ga = ga_[0:2]+ga_[2:4]-ga_[4:6];
            ag = ag+ga;
            ga_ = [];dc=[];ga=[];
        


        a1 = np.diff(ag1,axis = 0)
        pa_gd  =  cv2
        pa_gd = pa_gd[ala-1:,:];pa_gd = pa_gd[:,0:72];
        pa_gd = pa_gd[0:le,:];
        ld = range(0,len(pa_gd),lm[fa]);
    
        pa_gd = pa_gd[ld,:];
        pa_gd = np.diff(pa_gd,axis = 0);
        pa_gd_ = (pa_gd[:,rdn]).reshape(len(pa_gd[:,rdn]),1)
        pa_gd_ = np.hstack((pa_gd_ , (pa_gd[:,rdn+36]).reshape(len(pa_gd[:,rdn+36]),1) ));
    
        #print(pa_gd.shape)
        a2 = pa_gd_;
            
        print(a1)
 
            
        a2 = density_estimate(a2[:,0],a2[:,1]);
        a1 = density_estimate(a1[:,0],a1[:,1]);
            
           
        kl = (a1+ep)* ((a1+ep)-np.log(a2+ep));
        kl1 = (a2+ep)* ((a2+ep)-np.log(a1+ep));
        dkl = 0.5*(kl+kl1);
        df = np.hstack((df,dkl));
        print(df)
    return (df)





























#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#distance to KNN and number of nearest neighbour within certain range
def nn(start, data, who):
    df = [];lm = range(5,65,5);lm = np.hstack((1,lm));
    ala = start
    le = 10800

    print(lm);tr = range(5,80,5); 
    #KL divergence for my approach
    for i in range(0,13):

        #le = 3600;ala = 657;
    
        cv1 = cv2;
        cv1 = cv1[ala-1:,:];cv1 = cv1[:,0:72];
        cv1 = cv1[0:le,:];#selects data of interest 
        cv1x = cv1[0:le,0:36];cv1y = cv1[0:le,36:72];#separate data into x and y
        uy = data;#loads test data 
        #rdn = 0
        rdn = who
        
        print((cv1x[:,rdn]).shape)
        #uy = np.hstack(( (cv1x[:,rdn]).reshape(len(cv1x[:,rdn]),1), (cv1y[:,rdn]).reshape(len(cv1y[:,rdn]),1) ));


        #a6 = 0:1:10000;
        a6 = range(0,le,1)
        px = uy[:,0]
        py = uy[:,1]







        cv1x = px.reshape(len(px),1)- cv1x
        cv1y = py.reshape(len(py),1) - cv1y
        dist = np.sqrt( np.power(cv1x,2)   +   np.power(cv1y,2))

        for m in range(0,len(tr) ):

            g1 = [];g2 = [];
            for i in range (0, len(px) ):#len of data 

               fd = np.sort(dist[i,:]);fd = np.delete(fd,35,axis=0)
               g1 = len( fd[fd<=tr[m]] )
      
           
               if i == 0:
                   g2 = g1
                   d1 = fd.reshape(1,len(fd))
               else: 
                   g2 = np.vstack(( g2,g1 )); 
                   d1 = np.vstack(( d1, fd.reshape(1,len(fd)))); 
               g1 = []; 
    
       
            vc = np.sum(g2,axis = 1);
            vd = vc
            if m == 0:
                g5 = vc.reshape(len(vc),1)
                d5 = d1
            else:
                g5 = np.hstack((g5, vc.reshape(len(vc),1)  ))
                d5 = np.hstack((d5, d1))
 
        print(g5.shape)
        print(d5.shape)


        #le = 3600;ala = 657;
 
        cv1 = cv2;
        cv1 = cv1[ala-1:,:];cv1 = cv1[:,0:72];
        cv1 = cv1[0:le,:];#selects data of interest 
        cv1x = cv1[0:le,0:36];cv1y = cv1[0:le,36:72];#separate data into x and y
        #uy = xlsread('debe1807182.xlsx');#loads test data 
        rdn = who

        #print((cv1x[:,rdn]).shape)
        uy = np.hstack(( (cv1x[:,rdn]).reshape(len(cv1x[:,rdn]),1), (cv1y[:,rdn]).reshape(len(cv1y[:,rdn]),1) ));

        a6 = range(0,le,1)
        px = uy[:,0]
        py = uy[:,1]

        cv1x = np.delete(cv1x,rdn,axis = 1)
        cv1y = np.delete(cv1y,rdn,axis = 1)


        cv1x = px.reshape(len(px),1)- cv1x
        cv1y = py.reshape(len(py),1) - cv1y
        dist = np.sqrt( np.power(cv1x,2)   +   np.power(cv1y,2))

        for m in range(0,len(tr) ):
            g1 = [];g2 = [];
            for i in range (0, len(px) ):#size of data 
               fd = np.sort(dist[i,:]);
               g1 = len( fd[fd<=tr[m]] )
               
               if i == 0:
                   g2 = g1
                   d1 = fd.reshape(1,len(fd))
               else: 
                   g2 = np.vstack(( g2,g1 ));
                   d1 = np.vstack(( d1, fd.reshape(1,len(fd)))); 
               g1 = []; 
    

            vc = np.sum(g2,axis = 1);

            vd = vc
            if m == 0:
                g6 = vc.reshape(len(vc),1)
                d6 = d1
            else:
                g6 = np.hstack((g6, vc.reshape(len(vc),1)  ))
                d6 = np.hstack((d6, d1))
 
        print(g6.shape)
        print(np.abs(g5-g6))
        print(np.abs(d5-d6))
        aa = (np.abs(g5-g6))
        ab = (np.abs(d5-d6))
    return (aa,ab)








#KL divergence for boids
def boids_nn(start,who):
    ala = start;le = 10800
    df = [];
    lm = range(5,65,5);lm = np.hstack((1,lm));
    lm = np.hstack(( 1,lm ));
    #lm  = 1
    
    for fa in range (0,len(lm)):
        bv =   cv1;
        bv = bv[:,0:72];
        #ala = 657;
        #le = 62001;
        
        
        bv = bv[512:,:];
        bvx = bv[:,0:36];bvy = bv[:,36:73];
        rdn = who
        
        bg = np.vstack((bvx[:,rdn], bvy[:,rdn]));
        bg = bg.T
        
        bvx =np.delete(bvx,rdn,axis=1)
        bvy =np.delete(bvy,rdn,axis=1)
        
        
        ld = range(0,len(bvx),lm[fa]);
        
        bg = bg[ld,:]
        
        
        bx = bvx;by = bvy;
        bvx = bvx[0:,:];bvx1 = bvx[ld,:]; #x input
        bvy = bvy[0:,:];bvy1 = bvy[ld,:]; #y input
        bgv = np.diff(bg,axis = 0);
        avg_velox = (np.diff(bvx,axis = 0));avg_veloy = (np.diff(bvy,axis = 0));
        bg = bg[0:len(bg)-1,:];
        bvx1 = bvx1[0:len(bvx1)-1,:];
        bvy1 = bvy1[0:len(bvy1)-1,:];
        avg_x1 = avg_velox;avg_y1 = avg_veloy;
        
        
        centrox = bvx;centroy = bvy;
        dd = np.sqrt( np.power( ( (bg[:,0]).reshape(len(bg[:,0]),1) - bvx1), 2 )  +  np.power( ((bg[:,1]).reshape(len(bg[:,1]),1)- bvy1 ), 2) );
        ddg = np.sort(dd,1);
        ka = [];kad=[];
        
        
        for i in range (0, len(ddg)):
            ka = np.argsort(dd[i,:]);
            ka = ka.reshape(1,len(ka))
            if i ==0:
                kad = ka[:,0:15]
            else:
                kad = np.vstack((  kad  ,ka[:,0:15]  ));ka=[];


        ccx=[];ccy=[];avx=[];avy=[];
        for i in range (0, len(ddg)):
            ucx = centrox[i,:];   ucy = centroy[i,:];
            if i == 0:
                ccx = ucx[kad[i,:]]
                ccy = ucy[kad[i,:]]
            else:
                ccx = np.vstack(( ccx, ucx[kad[i,:]]  ));
                ccy = np.vstack(( ccy, ucy[kad[i,:]]  ));
            ucx = avg_x1[i,:];   ucy = avg_y1[i,:];
            if i == 0:
                avx = ucx[kad[i,:]]
                avy = ucy[kad[i,:]]
            else:
                avx = np.vstack(( avx,ucx[kad[i,:]] ));
                avy = np.vstack(( avy,ucy[kad[i,:]]  ));

        lax = ccx;lay=ccy;
        ccx = np.mean(ccx,axis = 1);ccy = np.mean(ccy,axis = 1);avx = np.mean(avx,axis = 1);avy = np.mean(avy,axis = 1);
    
        lada =  (bgv[:,0] + bgv[:,1]).reshape(len(bgv),1);
        lada = np.hstack( ( lada, (bg[:,0]-ccx + bg[:,1]-ccy).reshape(len(bg[:,0]),1), (avx+avy).reshape(len(avx),1), (-(bg[:,0]-lax[:,0])-(bg[:,1]-lay[:,0]) ).reshape(len(bg[:,0]),1)  ));
        print((lada[:,1:]).shape)
        print((lada[:,0]).shape)
        
        
        #weight = np.linalg.solve(np.array(lada[:,1:]), np.array(lada[:,0]) )
        weight = np.linalg.lstsq(np.array(lada[:,1:]), np.array(lada[:,0]) )[0]
        print(weight)
        
        
        
        
        
        w2 = weight;
        pa_gd  =   cv2   #loads data data data
        pa_gd = pa_gd[ala-1:,:];pa_gd = pa_gd[:,0:72];
        pa_gd = pa_gd[0:le,:];
        ld = range(0,len(pa_gd),lm[fa]);
        #ld = 1:lm[fa]:len(pa_gd)
        pa_gd = pa_gd[ld,:];
        
        ag = np.hstack(( pa_gd[0,rdn],pa_gd[0,rdn+36] ));
        print(ag)
        print("This is ag")
        pa_gd = np.delete(pa_gd,rdn,  axis=1)
        pa_gd = np.delete(pa_gd,rdn+35,  axis=1)#35 because you have deleted one
        #pa_gd(:,rdn)=[];pa_gd(:,rdn+36)=[];
        
        vv = np.diff(pa_gd, axis = 0);
        dc=[];d5=[];ag1=[];
        for i in range(0,len(pa_gd)-1):
            if i == 0:
                ag1 = ag
            else:
                ag1 = np.vstack(( ag1,ag ));
            #print((pa_gd[i,0:35]).shape)
            #print((pa_gd[i,35:71]).shape)
            dc1 = np.sqrt(  np.power((( ag[0]- pa_gd[i,0:35])),2) + np.power(((ag[1]-pa_gd[i,35:70])), 2))
            
            dc2 = np.argsort(dc1)
            
            dc2 = dc2[0:15];
            
            for j in range (0,15):
                if j == 0:
                    dc = dc2[j]
                else:
                    dc = np.hstack(( dc, dc2[j] ));
            
            
            d5 = dc[0];
            gux = pa_gd[i,0:35];guy = pa_gd[i,35:70];
            fux = vv[i,0:35];fuy = vv[i,35:70];
            l1 =  ag[0]-np.mean(gux[dc])
            l1 =  np.hstack(( l1, (ag[1] - np.mean(guy[dc] )) ));
            l2 = np.hstack(( np.mean(fux[dc]),      np.mean(fuy[dc])   ));
            l3 = np.hstack((  (ag[0]- gux[d5]), (ag[1] - guy[d5] )  ));
            
            ga = l1
            ga = np.hstack((ga,l2,l3));
            
            
            ga_ = w2[0]*ga[0:2]
            
            ga_ = np.hstack((ga_ ,  w2[1]*ga[2:4] , w2[2]*ga[4:6] ));
            ga = ga_[0:2]+ga_[2:4]-ga_[4:6];
            ag = ag+ga;
            ga_ = [];dc=[];ga=[];

        rec,red = nn(start, data, who)

return (rec,red)




                                    

                                
                                    

                                    



