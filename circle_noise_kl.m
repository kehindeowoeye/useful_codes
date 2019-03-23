%circle with noise kl divergence

 df = [];dada = zeros(1,10);
 theta = 0:0.01:2*pi;
 r = 2;
 x = r*cos(theta);
 y = r*sin(theta);
 %plot(x,y),axis equal
 a1 = [x,y];
 
 for j = 1:100
 
  for i = 1:10
     x1 = awgn(x,i,'measured');   
     y1 = awgn(y,i,'measured');  
     a2 = [x1,y1];
     
     
     [p1,~] = ksdensity(a1);
     [p2,~] = ksdensity(a2);
     
     kl = (p1+eps).* ((p1+eps)-log(p2+eps));
     kl1 = (p2+eps).* ((p2+eps)-log(p1+eps));
     
     kl = (kl(~isinf(kl)));
     kl = sum(kl(~isnan(kl)));
     
     kl1 = (kl1(~isinf(kl1)));
     kl1 = sum(kl1(~isnan(kl1)));
     
     dkl = 0.5*(kl+kl1);
     df = horzcat(df,dkl);
    
  end
 
  dada = dada + df;
  df = [];
 end
 
 dada = dada/100;
 plot(1:10,dada)
 xlabel('Signal to noise ratio')
 ylabel('KL Divergence')
