
startup
bv = csvread('alldata0401.csv');bv = bv(:,1:72);
bvx = bv(:,1:36);bvy = bv(:,37:72);
bg = [bvx(:,1),bvy(:,1)];
bvx(:,1)=[];bvy(:,1)=[];
ld = 1:10:length(bvx);
bx = bvx;by = bvy;
bvx = bvx(2:end,:);bvx1 = bvx(ld',:);%x input
bvy = bvy(2:end,:);bvy1 = bvy(ld',:);%y input
bgv = diff(bg);bgv = bgv(ld,:);%velocity target


avg_velox = (diff(bx));avg_veloy = (diff(by));
avg_x1 = avg_velox(ld',:);avg_y1 = avg_veloy(ld',:);


%centrox = mean(bvx,2);centroy = mean(bvy,2);
centrox = bvx;centroy = bvy;
cx1 = centrox(ld,:);
cy1 = centroy(ld,:);

bg = bg(2:end,:);bg = bg(ld',:);
dd = sqrt((bg(:,1)-bvx1).^2 +  (bg(:,2)-bvy1).^2);
ddg = sort(dd,2);
ka = [];kad=[];

for i = 1:length(ddg)
    for j = 1:15
       ka = horzcat(ka, find(dd(i,:)==ddg(i,j)) );
    end
    kad = vertcat(kad,ka);ka=[];
end

ccx=[];ccy=[];avx=[];avy=[];
for i = 1:length(ddg)
    ccx = centrox(kad);ccy = centroy(kad);
    avx = avg_x1(kad);avy = avg_y1(kad);
end

ccx = mean(ccx,2);ccy = mean(ccy,2);avx1 = mean(avx,2);avy1 = mean(avy,2);
bvx = bvx(ld,:);bvy = bvy(ld,:);
%lada = [];lada = horzcat(lada,sqrt(bgv(:,1).^2 + bgv(:,2).^2), bg(:,1)-ccx, bg(:,2)-ccy,avx,avy,-avx,-avy);
%lada = [];lada = horzcat(lada,sqrt(bgv(:,1).^2 + bgv(:,2).^2), bg(:,1)-ccx, bg(:,2)-ccy,avx1, avy1,-mean(avx(:,1:5),2),-mean(avy(:,1:5),2));
lada = [];lada = horzcat(lada, sqrt(bgv(:,1).^2 + bgv(:,2).^2),  sqrt((bg(:,1)-ccx).^2+ (bg(:,2)-ccy).^2), sqrt(avx1.^2+ avy1.^2), sqrt((-mean(avx(:,1:5),2)).^2+ (-mean(avy(:,1:5),2)).^2)) ;


meanfunc=[];
covfunc = @covSEard;
likfunc = @likGauss;

hyp = struct('mean',[],'cov',[0,0,0,0],'lik',-1);
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc,covfunc,likfunc,lada(:,2:end),lada(:,1));




%KL divergence for my approach 
pa = xlsread('yuyu_newestfeb06');pa = pa(1:3610,:);pa = diff(pa);
pa_gd  = csvread('alldata0404.csv');
pa_gd = pa_gd(10060:end,:);pa_gd = pa_gd(:,1:72);pa_gd = diff(pa_gd);
pa_gd = pa_gd(1:3609,:);
pa_gd = [pa_gd(:,1),pa_gd(:,37)];

%a1 = R(:,1);a2 = R(:,2);
a1 = pa;a2 = pa_gd;
cov1 = cov(a1);cov2 = cov(a2);
m1 = mean(a1);m2 = mean(a2);
dkl1 = 0.5*( trace(cov1*pinv(cov2)) -2 + (m2-m1)*pinv(cov2)*(m2-m1)' + log(det(cov(a2))/det(cov(a1))));
dkl2 = 0.5*( trace(cov2*pinv(cov1)) -2 + (m1-m2)*pinv(cov1)*(m1-m2)' + log(det(cov(a1))/det(cov(a2))));
dkl = 0.5*(dkl1+dkl2);


%weights
%w2 = [-2.7989 -2.2456 -4.0136 -3.5290 2.0900 0.3009];
%w2 = [-1.6311 -6.7958 -2.8643];
 w2 = [-5.5948   -0.0027   32.0705];

pa_gd  = csvread('alldata0404.csv');
pa_gd = pa_gd(10060:end,:);pa_gd = pa_gd(:,1:72);
pa_gd = pa_gd(1:3610,:);ag = [pa_gd(1,1),pa_gd(1,37)];pa_gd(:,1)=[];pa_gd(:,37)=[];
%pa_gd1 = sqrt( (diff(pa_gd(:,1:36))).^2+ (diff(pa_gd(:,37:72))).^2 );
%pa_gd1 = sort(pa_gd1,2);pa_gd1 = pa_gd1(:,1:15);
vv = diff(pa_gd);


dc=[];d5=[];ag1=[];
for i = 2:3610
    ag1 = vertcat(ag1,ag);
    dc1 = sqrt( (diff(pa_gd(i,1:36))).^2 +  (diff(pa_gd(i,1:36))).^ 2   );
    dc2 = sort(dc1,2);
    dc2 = dc2(:,1:15);
 
    for j = 1:15
        dc = horzcat(dc, find(dc1== dc2(j)) );
    end

    
    d5 = dc(:,1:5);
    gux = pa_gd(i,1:35);guy = pa_gd(i,36:70);
    fux = vv(i,1:35);fuy = vv(i,36:70);
    l1 =  [ ag(1)-mean(gux(dc) ), ag(2) - mean(guy(dc) )]; 
    l2 = [mean(fux(dc)),mean(fuy(dc))];  
    l3 = [mean(fux(d5)),mean(fuy(d5))] ;%dc=[];d5=[];
    ga = [l1,l2,l3];
    ga = [w2(1)*ga(:,1:2),w2(2)*ga(:,3:4) ,w2(3)*ga(:,5:6)];
    %ga = [w2(1)*ga(1),w2(2)*ga(2),w2(3)*ga(3),w2(4)*ga(4),w2(5)*ga(5),w2(6)*ga(6)];
 
    ga = ga(:,1:2)+ga(:,3:4)-ga(:,5:6);

    ag = ag+ga;
    ga = [];

end


a1 = diff(ag1);
pa_gd  = csvread('alldata0404.csv');
pa_gd = pa_gd(10060:end,:);pa_gd = pa_gd(:,1:72);pa_gd = diff(pa_gd);
pa_gd = pa_gd(1:3609,:);
pa_gd = [pa_gd(:,1),pa_gd(:,37)];

%a1 = R(:,1);a2 = R(:,2);
a2 = pa_gd;
cov1 = cov(a1);cov2 = cov(a2);
m1 = mean(a1);m2 = mean(a2);
dkl1 = 0.5*( trace(cov1*pinv(cov2)) -2 + (m2-m1)*pinv(cov2)*(m2-m1)' + log(det(cov(a2))/det(cov(a1))));
dkl2 = 0.5*( trace(cov2*pinv(cov1)) -2 + (m1-m2)*pinv(cov1)*(m1-m2)' + log(det(cov(a1))/det(cov(a2))));
dkl = 0.5*(dkl1+dkl2);
