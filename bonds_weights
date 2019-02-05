%codes to learn the parameters of boid model



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


