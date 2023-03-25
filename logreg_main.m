% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2023 <Team brAIn>

% Logistic regression testbench

clear all; close all;
[Xtr,Ytr,Xte,Yte,names] = loadpenguindata("sex");
Xtr = [ones(length(Xtr),1) Xtr];
Xte = [ones(length(Xte),1) Xte];
Y = Ytr(:,1);
Yte = Yte(:,1);
NX=normalizer("normal");
NXtr=NX.fit_transform(Xtr);
NXte=NX.transform(Xte);

## Initial configuration for the optimizer
opt=optimizer("method","sgd",
              "minibatch",11,
              "maxiter",600,
              "alpha",0.04);
###

theta0=rand(columns(NXtr),1)-0.5; ## Common starting point (column vector)

# test all optimization methods
methods={"sgd","momentum","batch"};
##methods={"batch"};

tts1=zeros(numel(methods),5); %para los thetas
es=zeros(1,67);%para la grafica 2d

for m=1:numel(methods)
  method=methods{m};
  printf("Probando método '%s'.\n",method);
  msg=sprintf(";%s;",method); ## use method in legends

  try
    opt.configure("method",method); ## Just change the method
    [ts,errs]=opt.minimize(@logreg_loss,@logreg_gradloss,theta0,NXtr,Y);
    theta=ts{end}

    tts1(m,:)=ts{end};

    py=logreg_hyp(theta,NXte);
    err=sum((py>0.5)!=Yte);
    tot=100*(err/rows(Yte));
    es(1,:)=py';


    printf("errores de prueba: %d de %d (%.2f%%)\n", err, length(Yte), tot);

    py=logreg_hyp(theta,NXtr);
    err=sum((py>0.5)!=Y);
    tot=100*(err/rows(Y));
    printf("errores de entreneamineto: %d de %d (%.2f%%)\n", err, length(Y), tot);


    figure(1);
    plot(errs,msg,"linewidth",2);
    hold on;
  catch
    printf("\n### Error detectado probando método '%s': ###\n %s\n\n",
           method,lasterror.message);
  end_try_catch
endfor
xlabel("Iteration");
ylabel("Loss");
grid on;
comp=100;
columna1=0;
columna2=0;

for i=1:4
  for j=i+1:5

    feats=[i,j];
    x2=Xtr(:,feats);
    N2=normalizer("normal");
    nx2=N2.fit_transform(x2);

    opt.configure("method","batch"); ## Just change the method
    [ts,errs]=opt.minimize(@logreg_loss,@logreg_gradloss,theta0(feats),nx2,Y);
    theta2=ts{end};

    py2=logreg_hyp(theta2,nx2);
    err2=sum((py2>0.5)!=Y);
    tot2=100*(err2/rows(Y));

    if tot2<=comp
      comp=tot2;
      columna1=i;
      columna2=j;
    endif

    mins=min(x2);
    maxs=max(x2);

    e1=linspace(mins(1),maxs(1),50);
    e2=linspace(mins(2),maxs(2),50);
  endfor

endfor

printf("el menor error obtenido es: %d al evaluar las columnas %d y %d\n", comp, columna1, columna2);

##xop=nx2(:,[columna1,columna2]);

##mins=min(xop);
##maxs=max(xop);
##e1=linspace(mins(1),maxs(1),50);
##e2=linspace(mins(2),maxs(2),50);

[ee1,ee2]=meshgrid(e1,e2);
x2test=N2.transform([ee1(:) ee2(:)]);

ytest=logreg_hyp(theta2,x2test);


################################
figure(2,"name","Probabilidad")
surf(ee1,ee2,reshape(ytest,size(ee1)));
xlabel("culmen length [mm]");
ylabel("bodymass [g]");
zlabel("p(Female|x");
hold on;

contour3(ee1,ee2,reshape(ytest,size(ee1)),[0.25,0.5,0.75],"linewidth",3,"linecolor","black");



################################
figure(3,"name","Frontera de decisión y datos de prueba")

for q=1:67
  if es(1,q)>0.5
  plot(Xte(q,4),Xte(q,5),'x','color','b');
  hold on;
  elseif
  plot(Xte(q,4),Xte(q,5),'o','color','r');
  hold on;
  endif
endfor
legend('y=1','','','y=0')
contour(ee1,ee2,reshape(ytest,size(ee1)),[0,0.5,1],"linewidth",3,"linecolor","black");
xlabel("culmen length mm");
ylabel("Flipper length mm");

################################
feats2=[2,columna1,columna2];
x3=Xtr(:,feats2);
N2=normalizer("normal");
nx3=N2.fit_transform(x3);

opt.configure("method","batch"); ## Just change the method
[ts,errs]=opt.minimize(@logreg_loss,@logreg_gradloss,theta0(feats2),nx3,Y);
theta3=ts{end};

py3=logreg_hyp(theta3,nx3);
err3=sum((py3>0.5)!=Y);
tot3=100*(err3/rows(Y));

################################
ttsb=zeros(30,3); %para los thetas
ttss=zeros(30,3); %para los thetas
ttsm=zeros(30,3); %para los thetas
  v1=1;
  v2=1;
  v3=1;
for i=1:3
  for j=i+1:4
    for k=j+1:5
      for m=1:numel(methods)
        method=methods{m};

        featsq=[i,j,k];
        x3=Xtr(:,featsq);
        N3=normalizer("normal");
        nx3=N3.fit_transform(x3);

        opt.configure("method",method); ## Just change the method

        if method="batch"
        [ts3,errs]=opt.minimize(@logreg_loss,@logreg_gradloss,theta0(featsq),nx3,Y);
        thets=ts3{end};
          ttsb(v1,:)=ts3{end};
          v1=v1+1;
        endif
        if method="sgd"
        [ts3,errs]=opt.minimize(@logreg_loss,@logreg_gradloss,theta0(featsq),nx3,Y);
        thets=ts3{end};
          ttss(v2,:)=ts3{end};
          v2=v2+1;
        endif

        if  method="momentum"
        [ts3,errs]=opt.minimize(@logreg_loss,@logreg_gradloss,theta0(featsq),nx3,Y);
        thets=ts3{end};
          ttsm(v3,:)=ts3{end};
          v3=v3+1;
        endif

        py3=logreg_hyp(thets,nx3);
        err3=sum((py3>0.5)!=Y);
        tot3=100*(err3/rows(Y));

        if tot3<=comp
          comp=tot3;
          columna1=i;
          columna2=j;
        endif

        mins=min(x3);
        maxs=max(x3);

        e1=linspace(mins(1),maxs(1),50);
        e2=linspace(mins(2),maxs(2),50);

        endfor
      endfor
  endfor
endfor
figure(4,"name","Trayectoria de los parámetros durante el entrenamiento para tres métodos de optimización")


plot3(ttsb(1,:),ttsb(2,:),ttsb(3,:));%%%%batch
hold on;
plot3(ttss(1,:),ttss(2,:),ttss(3,:));%%%%sgd
hold on;
plot3(ttsm(1,:),ttsm(2,:),ttsm(3,:));%%%%momentum
hold on;
legend('Batch', 'SGD','Momentum');
xlabel('\theta_1');
ylabel('\theta_2');
zlabel('\theta_3');

