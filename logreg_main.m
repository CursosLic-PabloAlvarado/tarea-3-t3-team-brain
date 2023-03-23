% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2023 <Su Copyright AQUÍ>

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
for m=1:numel(methods)
  method=methods{m};
  printf("Probando método '%s'.\n",method);
  msg=sprintf(";%s;",method); ## use method in legends

  try
    opt.configure("method",method); ## Just change the method
    [ts,errs]=opt.minimize(@logreg_loss,@logreg_gradloss,theta0,NXtr,Y);
    theta=ts{end}
    py=logreg_hyp(theta,NXte);
    err=sum((py>0.5)!=Yte);
    tot=100*(err/rows(Yte));
    ##[err1, num_errors1, percent_error1] = logreg_los(theta,NXtr,Y);
    ##printf("errores de entrenamiento: %d de %d (%.2f%%)\n", num_errors1, length(Y), percent_error1);
    ##[err11, num_errors11, percent_error11] = logreg_los(theta,NXte,Yte);
    printf("errores de prueba: %d de %d (%.2f%%)\n", err, length(Yte), tot);


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
comp=1;
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

    py=logreg_hyp(theta2,NXte);
    err=sum((py>0.5)!=Yte);
    tot=100*(err/rows(Yte));

    if tot<comp
      comp=tot;
      columna1=i;
      columna2=j;
    endif

    mins=min(x2);
    maxs=max(x2);

    e1=linspace(mins(1),maxs(1),50);
    e2=linspace(mins(2),maxs(2),50);
  endfor

endfor

printf("el menor error obtenido es: %d al evaluar las columnas %d %d\n", comp, columna1, columna2);

[ee1,ee2]=meshgrid(e1,e2);
x2test=N2.transform([ee1(:) ee2(:)]);

ytest=logreg_hyp(theta,x2test);

figure(2,"name","Probabilidad")
surf(ee1,ee2,reshape(ytest,size(ee1)));
hold on;

contour3(ee1,ee2,reshape(ytest,size(ee1)),[0.25,0.5,0.75],"linewidth",3,"linecolor","black");

################################
