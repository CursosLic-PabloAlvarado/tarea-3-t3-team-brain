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
NXte=NX.fit_transform(Xte);


% show the original data
plot(NXtr(:,2),Y,'x');
xlabel("x");
ylabel("y");
axis([-0.5,1,-0.1,1.1]);
grid on;
hold on;


figure(2,"name","Loss evolution");
hold on;

function h=linreg_hyp(theta,X)
  % theta must be a column vector
  h=X*theta(:);
endfunction

function err=linreg_loss(theta,X,y)
  ## residuals
  r=y-linreg_hyp(theta,X);
  err=0.5*(r'*r); # OLS
endfunction

function grad=linreg_gradloss(theta,X,y)
  ## residuals
  h=linreg_hyp(theta,X);
  grad=sum((h-y).*X);
endfunction

## Initial configuration for the optimizer
opt=optimizer("method","sgd",
              "minibatch",11,
              "maxiter",600,
              "alpha",0.03);
###

theta0=rand(columns(NXtr),1)-0.5; ## Common starting point (column vector)

px=bsxfun(@power,linspace(-0.5,1,100)',0:4);

# test all optimization methods
methods={"batch","sgd","momentum"};
##methods={"batch"};

for m=1:numel(methods)
    method=methods{m};
    printf("Probando método '%s'.\n",method);
    msg=sprintf(";%s;",method); ## use method in legends

    ##try
      opt.configure("method",method); ## Just change the method
      if strcmp(method, "batch")
        [ts,errs]=opt.minimize(@logreg_loss,@logreg_gradloss,theta0,NXtr,Y);
        theta=ts{end};
        py=logreg_hyp(theta,px);
        [err1, num_errors1, percent_error1] = logreg_los(theta,NXtr,Y);
        printf("errores de entrenamiento: %d de %d (%.2f%%)\n", num_errors1, length(Y), percent_error1);
        [err11, num_errors11, percent_error11] = logreg_los(theta,NXte,Yte);
        printf("errores de prueba: %d de %d (%.2f%%)\n", num_errors11, length(Yte), percent_error11);
      endif
      if strcmp(method, "sgd")
        idx = randperm(size(NXtr, 1))(1:opt.minibatch);
        idx2 = randperm(size(NXte, 1))(1:opt.minibatch);
        X_batch = NXtr(idx, :);
        y_batch = Y(idx, :);
        X2_batch = NXte(idx2, :);
        y2_batch = Yte(idx2, :);
        [ts,errs]=opt.minimize(@logreg_loss,@logreg_gradloss,theta0,X_batch,y_batch);
        theta=ts{end};
        py=logreg_hyp(theta,px);
        [err2, num_errors2, percent_error2] = logreg_los(theta,X_batch,y_batch);
        printf("errores de entrenamiento: %d de %d (%.2f%%)\n", num_errors2, length(y_batch), percent_error2);
        [err22, num_errors22, percent_error22] = logreg_los(theta,X2_batch,y2_batch);
        printf("errores de prueba: %d de %d (%.2f%%)\n", num_errors22, length(y2_batch), percent_error22);
      endif
      if strcmp(method, "momentum")
        [ts,errs]=opt.minimize(@linreg_loss,@linreg_gradloss,theta0,NXtr,Y);
        theta=ts{end};
        py=linreg_hyp(theta,px);
        [err3, num_errors3, percent_error3] = logreg_los(theta,NXtr,Y);
        printf("errores de entrenamiento: %d de %d (%.2f%%)\n", num_errors3, length(Y), percent_error3);
        [err33, num_errors33, percent_error33] = logreg_los(theta,NXte,Yte);
        printf("errores de prueba: %d de %d (%.2f%%)\n", num_errors33, length(Yte), percent_error33);
      endif

      figure(1);
      plot(px(:,2),py,msg,"linewidth",2);

      figure(2);
      plot(errs,msg,"linewidth",2);
    ##catch
      ##printf("\n### Error detectado probando método '%s': ###\n %s\n\n",
             ##method,lasterror.message);
    ##end_try_catch
endfor

###
px2=bsxfun(@power,linspace(-0.5,1,100)',0:2);
ERBATCH=0;
ERSGD=0;
ERMOM=0;

for i = 1:length(theta0)-1
  for j = i+1:length(theta0)
    NXtr2=NXtr(:,[i j]);
    NXtr2=[ones(length(NXtr2),1) NXtr2];
    NXte2=NXte(:,[i j]);
    NXte2=[ones(length(NXte2),1) NXte2];
    theta2=theta(:,[1 i j]);



    for m=1:numel(methods)
    method=methods{m};
    printf("Probando método '%s'.\n",method);
    msg=sprintf(";%s;",method); ## use method in legends

    ##try
      opt.configure("method",method); ## Just change the method
      if strcmp(method, "batch")
        [err1, num_errors1, percent_error1] = logreg_los(theta2,NXtr2,Y);
        printf("usando columnas %d y %d \n", i, j);
        printf("errores de entrenamiento: %d de %d (%.2f%%)\n", num_errors1, length(Y), percent_error1);
        [err11, num_errors11, percent_error11] = logreg_los(theta2,NXte2,Yte);
        printf("errores de prueba: %d de %d (%.2f%%)\n", num_errors11, length(Yte), percent_error11);
        ERBATCH(end+1)=percent_error1;
      endif
      if strcmp(method, "sgd")
        idx = randperm(size(NXtr2, 1))(1:opt.minibatch);
        idx2 = randperm(size(NXte2, 1))(1:opt.minibatch);
        X_batch = NXtr2(idx, :);
        y_batch = Y(idx, :);
        X2_batch = NXte2(idx2, :);
        y2_batch = Yte(idx2, :);
        [err2, num_errors2, percent_error2] = logreg_los(theta2,X_batch,y_batch);
        printf("usando columnas %d y %d \n", i, j);
        printf("errores de entrenamiento: %d de %d (%.2f%%)\n", num_errors2, length(y_batch), percent_error2);
        [err22, num_errors22, percent_error22] = logreg_los(theta2,X2_batch,y2_batch);
        printf("errores de prueba: %d de %d (%.2f%%)\n", num_errors22, length(y2_batch), percent_error22);
        ERSGD(end+1)=percent_error2;
      endif
      if strcmp(method, "momentum")
        [err3, num_errors3, percent_error3] = logreg_los(theta2,NXtr2,Y);
        printf("usando columnas %d y %d \n", i, j);
        printf("errores de entrenamiento: %d de %d (%.2f%%)\n", num_errors3, length(Y), percent_error3);
        [err33, num_errors33, percent_error33] = logreg_los(theta2,NXte2,Yte);
        printf("errores de prueba: %d de %d (%.2f%%)\n", num_errors33, length(Yte), percent_error33);
        ERMOM(end+1)=percent_error3;
      endif
    ##catch
      ##printf("\n### Error detectado probando método '%s': ###\n %s\n\n",
             ##method,lasterror.message);
    ##end_try_catch
    endfor

  endfor

endfor

ERMOM=ERMOM(:,[2 3 4 5 6 7 8 9 10 11]);
ERSGD=ERSGD(:,[2 3 4 5 6 7 8 9 10 11]);
ERBATCH=ERBATCH(:,[2 3 4 5 6 7 8 9 10 11]);
[valorbatch, posicionbatch] = min(ERBATCH);
[valorsgd, posicionsgd] = min(ERSGD);
[valormom, posicionmom] = min(ERMOM);

figure(2);
xlabel("Iteration");
ylabel("Loss");0
grid on;


################################
