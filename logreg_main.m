% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2023 <Su Copyright AQUÍ>

% Logistic regression testbench

clear all; close all;

[Xtr,Ytr,Xte,Yte,names] = loadpenguindata("sex");

Xtr = [ones(length(Xtr),1) Xtr];
Y = Ytr(:,1);
NX=normalizer("normal");
NXtr=NX.fit_transform(Xtr);

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
              "minibatch",10,
              "maxiter",600,
              "alpha",0.003);
###

theta0=rand(columns(Xtr),1)-0.5; ## Common starting point (column vector)

px=bsxfun(@power,linspace(-0.5,1,100)',0:4);

# test all optimization methods
methods={"batch","sgd","momentum"};
for m=1:numel(methods)
  method=methods{m};
  printf("Probando método '%s'.\n",method);
  msg=sprintf(";%s;",method); ## use method in legends

  try
    opt.configure("method",method); ## Just change the method
    if strcmp(method, "batch")
      [ts,errs]=opt.minimize(@logreg_loss,@logreg_gradloss,theta0,NXtr,Y);
      theta=ts{end};
      py=logreg_hyp(theta,px);
    endif
    if strcmp(method, "sgd")
      idx = randperm(size(NXtr, 1))(1:opt.minibatch);
      X_batch = NXtr(idx, :);
      y_batch = Y(idx, :);
      [ts,errs]=opt.minimize(@logreg_loss,@logreg_gradloss,theta0,X_batch,y_batch);
      theta=ts{end};
      py=logreg_hyp(theta,px);
    endif
    if strcmp(method, "momentum")
      [ts,errs]=opt.minimize(@linreg_loss,@linreg_gradloss,theta0,NXtr,Y);
      theta=ts{end};
      py=linreg_hyp(theta,px);
    endif

    figure(1);
    plot(px(:,2),py,msg,"linewidth",2);

    figure(2);
    plot(errs,msg,"linewidth",2);
  catch
    printf("\n### Error detectado probando método '%s': ###\n %s\n\n",
           method,lasterror.message);
  end_try_catch
endfor

figure(2);
xlabel("Iteration");
ylabel("Loss");
grid on;

################################
