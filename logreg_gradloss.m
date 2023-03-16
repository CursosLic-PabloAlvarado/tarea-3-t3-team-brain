% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2023 <Su Copyright AQUÍ>

% Loss function used in logistic regression
function grad=logreg_gradloss(theta,X,y)
  m = length(y); % número de ejemplos de entrenamiento
  h = logreg_hyp(theta,X); % calcular la hipótesis
  ##grad = (2/m)*(h-y); % calcular el gradiente de MSE
  grad = 2*((X' * (h-y)))'/m;
endfunction
