% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2023 <Su Copyright AQUÍ>

% Loss function used in logistic regression
function grad=logreg_gradloss(theta,X,y)
  m = length(y); % número de ejemplos de entrenamiento
  h = 1 ./ (1 + exp(-(X * theta'))); % calcular la hipótesis
  %grad = ((X' * (h - y))'/m); % calcular el gradiente de MSE
  grad = (2/m)*(X'h - y);
endfunction
