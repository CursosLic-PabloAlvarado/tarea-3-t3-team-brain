% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2023 <Team brAIn>

% Loss function used in logistic regression
function err=logreg_loss(theta,X,y)

  m = length(y); % número de ejemplos de entrenamiento
  h = logreg_hyp(theta,X); % calcular la hipótesis
  err = (1 / (m)) * sum((h - y) .^ 2); % calcular el error de MSE

endfunction
