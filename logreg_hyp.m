% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2023 <Su Copyright AQUÍ>

% Hypothesis function used in logistic regression
function h=logreg_hyp(theta,X)
  assert(rows(theta)==1,"theta tiene que ser un vector fila");
  h = 1 ./ (1 + exp(-(X * theta')));

endfunction
