% Copyright (C) 2022-2023 Pablo Alvarado
% EL5857 Aprendizaje Automático
% Tarea 3
% (C) 2023 <Team brAIn>

% Hypothesis function used in logistic regression
function h=logreg_hyp(theta,X)
  h = 1 ./ (1 + exp(-(X*theta(:))));

endfunction
