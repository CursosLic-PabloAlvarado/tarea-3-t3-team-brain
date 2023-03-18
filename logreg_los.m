## Copyright (C) 2023 Gollo
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {} {@var{retval} =} logreg_los (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Gollo <Gollo@DESKTOP-TV0ASU4>
## Created: 2023-03-17

function [err, num_errors, percent_error] = logreg_los(theta,X,y)

  m = length(y); % número de ejemplos de entrenamiento
  h = logreg_hyp(theta,X); % calcular la hipótesis
  err = (1 / (m)) * sum((h - y) .^ 2); % calcular el error de MSE

  % Calcular el número y el porcentaje de errores
  predictions = round(h); % Redondear las predicciones a 0 o 1
  num_errors = sum(predictions != y); % Contar el número de predicciones incorrectas
  percent_error = (num_errors / m) * 100; % Calcular el porcentaje de errores

endfunction
