function [completed,res,tTotal] = nn_egocar_reach(varargin)
% nn_egocar_reach - Reachability for a 1D ego car controller (ONNX)
% Adapted from Ocean's code!
%
% State (6D): x = [x_f; v_f; x_e; v_e; x_b; v_b]
% NN inputs (4D): [d_front; d_back; v_front; v_back]
% Actions: discrete accelerations -> map to linear systems
%
% Usage:
%   [completed,res,tTotal] = nn_egocar_reach('onnx','nn_model.onnx', 'dt',0.1, 'tFinal',2.5)
%
% Optional Name-Value inputs:
%  'onnx'   - path to ONNX file (default 'nn_model.onnx')
%  'dt'     - control timestep in seconds (default 0.1)
%  'tFinal' - total simulation time in seconds (default 2.5)

% Parse inputs ----------------------------------
[onnxFile, dt, tFinal] = setDefaults(varargin);
numSteps = round(tFinal/dt);

fprintf('nn_egocar_reach: ONNX=%s dt=%.3f tFinal=%.2f', onnxFile, dt, tFinal);

% Default dynamics params -----------------------
actions = [-5, 0, 3]; % [brake, idle, accel]

% Initial set -----------------------------------
x_f = interval(100,101); v_f = interval(5,5.1);
x_e = interval(30,31);  v_e = interval(5,5.1);
x_b = interval(0,1);    v_b = interval(5,5.1);
R0 = interval([x_f.inf; v_f.inf; x_e.inf; v_e.inf; x_b.inf; v_b.inf], [x_f.sup; v_f.sup; x_e.sup; v_e.sup; x_b.sup; v_b.sup]);
% R0 = interval([100; 5; 30; 5; 0; 5], [110; 6; 31; 6; 1; 6]);


% convert to zonotope for CORA reach
Ri = zonotope(R0);
R_all = {Ri};

% ONNX import & normalization -------------------
network = neuralNetwork.readONNXNetwork(onnxFile);

means = [68.479297, 69.437695, 0.393321, 0.718784]';
stds   = [52.128595, 52.274740, 2.142253, 1.982537]';

A_norm = diag(1./stds);
B_norm = -means ./ stds;

% Input mapping from 6D state to 4D NN input: X = C_map * x + k_map
C_map = [ 1,0,-1,0,0,0;  % d_front = x_f - x_e
          0,0,1,0,-1,0;  % d_back  = x_e - x_b
          0,1,0,-1,0,0;  % v_front = v_f - v_e
          0,0,0,1,0,-1]; % v_back  = v_e - v_b
k_map = zeros(4,1);

% Compose mapping and normalization:
%   X_norm = A_norm*(C_map*x + k_map) + B_norm
% so overall linear mapping:
%   X_norm = (A_norm*C_map)*x + (A_norm*k_map + B_norm)
M_nn = A_norm * C_map;
k_nn = A_norm * k_map + B_norm;

% System matrices --------------------------------
% State x = [x_f, v_f, x_e, v_e, x_b, v_b]
% We assume worst-case:
%   - front car permanently brakes at some constant value
%   - back car permanently accelerates at some constant value
% Continuous-time affine dynamics: x_dot = A*x + B*u + c
A = [0 1 0 0 0 0; % applied to state vector -> isolates velocities
     0 0 0 0 0 0;
     0 0 0 1 0 0;
     0 0 0 0 0 0;
     0 0 0 0 0 1;
     0 0 0 0 0 0];
B = [0;0;0;1;0;0]; % applied to input vetor -> NN only has control over ego car's acceleration
c_base = [0; -2; 0; 0; 0; 0]; % front car accelerates -2 m/s^2, back car 0

% Build linearSys objects for each discrete action (using CORA linearSys)
sys_array = cell(1,3);
for i=1:3
    acc = actions(i);
    c_i = c_base + B * acc; % a_ego = acc
    sys_array{i} = linearSys(sprintf('act_%d',i), A, [], c_i);
end

% Reachability params -----------------------------
options_cont = struct();
options_cont.timeStep = 0.01; % internal ODE step for reach
% options_cont.timeStep = 0.05;
options_cont.alg = 'lin';
options_cont.taylorTerms = 4;
options_cont.zonotopeOrder = 50;

options_nn = struct();
options_nn.nn.poly_method = 'singh';

% Unsafe sets -----------------------------------
% Front safety: x_f - x_e - T_gap*v_e < D_default  -> H_unsafe_1 * x <= k_unsafe_1
T_gap = 1.4; D_default = 4; epsilon = 1e-6;
H_unsafe_1 = [1,0,-1,-T_gap,0,0]; k_unsafe_1 = D_default - epsilon;
H_unsafe_2 = [0,0,1,0,-1,0];    k_unsafe_2 = D_default - epsilon; % back safety
unsafe_front = halfspace(H_unsafe_1, k_unsafe_1);
unsafe_back  = halfspace(H_unsafe_2, k_unsafe_2);

% Reachability main loop ------------------------
res = 'VERIFIED';

timerVal = tic;
for step = 1:numSteps
    R_nn_unnorm = M_nn * Ri + k_nn;

    logits_set = network.evaluate(R_nn_unnorm, options_nn);
    logits_int = interval(logits_set);

    % argmax check on intervals
    action_next = -1;
    for k = 1:3
        others = setdiff(1:3,k);
        if all(supremum(logits_int(others)) <= infimum(logits_int(k)))
            action_next = k; break;
        end
    end
    % action_next = 1; % force brake at each timestep

    if action_next == -1
        res = 'UNKNOWN'; fprintf('Step %d: Ambiguous NN output -> UNKNOWN', step); break;
    end

    % run continuous reach for this control action over dt
    current_sys = sys_array{action_next};
    reach_params.R0 = Ri;
    reach_params.tFinal = dt;

    R_step = reach(current_sys, reach_params, options_cont);
    % final reachable set of this step
    Ri = R_step.timeInterval.set{end};
    R_all{end+1} = Ri;

    % check unsafe intersection
    if isIntersecting(Ri, unsafe_front)
        res = 'VIOLATED (Front)'; fprintf('Step %d: Unsafe front -> VIOLATED', step); break;
    elseif isIntersecting(Ri, unsafe_back)
        res = 'VIOLATED (Back)'; fprintf('Step %d: Unsafe back -> VIOLATED', step); break;
    end
end

tComp = toc(timerVal);
tTotal = tComp;
completed = true;

% Plotting -------------------------------------
plotReachabilityResults(R_all, dt, res, T_gap, 10);

% helpers --------------------------------------
function [onnxFile, dt, tFinal] = setDefaults(args)
onnxFile = 'nn_model.onnx'; dt = 0.1; tFinal = 2.5;
% onnxFile = 'nn_model.onnx'; dt = 0.1; tFinal = 5;
if isempty(args), return; end
for i=1:2:length(args)
    key = lower(args{i}); val = args{i+1};
    switch key
        case 'onnx', onnxFile = val;
        case 'dt', dt = val;
        case 'tfinal', tFinal = val;
    end
end
end

function plotReachabilityResults(Rcells, dt, resultStr, T_gap, D_default)
% plots front/back distances vs safe thresholds and saves the figures
n = length(Rcells);
time = 0:dt:(n-1)*dt;
D_rel_f_min = zeros(1,n); D_rel_f_max = zeros(1,n);
D_safe_f_min = zeros(1,n); D_safe_f_max = zeros(1,n);
D_rel_r_min = zeros(1,n); D_rel_r_max = zeros(1,n);
D_safe_r_min = zeros(1,n); D_safe_r_max = zeros(1,n);

for ii = 1:n
    Rii = Rcells{ii};
    % Map to distances and threshold
    DM = [1,0,-1,0,0,0; 0,0,0,T_gap,0,0; 0,0,1,0,-1,0; 0,0,0,0,0,0];
    Db = [0; D_default; 0; D_default];
    Rdist = DM * Rii + Db;
    RdistI = interval(Rdist);
    D_rel_f_min(ii) = infimum(RdistI(1)); D_rel_f_max(ii) = supremum(RdistI(1));
    D_safe_f_min(ii) = infimum(RdistI(2)); D_safe_f_max(ii) = supremum(RdistI(2));
    D_rel_r_min(ii) = infimum(RdistI(3)); D_rel_r_max(ii) = supremum(RdistI(3));
    D_safe_r_min(ii) = infimum(RdistI(4)); D_safe_r_max(ii) = supremum(RdistI(4));
end

% Front plot
figure; hold on; box on;
fill([time, fliplr(time)], [D_safe_f_min, fliplr(D_safe_f_max)], [1,0.85,0.85], 'EdgeColor','none');
fill([time, fliplr(time)], [D_rel_f_min, fliplr(D_rel_f_max)], [0.85,1,0.85], 'EdgeColor','none');
plot(time, D_rel_f_min, 'k--'); plot(time, D_rel_f_max, 'k-');
title(sprintf('Front Safety (Result: %s)', resultStr)); xlabel('Time (s)'); ylabel('Distance (m)');
legend({'Unsafe','Reachable'}, 'Location','northeast');

% Save front plot
saveas(gcf, sprintf('FrontSafety_%s.png', resultStr));

% Back plot
figure; hold on; box on;
% Fix D_safe_r to produce a visible band
D_safe_r_min = zeros(size(D_safe_r_min));
D_safe_r_max = D_default * ones(size(D_safe_r_max));

fill([time, fliplr(time)], [D_safe_r_min, fliplr(D_safe_r_max)], [1,0.85,0.85], 'EdgeColor','none');
fill([time, fliplr(time)], [D_rel_r_min, fliplr(D_rel_r_max)], [0.85,0.85,1], 'EdgeColor','none');
plot(time, D_rel_r_min, 'k--'); plot(time, D_rel_r_max, 'k-');
title(sprintf('Back Safety (Result: %s)', resultStr)); xlabel('Time (s)'); ylabel('Distance (m)');
legend({'Unsafe','Reachable'}, 'Location','northeast');

% Save back plot
saveas(gcf, sprintf('BackSafety_%s.png', resultStr));

end

end

