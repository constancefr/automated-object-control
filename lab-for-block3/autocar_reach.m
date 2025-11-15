% function [completed,res,tTotal] = autocar_reach

% Simulation time
params.tFinal = 2.5;
% Time step
params.dt = 0.1;
% Number of step
numSteps = params.tFinal / params.dt;

% action to take, decide by NN: [break idle accel]
actions = [-5, 0, 3]; 

% ======== Initial state of three cars ==============

% x1 = x_f (Front car displacement)
% x2 = v_f (Front car velocity)
% x3 = x_e (Ego d)
% x4 = v_e (Ego v)
% x5 = x_b (Back car d)
% x6 = v_b (Back car v)

% initial stats = 
% [x_front; v_front; x_ego; v_ego; x_back, v_back]

R0 = interval( ...
    [100; 5; 30; 5; 0; 5], ...
    [100+1; 5+0.1; 30+1; 5+0.1; 0+1; 5+0.1] ...
    );

params.R0_poly = zonotope(R0);

% load NN with CORA tool box
network = neuralNetwork.readONNXNetwork('nn_model.onnx');

% define the normalisation function
means = [68.479297, 69.437695, 0.393321, 0.718784]';
SDs   = [52.128595, 52.274740, 2.142253, 1.982537]';
normalise = @(x_unnorm) (x_unnorm - means) ./ SDs;

A_norm = diag(1 ./ SDs);
B_norm = -means ./ SDs;

% ======= Mapping states to NN input =========

    % d_front   = x(1) - x(3)   x_f - x_e
    % d_back    = x(3) - x(5)   x_e - x_b   
    % v_front   = x(2) - x(4)   v_f - v_e
    % v_back    = x(4) - x(6)   v_e - v_b

% X for NN input is given by:
% X = C_map*x + k_map

C_map = [ 1,  0, -1,  0,  0,  0;  % d_front
          0,  0,  1,  0, -1,  0;  % d_back
          0,  1,  0, -1,  0,  0;  % v_front
          0,  0,  0,  1,  0, -1]; % v_back

k_map = [0; 0; 0; 0];

% ======= System model ==========

% x_dot = A*x + B*u + C

% x1_dot = x2       (v_f)
% x2_dot = -2       (a_f = -2)
% x3_dot = x4       (v_e)
% x4_dot = u        (a_e = u)
% x5_dot = x6       (v_b)
% x6_dot = 0        (a_b = 0)

A = [0 1 0 0 0 0;
     0 0 0 0 0 0;
     0 0 0 1 0 0;
     0 0 0 0 0 0;
     0 0 0 0 0 1;
     0 0 0 0 0 0];

B = [0; 0; 0; 1; 0; 0];
c = [0; -2; 0; 0; 0; 0];

% System 1: Break (a_ego = -5)
c_break = c + B * actions(1);
sys_break = linearSys('break', A, [], c_break); 

% System 2: Idle (a_ego = 0)
c_idle = c + B * actions(2);
sys_idle  = linearSys('idle', A, [], c_idle);

% System 3: Accelerate (a_ego = 3)
c_accel = c + B * actions(3); 
sys_accel = linearSys('accel', A, [], c_accel);

sys_array = {sys_break, sys_idle, sys_accel};

% ======== Reachability parameters ==========

options_cont = struct();
options_cont.timeStep = 0.01; 
options_cont.alg = 'lin'; 
options_cont.taylorTerms = 10; 
options_cont.zonotopeOrder = 50;

options_nn = struct();
options_nn.nn.poly_method = "singh";

% ========= Unsafe set ===========

T_gap = 1.4; D_default = 4;

% Unsafe to front: D_rel_front < D_safe_front
% x(1) - x(3) < T_gap*x(4) + D_default  =>  x(1) - x(3) - T_gap*x(4) < D_default
H_unsafe_1 = [1, 0, -1, -T_gap, 0, 0];
k_unsafe_1 = D_default;

% Unsafe to back: D_rel_rear < D_safe_rear
% x(3) - x(5) < D_default
H_unsafe_2 = [0, 0, 1, 0, -1, 0];
k_unsafe_2 = D_default;

% (H*x <= k)
epsilon = 1e-6; % Safety margin
unsafeSet_front = halfspace(H_unsafe_1, k_unsafe_1 - epsilon);
unsafeSet_rear  = halfspace(H_unsafe_2, k_unsafe_2 - epsilon);

% ======== Reachability analysis =========

timerVal = tic;
Ri = params.R0_poly;    
R_all = {Ri};          
res = 'VERIFIED';      

for i = 1:numSteps

    R_nn_unnorm = C_map * Ri + k_map;
    R_nn_norm = A_norm * R_nn_unnorm + B_norm;
    
 
    logits_set = network.evaluate(R_nn_norm, options_nn);
    logits_int = interval(logits_set);
    
   
    action_next = -1; 

    for k = 1:3 
        k_ = [1:k-1, k+1:3]; 
        
        if all(supremum(logits_int(k_)) <= infimum(logits_int(k)))
            action_next = k; 
            break;
        end
    end

    if action_next == -1
        res = 'UNKNOWN';
        fprintf('Loop %d stop: Argmax unclear\n', i);
        disp(logits_int);
        break;
    end

    current_sys = sys_array{action_next};
    
    reach_params.R0 = Ri;
    reach_params.tFinal = params.dt; 
    
    R_step = reach(current_sys, reach_params, options_cont);
    
    Ri = R_step.timeInterval.set{end};
    R_all{end+1} = Ri;

    if isIntersecting(Ri, unsafeSet_front)
        res = 'VIOLATED (Front)';
        fprintf('Loop %d stop: Unsafe to front\n', i);
        break;
    elseif isIntersecting(Ri, unsafeSet_rear)
        res = 'VIOLATED (Rear)';
        fprintf('Loop %d stop: Unsafe to back\n', i);
        break;
    end

end

tComp = toc(timerVal);
tTotal = tComp; 
disp(res);

completed = true;


% ======== Ploting ==========

T_gap = 1.4; D_default = 10;
DM = [ 1,  0, -1,  0,      0,  0;  % 1. D_rel_f = x1 - x3
       0,  0,  0,  T_gap,  0,  0;  % 2. D_safe_f = T_gap*x4
       0,  0,  1,  0,     -1,  0;  % 3. D_rel_r = x3 - x5
       0,  0,  0,  0,      0,  0]; % 4. D_safe_r (constant part)
Db = [0; D_default; 0; D_default];

t_vec = 0:params.dt:((length(R_all)-1)*params.dt);

nPoints = length(R_all);
D_rel_f_min = zeros(1, nPoints); D_rel_f_max = zeros(1, nPoints);
D_safe_f_min = zeros(1, nPoints); D_safe_f_max = zeros(1, nPoints);
D_rel_r_min = zeros(1, nPoints); D_rel_r_max = zeros(1, nPoints);
D_safe_r_min = zeros(1, nPoints); D_safe_r_max = zeros(1, nPoints);

for i = 1:nPoints
    Ri = R_all{i};
    R_dist = DM * Ri + Db;
    R_dist_int = interval(R_dist);
    
    D_rel_f_min(i) = infimum(R_dist_int(1));
    D_rel_f_max(i) = supremum(R_dist_int(1));
    D_safe_f_min(i) = infimum(R_dist_int(2));
    D_safe_f_max(i) = supremum(R_dist_int(2));
    D_rel_r_min(i) = infimum(R_dist_int(3));
    D_rel_r_max(i) = supremum(R_dist_int(3));
    D_safe_r_min(i) = infimum(R_dist_int(4));
    D_safe_r_max(i) = supremum(R_dist_int(4));
end

figure; 
hold on; box on;
final_time_str = sprintf('%.1fs', t_vec(end));
title(sprintf('Front Safety (Result: %s @ t=%s)', res, final_time_str));

unsafeColor = [1, 0.8, 0.8]; 
safeColor = [0.8, 1, 0.8]; 

fill_safe_f_min = zeros(size(D_safe_f_min));

fill([t_vec, fliplr(t_vec)], [fill_safe_f_min, fliplr(D_safe_f_max)], ...
     unsafeColor, 'EdgeColor', 'none', 'DisplayName', 'D_{safe, front} (Threshold)');
     
fill([t_vec, fliplr(t_vec)], [D_rel_f_min, fliplr(D_rel_f_max)], ...
     safeColor, 'EdgeColor', 'none', 'DisplayName', 'D_{rel, front} (Actual)');

xlabel('Time (s)');
ylabel('Distance (m)');
legend('Location', 'northeast');

y_max_1 = max(max(D_rel_f_max), max(D_safe_f_max));
ylim([0, y_max_1 + 5]); 
hold off;

figure; 
hold on; box on;
title(sprintf('Rear Safety (Result: %s @ t=%s)', res, final_time_str));

rearSafeColor = [0.8, 0.8, 1];

fill_safe_r_min = zeros(size(D_safe_r_min));

fill([t_vec, fliplr(t_vec)], [fill_safe_r_min, fliplr(D_safe_r_max)], ...
     unsafeColor, 'EdgeColor', 'none', 'DisplayName', 'D_{safe, rear} (Threshold)');
     
fill([t_vec, fliplr(t_vec)], [D_rel_r_min, fliplr(D_rel_r_max)], ...
     rearSafeColor, 'EdgeColor', 'none', 'DisplayName', 'D_{rel, rear} (Actual)');

xlabel('Time (s)');
ylabel('Distance (m)');
legend('Location', 'northeast');

y_max_2 = max(max(D_rel_r_max), max(D_safe_r_max));
ylim([0, y_max_2 + 5]);
hold off;