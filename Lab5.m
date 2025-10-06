%% ISC 3222 - Lab 5: Particle Swarm Optimization
% This script implements a Particle Swarm Optimizer to find the maximum
% of a given evaluation function and visualizes the process.

% Clear workspace, command window, and close all figures
clear;
clc;
close all;

%% Exercise 1: Create the Evaluation Function
% We define the evaluation function f(x, y) as a vectorized anonymous function
% to easily compute scores for all particles at once.
f = @(x,y) -x.^2 + 2.*cos(pi.*x) - y.^2 - sin(pi.*y);


%% Exercise 2: Plot the Evaluation Function
% To visualize the problem space, we create a 3D surface plot.

% Generate 100 linearly spaced points for x and y axes
x_grid = linspace(-4, 4, 100);
y_grid = linspace(-4, 4, 100);

% Create a 2D grid of x and y values
[X, Y] = meshgrid(x_grid, y_grid);

% Calculate the z value (score) for each point on the grid
Z = f(X, Y);

% Plot the surface
figure('Name', 'Evaluation Function Surface');
surf(X, Y, Z);
shading interp; % Smooths the coloring
colorbar;
title('3D Surface of the Evaluation Function f(x,y)');
xlabel('Parameter x');
ylabel('Parameter y');
zlabel('Score f(x,y)');


%% Exercise 3: Implement the Particle Swarm Optimizer
% Here we set up and run the PSO algorithm.

% --- PSO Parameters ---
n = 10;             % Number of individuals (particles)
n_steps = 100;      % Number of optimization steps
bounds = [-4, 4];   % Search space bounds for x and y
cp = 2.0;           % Cognitive (personal) coefficient
cg = 2.0;           % Social (group) coefficient

% Inertia weight 'w' decreases from 0.8 to 0 over the 100 steps
w = linspace(0.8, 0, n_steps);

% --- Initialization ---
% Initialize positions randomly within the search space
% pos is a 10x2 matrix: [x1, y1; x2, y2; ...]
pos = (bounds(2) - bounds(1)) * rand(n, 2) + bounds(1);

% Initialize velocities to zero for all particles
vel = zeros(n, 2);

% Initialize personal best positions and scores
bstPos = pos;
bstScores = f(pos(:,1), pos(:,2));

% Find the initial global best position and score
[max_score, idx] = max(bstScores);
bstScoreG = max_score;
bstPosG = pos(idx, :);

% Store history for plotting
pos_history = zeros(n, 2, n_steps);

% --- Setup for Movies (Exercises 4, 5, 6) ---
% Setup for 2D movie
fig2D = figure('Name', '2D PSO Animation');
vid2D = VideoWriter('pso_2d_animation.mp4', 'MPEG-4');
open(vid2D);

% Setup for 3D movie
fig3D = figure('Name', '3D PSO Animation');
set(fig3D, 'Position', [50, 50, 900, 700]); % Make window larger
vid3D = VideoWriter('pso_3d_animation.mp4', 'MPEG-4');
open(vid3D);


% --- Main Optimization Loop ---
for t = 1:n_steps
    
    % Generate random numbers for stochastic velocity update
    rp = rand(n, 2);
    rg = rand(n, 2);
    
    % Update velocity using the PSO formula
    % Note: Modern MATLAB implicitly expands bstPosG to match dimensions
    vel = w(t).*vel ...
        + cp.*rp.*(bstPos - pos) ...
        + cg.*rg.*(bstPosG - pos);
        
    % Update position
    pos = pos + vel;
    
    % --- Boundary Handling ---
    % If particles go beyond the [-4, 4] bounds, clamp them to the boundary.
    pos(pos < bounds(1)) = bounds(1);
    pos(pos > bounds(2)) = bounds(2);
    
    % Evaluate scores at new positions
    new_scores = f(pos(:,1), pos(:,2));
    
    % Update personal best positions
    update_idx = new_scores > bstScores;
    bstScores(update_idx) = new_scores(update_idx);
    bstPos(update_idx,:) = pos(update_idx,:);
    
    % Update global best position
    [max_step_score, idx] = max(bstScores);
    if max_step_score > bstScoreG
        bstScoreG = max_step_score;
        bstPosG = bstPos(idx, :);
    end
    
    % Store current positions for the movie
    pos_history(:,:,t) = pos;
    
    % --- Frame Generation for 2D Movie (Exercise 4) ---
    figure(fig2D);
    clf;
    contour(X, Y, Z, 20); % Background context
    hold on;
    plot(pos(:,1), pos(:,2), 'b.', 'MarkerSize', 20); % Particles
    plot(bstPosG(1), bstPosG(2), 'r*', 'MarkerSize', 15); % Global best
    hold off;
    xlim(bounds);
    ylim(bounds);
    title(sprintf('2D Particle Positions (Step: %d / %d)', t, n_steps));
    xlabel('Parameter x');
    ylabel('Parameter y');
    legend('Function Contour', 'Particles', 'Global Best');
    drawnow;
    
    % Write frame to 2D video
    frame = getframe(fig2D);
    writeVideo(vid2D, frame);

    % --- Frame Generation for 3D Movie (Exercises 5 & 6) ---
    figure(fig3D);
    clf;
    surf(X, Y, Z, 'FaceAlpha', 0.6, 'EdgeColor', 'none'); % Surface
    hold on;
    % Get z-values for current particle positions
    z_pos = f(pos(:,1), pos(:,2));
    plot3(pos(:,1), pos(:,2), z_pos, 'k.', 'MarkerSize', 25); % Particles on surface
    plot3(bstPosG(1), bstPosG(2), bstScoreG, 'r*', 'MarkerSize', 15, 'LineWidth', 2); % Global best
    hold off;
    title(sprintf('3D Particle Positions (Step: %d / %d)', t, n_steps));
    xlabel('Parameter x');
    ylabel('Parameter y');
    zlabel('Score f(x,y)');
    legend('', 'Particles', 'Global Best');
    zlim([min(Z(:))-2, max(Z(:))+2]); % Set z-limits
    
    % Change the viewing perspective for an interesting movie
    az = 120 * cos(t / n_steps * 2 * pi) - 30; % Azimuth rotates
    el = 25; % Elevation is fixed
    view(az, el); % Apply the new view
    
    drawnow;
    
    % Write frame to 3D video
    frame = getframe(fig3D);
    writeVideo(vid3D, frame);
    
end

% --- Finalization ---
% Close the video files
close(vid2D);
close(vid3D);

% Display the final results in the Command Window
fprintf('\n--- Optimization Complete ---\n');
fprintf('Found maximum score: %.4f\n', bstScoreG);
fprintf('at parameters (x, y) = (%.4f, %.4f)\n', bstPosG(1), bstPosG(2));
disp('Movies "pso_2d_animation.mp4" and "pso_3d_animation.mp4" have been saved.');