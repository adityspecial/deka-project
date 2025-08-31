% Import data from Excel file
data = xlsread('output.xlsx'); % Replace 'output.xlsx' with the actual file name

% Extract x and y data
x = data(:, 2); % Assuming x data is in the second column
y = data(:, 1); % Assuming y data is in the first column

% Set background color to dark blue
set(gcf, 'color', [0, 0, 0.5]); % [R, G, B] with values between 0 and 1

% Plot the original data
plot(x, y, 'b.-'); % Plot with blue line connecting points and dots at data points
hold on; % Hold the plot

% Plot the flipped data
plot(-x, y, 'r.-'); % Plot with red line connecting points and dots at data points

% Set the x-axis limits to include both sets of data
x_limit = max(abs(x));
xlim([-x_limit, x_limit]);

% Fill the entire plot area with dark blue
x_limits = xlim;
y_limits = ylim;
fill([x_limits(1), x_limits(2), x_limits(2), x_limits(1)], [y_limits(1), y_limits(1), y_limits(2), y_limits(2)], [0,0,0.5] , 'EdgeColor', 'none');

% Fill the area under the original data with dark red
fill([x; flipud(x)], [y; zeros(size(y))], [0.5, 0, 0], 'EdgeColor', 'none');

% Fill the area under the flipped data with dark red
fill([-x; flipud(-x)], [y; zeros(size(y))], [0.5, 0, 0], 'EdgeColor', 'none');

% Label the axes
xlabel('--');
ylabel('--');
title('--');
grid on; % Turn on grid

% Add a legend
%legend({'Original', 'Flipped'}, 'Location', 'best');

hold off; % Release the hold

% Rotate the entire figure 90 degrees anticlockwise
set(gcf, 'Position', get(0, 'Screensize')); % Maximize figure window
view(0, 90); % Rotate the figure 90 degrees anticlockwise
