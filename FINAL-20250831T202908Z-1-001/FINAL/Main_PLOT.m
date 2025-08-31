% Import data from Excel file
data = xlsread('output.xlsx'); % Replace 'output.xlsx' with the actual file name

% Extract x and y data
x = data(:, 2); % Assuming x data is in the second column
y = data(:, 1); % Assuming y data is in the first column

% Plot the original data
plot(x, y, 'b.-'); % Plot with blue line connecting points and dots at data points
hold on; % Hold the plot

% Label the axes
xlabel('X Axis');
ylabel('Y Axis');
title('Plot of X vs Y');
grid on; % Turn on grid

% Rotate the plot 90 degrees anticlockwise
set(gca, 'XLimMode', 'auto', 'YLimMode', 'auto');
view(90, -90);

% Flip the plot along the y-axis
xlim([-max(x), max(x)]);

% Plot the flipped data
plot(-x, y, 'r.-'); % Plot with red line connecting points and dots at data points

% Label the axes
xlabel('X Axis');
ylabel('Y Axis');
title('Plot of X vs Y (Rotated and Flipped)');
grid on; % Turn on grid

% Add a legend
legend({'Original', 'Flipped'}, 'Location', 'best');

hold off; % Release the hold

% Rotate the entire figure 90 degrees anticlockwise
set(gcf, 'Position', get(0, 'Screensize')); % Maximize figure window
view(0, 90); % Rotate the figure 90 degrees anticlockwise
