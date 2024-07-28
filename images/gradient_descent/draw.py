import numpy as np
import matplotlib.pyplot as plt


def function_to_minimize(x):
    """Function: f(x) = x^2 - 2x + 3"""
    return x ** 2 - 2 * x + 3


def gradient(x):
    """Gradient of the function: f'(x) = 2x - 2"""
    return 2 * x - 2


def plot_gradient_descent():
    # Create a range of x values
    x = np.linspace(-2, 4, 400)
    y = function_to_minimize(x)

    # Plot the function
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='$f(x) = x^2 - 2x + 3$', color='blue', zorder=1)

    # Points to plot the gradient
    points = np.array([1.2, 1.5, 2, 2.5, 3, 3.5, 4])

    for point in points:
        grad = gradient(point)
        # Plot the point
        plt.plot(point, function_to_minimize(point), 'bo', zorder=2, markersize=5)
        # Plot the gradient vector
        a = 1 if grad < 0 else -1
        b = -abs(grad)
        print(a, b)
        # a = 3 * a / math.sqrt(a*a + b*b)
        # b = 3 * b / math.sqrt(a*a + b*b)
        print(a, b)

        plt.quiver(point, function_to_minimize(point), a, b, color='red', angles='xy', scale_units='xy', scale=5,
                   zorder=3)

    plt.ylim(bottom=-1, top=13)  # Adjust these values as needed
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title('Minh họa giảm độ dốc')
    plt.legend()
    plt.grid(True)
    plt.savefig('gradient_descent_plot.png', format='png')
    plt.show()


# Run the function to plot the gradient descent visualization
plot_gradient_descent()
