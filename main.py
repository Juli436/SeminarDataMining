import matplotlib.pyplot as plt
import data_generation as dg



if __name__ == '__main__':
    x, y, betas = dg.get_data(100, 2, 0.1, 0, 1, 10)
    print(betas)
    plt.scatter(x[1], y)
    plt.show()



