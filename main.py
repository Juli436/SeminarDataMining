import matplotlib.pyplot as plt
import data_generation as dg



if __name__ == '__main__':
    x, y = dg.get_data_from_sin(100, 0.1, 12.5)
    plt.scatter(x,y)
    plt.show()



