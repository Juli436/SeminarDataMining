import matplotlib.pyplot as plt
import numpy as np
import bayesian_estimator


import frequentistEstimator as fe
import parse_data as data



def get_error(test_x, test_y, est_beta):
    est_y = np.dot(est_beta, test_x)
    error = test_y - est_y
    error = np.square(error)
    sum_error = np.sum(error)
    return np.sqrt(sum_error/len(test_y))




def get_cov(n, sig):
    cov = np.ndarray((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                cov[i][j] = sig[i] ** 2
            else:
                cov[i][j] = 0
    return cov



def plot_dims(mean, var):
    num = 1000
    x = np.linspace(-1.5, 1.5, num)
    y = bayesian_estimator.gaussian_var(mean, x, var)

    plt.plot(x, y)

    plt.title("Posterior over beta_2_1")
    plt.xlabel("beta_2_1")
    plt.ylabel("Probability density")



    plt.show()




def eval_data():
    x, y = data.get_data_cat()
    betas_est = fe.ml_estimator(x,y)
    print(betas_est)

    sig_betas = np.full(12, 1)
    mean_betas = np.zeros(12)
    cov = get_cov(12, sig_betas)
    mean, var = bayesian_estimator.bayesian_estimate_with_gaussian_prior(x,y,0.1,mean_betas, cov)
    print(mean)
    plt.imshow(np.absolute(var))
    plt.colorbar()
    plt.show()
    plot_dims(mean[3], var[3][3])


#print mse of parameters depending on number of samples
def plot_mse():
    sig_betas = np.full(12, 1)
    mean_betas = np.zeros(12)
    mean_betas = np.array(mean_betas)
    cov = get_cov(12, sig_betas)

    n = 80
    errors_ml = np.zeros(n)
    errors_bayes = np.zeros(n)
    num_samples = np.zeros(n)
    trials_to_avg = 5

    for i in range(0, n):
        trials_to_avg_ml = trials_to_avg
        for j in range(0, trials_to_avg):
            samples = (i + 1) * (1200 // n) + 125
            num_samples[i] = samples
            x, x_test, y, y_test = data.get_data_cat_test(samples)
            try:  # if matrix is not invertible
                betas_est = fe.ml_estimator(x, y)
            except:
                try:
                    x, x_test, y, y_test = data.get_data_cat_test(samples)
                    betas_est = fe.ml_estimator(x, y)
                except:
                    try:
                        x, x_test, y, y_test = data.get_data_cat_test(samples)
                        betas_est = fe.ml_estimator(x, y)
                    except:
                        continue

            mean, var = bayesian_estimator.bayesian_estimate_with_gaussian_prior(x, y, 0.1, mean_betas, cov)
            errors_bayes[i] += get_error(x_test, y_test, mean)
            er_ml = get_error(x_test, y_test, betas_est)
            if er_ml<1e10:
                errors_ml[i] += get_error(x_test, y_test, betas_est)
            else:
                trials_to_avg_ml -= 1
        errors_bayes[i] /= trials_to_avg
        errors_ml[i] /= trials_to_avg_ml

    plt.plot(num_samples, errors_ml, label="ML Estimate")
    plt.plot(num_samples, errors_bayes, label="MAP Estimate")
    plt.xlabel("Number of samples in training data")
    plt.ylabel("Root mean square error")
    plt.title("Average root mean square error as a function of the number of samples\n across 5 trials/number of samples")
    plt.legend()
    plt.show()
    print(errors_ml)
    print(errors_bayes)

def train_and_test():
    x, x_test, y, y_test = data.get_data_cat_test(1200)
    betas_est = fe.ml_estimator(x, y)
    print(betas_est)

    sig_betas = np.full(12, 1)
    sig_betas[0]= 1
    mean_betas = np.zeros(12)
    mean_betas = np.array(mean_betas)
    cov = get_cov(12, sig_betas)

    mean, var = bayesian_estimator.bayesian_estimate_with_gaussian_prior(x, y, 0.1, mean_betas, cov)
    print(mean)
    error_ml = get_error(x_test, y_test, betas_est)
    error_bayes = get_error(x_test, y_test, mean)
    print("Error ML: "+ str(error_ml))
    print("Error Bayes: " + str(error_bayes))
    #x = ["beta_0", "very\nwell", "somewhat\nwell", "not very\nwell", "not at\nall well", "very\nworried", "worried", "not too\nworried", "not at\nall worried", "too\nhigh", "reasonable", "too\nlow"]
    x_label = ["0", "1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "2_4", "3_1", "3_2", "3_3"]
    plt.bar(x_label, mean)
    #plt.title("Coefficients calculated with MAP estimate\nwhen only considering whether person wants less climate policies")
    plt.title("Coefficients calculated with MAP estimate")
    plt.show()

    fig, ax = plt.subplots(1, 1)
    img = ax.imshow(np.absolute(var), extent=[0, 12, 0, 12])

    ax.set_xticks(np.linspace(0.5,11.5,12))
    x_label_list = ["0", "1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3", "2_4", "3_1", "3_2", "3_3"]
    ax.set_xticklabels(x_label_list)

    ax.set_yticks(np.linspace(0.5, 11.5, 12))
    x_label_list.reverse()

    ax.set_yticklabels(x_label_list)
    fig.colorbar(img)
    plt.ylabel("Value of coefficient")
    plt.xlabel("Coefficient")

    plt.title("Absolute Values of Covariance Matrix of the Coefficients")
    plt.show()
    plot_dims(mean[5], var[5][5])

    mean_pred, var_pred  = bayesian_estimator.predict([1, 0,1,0,0,0, 1, 0, 0, 1 ,0, 0], mean, var, 0.1)
    y_plot = np.linspace(-1, 1, 1000)
    p_plot_1 = bayesian_estimator.gaussian_var(mean_pred, y_plot, var_pred)
    plt.plot(y_plot, p_plot_1, label ="Estimate including noise")
    mean_pred, var_pred = bayesian_estimator.predict([1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], mean, var)
    p_plot_2 = bayesian_estimator.gaussian_var(mean_pred, y_plot, var_pred)
    plt.plot(y_plot, p_plot_2, label= "Estimate excluding noise")
    plt.legend()
    plt.title("Predictive distribution")
    plt.ylabel("p(y)")
    plt.xlabel("y")
    plt.show()
    # Creating figure and axis objects using subplots()
    fig, ax = plt.subplots(figsize=[9, 7])

    # Plotting the firts line with ax axes
    ax.plot(y_plot, p_plot_1,
            color='green', label ="Estimate including noise")
    plt.xticks(rotation=60)
    ax.set_xlabel('y', fontsize=14)
    ax.set_ylabel('p(y) when including noise', color='green', fontsize=14 )


    # Create a twin axes ax2 using twinx() function
    ax2 = ax.twinx()

    # Now, plot the second line with ax2 axes
    ax2.plot(y_plot, p_plot_2,
             color='purple', label ="Estimate excluding noise")

    ax2.set_ylabel('p(y) when excluding noise', color='purple', fontsize=14 )
    plt.title("Predictive distribution", fontsize=14)


    plt.show()


train_and_test()






