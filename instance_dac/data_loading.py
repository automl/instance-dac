from dacbench.plotting import plot_performance, plot_performance_per_instance
from dacbench.logger import Logger, log2dataframe, load_logs
    
import matplotlib.pyplot as plt


# Load performance of last seed into pandas DataFrame
    logs = load_logs(performance_logger.get_logfile())
    dataframe = log2dataframe(logs, wide=True)

    # Plot overall performance
    plot_performance(dataframe)
    plt.show()

    # Plot performance per instance
    plot_performance_per_instance(dataframe)
    plt.show()