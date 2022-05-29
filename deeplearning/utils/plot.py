import matplotlib.pyplot as plt

def plotTrainingHistory(train_history, val_history, metric):
    fig, (loss_ax, acc_ax) = plt.subplots(nrows=2)
    loss_ax.set_title("Cross Entropy Loss")
    loss_ax.plot(train_history["loss"], label="train")
    loss_ax.plot(val_history["loss"], label="val")
    loss_ax.legend(loc="best")

    acc_ax.set_title(f"Classification {metric}")
    acc_ax.plot(train_history["metric"], label="train")
    acc_ax.plot(val_history["metric"], label="val")

    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()