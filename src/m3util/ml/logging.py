from m3util.util.IO import append_to_csv

def write_csv(write_CSV,
              path,
              model_name,
              optimizer_name,
              i,
              noise,
              epochs,
              total_time,
              train_loss,
              batch_size,
              loss_func,
              seed,
              stoppage_early,
              model_updates,
              filename):
    """
    Writes the training details to a CSV file if the 'write_CSV' path is provided.

    Args:
        write_CSV (str): The filename or path to the CSV file. If None, the function does nothing.
        path (str): The directory path where the model is saved.
        model_name (str): The name of the model.
        optimizer_name (str): The name of the optimizer used during training.
        i (int): Training instance number or index.
        noise (float): The noise level in the dataset.
        epochs (int): The number of epochs completed during training.
        total_time (float): The total time taken for training in seconds.
        train_loss (float): The final training loss after all epochs.
        batch_size (int): The batch size used during training.
        loss_func (callable): The loss function used during training.
        seed (int): The random seed used for reproducibility.
        stoppage_early (bool): Whether early stopping was triggered during training.
        model_updates (int): The number of model updates (training steps) performed.

    Returns:
        None: Writes the data to the specified CSV file if 'write_CSV' is not None.
    """

    # Check if a CSV file path is provided
    if write_CSV is not None:
        # Define the CSV headers
        headers = ["Model Name",
                   "Training Number",
                   "Noise",
                   "Optimizer",
                   "Epochs",
                   "Training_Time",
                   "Train Loss",
                   "Batch Size",
                   "Loss Function",
                   "Seed",
                   "Filename",
                   "Early Stoppage",
                   "Model Updates",
                   "Filename"]

        # Compile the data to be written to the CSV
        data = [model_name,
                i,
                noise,
                optimizer_name,
                epochs,
                total_time,
                train_loss,
                batch_size,
                loss_func,
                seed,
                f"{path}/{model_name}_model_epoch_{epochs}_train_loss_{train_loss}.pth",
                f"{stoppage_early}",
                f"{model_updates}",
                filename]

        # Append the data to the CSV file at the specified path
        append_to_csv(f"{path}/{write_CSV}", data, headers)


def save_list_to_txt(lst, filename):
    """
    Saves a list of items to a text file.

    Args:
        lst (list): The list of items to save.
        filename (str): The name of the file to save to.
    """
    with open(filename, "w") as file:
        for item in lst:
            file.write(str(item) + "\n")
