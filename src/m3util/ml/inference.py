import torch
import time
import numpy as np


def computeTime(
    model, train_dataloader, batch_size, device="cuda", write_to_file=False
):
    """
    Compute the execution time of a model on a given dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        train_dataloader (torch.utils.data.DataLoader): The dataloader containing the input data.
        batch_size (int): The batch size used during inference.
        device (str, optional): The device to use for inference. Defaults to 'cuda'.
        write_to_file (bool, optional): Whether to write the execution time to a file. Defaults to False.

    Returns:
        str: The average execution time in milliseconds, if write_to_file is True.

    """

    model.to(device)  # Ensure the model is on the correct device
    model.eval()

    time_spent = []
    batch_count = 0  # Initialize batch counter
    for i, data in enumerate(train_dataloader, 1):
        batch_count += 1  # Increment batch counter
        start_time = time.time()
        with torch.no_grad():
            # Check if data is a tuple/list and extract inputs
            if isinstance(data, (list, tuple)):
                inputs = data[0].to(device)
            else:
                inputs = data.to(device)
            _ = model(inputs)

        if device == "cuda":
            torch.cuda.synchronize()  # Wait for CUDA to finish (CUDA is asynchronous!)

        if i != 0:
            time_spent.append(time.time() - start_time)

    # Handle case where no batches were processed
    if batch_count == 0:
        print(f"No batches were processed. Dataloader might be empty.")
        return None

    print(
        f"Mean execution time computed for {batch_count} batches of size {batch_size}"
    )

    averaged_time = np.mean(time_spent) * 1000  # Convert to milliseconds
    std_time = np.std(time_spent) * 1000
    print(
        f"Average execution time per batch (ms): {averaged_time:.6f} ± {std_time:.6f}"
    )
    print(
        f"Average execution time per iteration (ms): {averaged_time / batch_size:.6f} ± {std_time / batch_size:.6f}"
    )

    print(f"Total execution time (s): {np.sum(time_spent):.2f} ")

    if write_to_file:
        return f"Avg execution time (ms): {averaged_time:.6f}"
