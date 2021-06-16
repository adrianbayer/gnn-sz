#-----------
# See for explanation of different field in Illustris files:
# https://www.tng-project.org/data/docs/specifications/#sec2a
# Example use h5py: https://github.com/franciscovillaescusa/Pylians3/blob/master/documentation/miscellaneous.md#h5py_P
#-----------

# change namemodel()

import time, datetime, psutil
from Source.networks import *
from Source.training import *
from Source.plotting import *
from Source.load_data import *


# Main routine to train the neural net
def main(params, verbose = True):

    use_model, learning_rate, weight_decay, n_layers, k_nn, n_epochs, training, simtype, simset, n_sims = params

    # Load data and create dataset
    dataset, node_features = create_dataset(simtype, simset, n_sims)

    # Split dataset among training, validation and testing datasets
    train_loader, valid_loader, test_loader = split_datasets(dataset)

    # Initialize model
    model = ModelGNN(use_model, node_features, n_layers, k_nn)
    model.to(device)
    if verbose: print("Model: " + namemodel(params)+"\n")

    # Print the memory (in GB) being used now:
    process = psutil.Process()
    print("Memory being used (GB):",process.memory_info().rss/1.e9)

    # Train the net
    if training:
        if verbose: print("Training!\n")
        train_losses, valid_losses = training_routine(model, train_loader, valid_loader, params, verbose)

    # Test the net
    if verbose: print("\nTesting!\n")
    if training==False: params[7]="IllustrisTNG"   # change for loading the model
    state_dict = torch.load("Models/"+namemodel(params), map_location=device)
    if training==False: params[7]="SIMBA"   # change after loading the model
    model.load_state_dict(state_dict)
    test_loss, rel_err = test(test_loader, model, torch.nn.MSELoss(), params, message_reg=sym_reg)
    if verbose: print("Test Loss: {:.2e}, Relative error: {:.2e}".format(test_loss, rel_err))

    # Plot loss trends
    if training:
        plot_losses(train_losses, valid_losses, test_loss, rel_err, params)

    # Plot true vs predicted halo masses
    plot_out_true_scatter(params)

    #if training:
    #    return np.amin(valid_losses)
    return test_loss


#--- MAIN ---#

if __name__ == "__main__":

    time_ini = time.time()

    for path in ["Plots", "Models", "Outputs"]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Load default parameters
    from params import params

    main(params)

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
