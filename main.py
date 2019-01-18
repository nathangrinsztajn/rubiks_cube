from cube import *
from network import *
import random
from keras.callbacks import ModelCheckpoint
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--modelpath", type=str, default = model_path,
                help="path to output model")
ap.add_argument("-sname", "--modelsavename", type=str, default = "",
                help="model name")
ap.add_argument("-lname", "--modelloadname", type=str, default = "",
                help="model name")
ap.add_argument("-e", "--epoch", type=int, default=2,
                help="number of epochs")
ap.add_argument("-ns", "--nshuffle", type=int, default=3,
                help="number of shuffling")
ap.add_argument("-nb", "--nbatch", type=int, default=1,
                help="number of batch per epoch")
ap.add_argument("-nit", "--niter", type=int, default=500,
                help="number of iteration")
args = vars(ap.parse_args())

n_iter = args["niter"]
n_shuffle = args["nshuffle"]
n_cube = args["nbatch"]*int(64/n_shuffle)
model_path = args["modelpath"]
model_save_name = model_path + "weights" + args["modelsavename"] + ".h5"
model_load_name = model_path + "weights" + args["modelloadname"] + ".h5"

model.load_weights(model_load_name)

model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer)

def generate_dataset(_model):
    ni = []
    X = np.zeros((n_shuffle*n_cube, 20*24))
    Y_policy = np.zeros((n_shuffle*n_cube, 12))
    Y_value = np.zeros((n_shuffle*n_cube, 1))
    for nc in range(n_cube):
        _cube = Cube()
        for i in range(n_shuffle):
            _move = moves[random.randint(0, 11)]
            _cube, r = _cube.move(_move)
            children_states = np.zeros((12, 20*24))
            children_rewards = []
            for m in range(12):
                child, r_child = _cube.move(moves[m])
                children_states[m, :] = child.reducted_state()
                children_rewards.append(r_child)
            children_rewards = np.array(children_rewards)
            children_states = np.array(children_states)
            children_values = _model.predict(children_states)[1].reshape(12)
            children_values = np.array(children_values)*((children_rewards-1)/(-2)) + np.array(children_rewards)
            max_pol = np.argmax(children_values)
            max_val = children_values[max_pol]
            max_pol_array = np.zeros(12)
            max_pol_array[max_pol]=1
            ni.append(i+1)
            X[nc*n_shuffle + i, :] = _cube.reducted_state()
            Y_policy[nc*n_shuffle + i, :] = max_pol_array
            Y_value[nc*n_shuffle + i, :] = max_val
    return X, Y_policy, Y_value, ni

def generate_dataset_weighted(_model):
    ni = []
    X = []
    Y_policy = []
    Y_value = []
    probas_weight = 1 / ((np.arange(n_shuffle) + 1) * (np.arange(n_shuffle) + 2))
    probas_weight[n_shuffle-1] = 1/n_shuffle
    for nc in range(n_cube):
        _cube = Cube()
        n_shuffle_local = np.random.choice(np.arange(n_shuffle), p=probas_weight)+8 #+1
        for i in range(n_shuffle_local):
            _move = moves[random.randint(0, 11)]
            _cube, r = _cube.move(_move)
            children_states = np.zeros((12, 20*24))
            children_rewards = []
            for m in range(12):
                child, r_child = _cube.move(moves[m])
                children_states[m, :] = child.reducted_state()
                children_rewards.append(r_child)
            children_rewards = np.array(children_rewards)
            children_states = np.array(children_states)
            children_values = _model.predict(children_states)[1].reshape(12)
            children_values = np.array(children_values)*((children_rewards-1)/(-2)) + np.array(children_rewards)
            max_pol = np.argmax(children_values)
            max_val = children_values[max_pol]
            max_pol_array = np.zeros(12)
            max_pol_array[max_pol]=1
            ni.append(i+1)
            X.append(_cube.reducted_state())
            Y_policy.append(max_pol_array)
            Y_value.append(max_val)
    X = np.array(X)
    Y_policy = np.array(Y_policy)
    Y_value = np.array(Y_value)
    return X, Y_policy, Y_value, ni

# checkpointer = ModelCheckpoint(filepath="/Users/anneabeille/Documents/MVA/RL/rubiks_cube/models/" + 'model',
#                                verbose=1)

for epoch in range(n_iter):
    print(epoch)
    X, Y_policy, Y_value, ni = generate_dataset_weighted(model)
    model.fit(X, {"output_policy": Y_policy, "output_value": Y_value}, epochs=args["epoch"], batch_size=64)#, callbacks=[checkpointer])
    model.save_weights(model_save_name)
