from keras.models import Model, load_model
from keras import backend as K
from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Concatenate, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, SGD


reduced_state = Input(shape=(20*24,))
layer1 = Dense(4096, activation="relu")(reduced_state)
layer2 = Dense(2048, activation="relu")(layer1)
layer3_policy = Dense(512, activation="relu")(layer2)
layer3_value = Dense(512, activation="relu")(layer2)
output_value = Dense(1, name="output_value")(layer3_value)
output_policy = Dense(12, name="output_policy", activation="softmax")(layer3_policy)

model = Model(
    inputs=reduced_state,
    outputs=[output_policy, output_value])

losses = {
	"output_policy": "categorical_crossentropy",
	"output_value": "mse"
}
lossWeights = {"output_policy": 1.0, "output_value": 4.0}

optimizer = RMSprop(lr=0.0004, rho=0.9, epsilon=None, decay=0.0)
# optimizer = SGD(lr=0.001, momentum=0.2)
