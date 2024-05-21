from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, ReLU, Flatten, MultiHeadAttention, Reshape, BatchNormalization
from numpy import concatenate
from xgboost import XGBRegressor
import pickle
import pandas as pd
import forecast_evaluation  # This is a custom module for forecast evaluation

# Define a function for building MHA-MLP Autoencoder model
def build_mha_mlp_ae_model(n_inputs):
    # Number of attention heads
    num_heads = 8
    
    # Define encoder layers
    visible = Input(shape=(n_inputs,))
    e = Dense(128, activation='relu', kernel_initializer='he_uniform')(visible)
    e = Reshape((128, 1), input_shape=(128,))(e)
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=128, kernel_initializer='he_uniform')(e, e)
    e = Flatten()(attention)
    e = Dense(64, activation='relu', kernel_initializer='he_uniform')(e)
    e = BatchNormalization()(e)
    e = ReLU()(e)
	
    # Define bottleneck layer
    n_bottleneck = 32
    bottleneck = Dense(n_bottleneck, activation='relu', kernel_initializer='he_uniform')(e)
	
    # Define decoder layers
    d = Dense(64, activation='relu', kernel_initializer='he_uniform')(bottleneck)
    d = BatchNormalization()(d)
    d = ReLU()(d)
    d = Reshape((64, 1), input_shape=(64,))(d)
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=128, kernel_initializer='he_uniform')(d, d)
    d = Flatten()(attention)
    d = Dense(128, activation='relu', kernel_initializer='he_uniform')(d)
	
    # Output layer
    output = Dense(n_inputs, activation='linear', kernel_initializer='he_uniform')(d)
	
    # Define autoencoder model
    mha_mlp_ae_model = Model(inputs=visible, outputs=output)
	
    # Compile autoencoder model
    mha_mlp_ae_model.compile(optimizer='adam', loss='mse')
	
    return mha_mlp_ae_model

# Define a function for training and testing the models
def mha_mlp_ae_egb_model(X, y, epochs=1, file_name='model_prediction.csv'):
    # Get the dimensions of input data
    batches = X.shape[0]
    timesteps = X.shape[1]
    features = X.shape[2]

    # Reshape the input data if needed
    X = X.reshape(batches, timesteps * features)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)
    
    # Get number of input columns
    n_inputs = x_train.shape[1]
    
    # Build MHA-MLP Autoencoder model
    mha_mlp_ae_model = build_mha_mlp_ae_model(n_inputs)
   
    # Fit MHA-MLP Autoencoder model on the training data
    print('[INFO]---|| *** Training MHA-MLP Autoencoder Model...\n')
    mha_mlp_ae_model.fit(x_train, x_train, epochs=epochs, batch_size=64, validation_data=(x_test, y_test), verbose=2)
    print('[INFO]---|| *** MHA-MLP Autoencoder Model Trained!\n')
    
    # Define MHA-MLP encoder model without the decoder
    mha_mlp_e_model = Model(inputs=mha_mlp_ae_model.inputs, outputs=mha_mlp_ae_model.layers[8].output)
    
    # Save the MHA-MLP encoder
    print('[INFO]---|| *** Saving the MHA-MLP Encoder Model...\n')
    mha_mlp_e_model.save('Models/mha_mlp_e_model.h5')
    print('[INFO]---|| *** MHA-MLP Encoder Model Saved!\n')
    
    # Extract features using the MHA-MLP encoder for training
    mha_mlp_e_model_features = mha_mlp_e_model.predict(x_train, verbose=0)
	
    # Define XGBoost model
    egb = XGBRegressor(n_estimators=128, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
   
    # Fit XGBoost model on the training dataset
    print('[INFO]---|| *** Training the XGBoost Model...\n')
    egb.fit(mha_mlp_e_model_features, y_train)
    print('[INFO]---|| *** XGBoost Model Trained!\n')
    
    # Save the model to a file
    print('[INFO]---|| *** Saving the XGBoost Model...\n')
    with open('Models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(egb, f)
    print('[INFO]---|| *** XGBoost Model Saved!\n')

    # Extract features using the MHA-MLP encoder for testing
    mha_mlp_e_model_features = mha_mlp_e_model.predict(x_test, verbose=0)

    print('[INFO]---|| *** Testing the XGBoost Model...\n')    
    yhat = egb.predict(mha_mlp_e_model_features)
    print('[INFO]---|| *** XGBoost Model Testing Completed!\n')

    # Saving predictions to a CSV file
    df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': yhat.flatten()})
    df.to_csv(file_name, index=False)
    print("CSV file '{}' created successfully.".format(file_name))

    # Evaluating model predictions
    forecast_evaluation.evaluate_forecasts(y_test, yhat)
