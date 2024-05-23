import optuna
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from LSTM import LSTM1
import math
from batches import batch
import copy


def LSTM_hyperparametersearch(X_train_tensor, y_train_tensor, n_notes_train, X_test_tensor, y_test_tensor, n_notes_test, resampled = False):
    
    def objective(trial):
        #Hyperparameterspaces to explore
        #num_epochs = trial.suggest_int("num_epochs", 2, 10, step = 1)
        learning_rate = trial.suggest_categorical("learning_rate", [0.0001, 0.001, 0.01, 0.1])
        batch_size = trial.suggest_int("batch_size", 16, 64, step = 16)
        hidden_size = trial.suggest_int("hidden_size", 32, 128, step = 16)

 #number of features in hidden state


        input_size = X_train_tensor.shape[2] #number of features
        #hidden_size = 60 #number of features in hidden state
        num_layers = 1 #number of stacked lstm layers
        num_epochs = 10

        num_classes = 2 #number of output classes 
        seq_length = X_train_tensor.shape[1] #maximum amount of notes

        
        ######## TRAINING LSTM

        #instantiating LSTM1 object
        my_LSTM = LSTM1(num_classes, input_size, hidden_size, num_layers, seq_length).to("cuda")

        for param in my_LSTM.parameters():
            print (param.device)

        if resampled == False:
            weights = [0.02, 0.98]
        else:
            weights = [0.2, 0.8]

        class_weights = torch.FloatTensor(weights).cuda() 

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights) 
        optimizer = torch.optim.Adam(my_LSTM.parameters(), lr=learning_rate) 


        #Initialize Variables for EarlyStopping
        best_loss = float('inf')
        best_model_weights = None
        patience = 3

        #training loop
        for epoch in range(num_epochs):
            
            #batch_size = 32
            batches_x = batch(X_train_tensor, batch_size) 
            batches_y = batch(y_train_tensor, batch_size)
            batches_n_notes = batch(n_notes_train, batch_size)

            n_batches = math.ceil(len(X_train_tensor)/batch_size)
            
            for i in range(n_batches):
                
                batch_x = next(batches_x)
                for j in batch_x:
                    batch_x_new = torch.stack(batch_x)
                
                batch_y = next(batches_y)
                for k in batch_y:
                    batch_y_new = torch.stack(batch_y)

                batch_n_notes = next(batches_n_notes)
                for l in batch_n_notes:
                    batch_n_notes_new = torch.stack(batch_n_notes)

                #Converting batch_y_new to 2 dim torch
                batch_y_array = []
                for i in batch_y_new:
                    if i == 0. or i == 0:
                        y_value = [1, 0]
                    elif i == 1. or i == 1:
                        y_value = [0, 1]
                    batch_y_array.append(y_value)

                batch_y_torch = torch.Tensor(batch_y_array).cuda()

                outputs = my_LSTM.forward(batch_x_new, batch_n_notes)# (X_train_tensor) #forward pass
                optimizer.zero_grad() #calculate the gradient, manually setting to 0

                #obtaining loss function
                loss = criterion(outputs, batch_y_torch) 

                loss.backward() #calculates the loss of the loss function

                optimizer.step() #improve from loss, i.e backprop
            if epoch % 2 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
                    # Early stopping
            
            if loss < best_loss:
                best_loss = loss
                best_model_weights = copy.deepcopy(my_LSTM.state_dict())  # Deep copy here      
                patience = 3  # Reset patience counter
            else:
                patience -= 1
                if patience == 0:
                    break 
        
            # Load the best model weights
            my_LSTM.load_state_dict(best_model_weights)


        
        ######## TESTING LSTM       
        batches_x = batch(X_test_tensor, batch_size)
        batches_y = batch(y_test_tensor, batch_size)
        batches_n_notes = batch(n_notes_test, batch_size)

        n_batches = math.ceil(len(X_test_tensor)/batch_size)
            
        test_loss = 0.0
        # Generate predictions for the test dataset
        predictions_temp = []

        with torch.no_grad():
            for i in range(n_batches):
                
                batch_x = next(batches_x)
                for j in batch_x:
                    batch_x_new = torch.stack(batch_x)
                
                batch_y = next(batches_y)
                for k in batch_y:
                    batch_y_new = torch.stack(batch_y)
                
                batch_n_notes = next(batches_n_notes)
                for l in batch_n_notes:
                    batch_n_notes_new = torch.stack(batch_n_notes)

                #Converting batch_y_new to 2 dim torch
                batch_y_array = []
                for i in batch_y_new:
                    if i == 0. or i == 0:
                        y_value = [1, 0]
                    elif i == 1. or i == 1:
                        y_value = [0, 1]
                    batch_y_array.append(y_value)

                batch_y_torch = torch.Tensor(batch_y_array).cuda()


                # Forward pass
                outputs = my_LSTM(batch_x_new, batch_n_notes_new)

                loss = criterion(outputs, batch_y_torch)

                test_loss += loss.item()

                # Save the predictions
                predictions_temp += outputs.squeeze().tolist()
                

            # Compute the evaluation metrics
            avg_test_loss = test_loss / len(X_test_tensor)
            print('Average Test Loss: {:.4f}'.format(avg_test_loss))


        # Convert the predictions and actual values to numpy arrays
        predictions_temp = np.array(predictions_temp)
        predictions = []
        for i in predictions_temp:
            pred = np.argmax(i)
            predictions.append(pred)
            

        actuals = y_test_tensor.cpu().numpy()


        #Evaluate based on ROC-AUC
        roc_auc = roc_auc_score(actuals, predictions)

        return roc_auc


    #Run study to find the hyperparameters that maximize the ROC-AUC score of the LSTM model
    study = optuna.create_study(direction = "maximize") 
    study.optimize(objective, n_trials = 50)  

    return study.best_params