import torch
import numpy as np
from LSTM import LSTM1
import math
from batches import batch
import copy



def LSTM_evaluation(X_train_tensor, y_train_tensor, n_notes_train, X_test_tensor, y_test_tensor, n_notes_test, learning_rate, batch_size, hidden_size, resampled = False):

    input_size = X_train_tensor.shape[2] #number of features
    seq_length = X_train_tensor.shape[1] #maximum amount of notes 
    num_layers = 1 #number of stacked lstm layers
    num_epochs = 100
    num_classes = 2 #number of output classes 
    

        
    ######## TRAINING LSTM

    #instantiating LSTM1 object
    my_LSTM = LSTM1(num_classes, input_size, hidden_size, num_layers, seq_length).to("cuda")

    for param in my_LSTM.parameters():
        print (param.device)

    if resampled == False:
        weights = [0.03, 0.97] #The positive percentage in the "a prediction every day"-data
    else:
        weights = [0.3, 0.7] #The positive percentage in the "one prediction per patient"-data

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
        
        if loss < best_loss:
            best_loss = loss
            best_model_weights = copy.deepcopy(my_LSTM.state_dict())  # Deep copy here      
            patience = 3  # Reset patience counter
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping")
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

    from sklearn.metrics import classification_report
    #[print(i) for i in predictions]

    print(max(np.unique(predictions)))

    # Applying transformation to get binary values predictions with 0.5 as thresold
    binary_predictions = list(map(lambda x: 0 if x < 0.5 else 1, predictions))

    ####################### MODEL EVALUATION #############################

    print(classification_report(y_test_tensor.cpu(), binary_predictions))

    def perf_measure(y_actual, y_hat):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                FP += 1
            if y_actual[i]==y_hat[i]==0:
                TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                FN += 1

        return(f"True Positives: {TP}. False Positives: {FP}. True negatives: {TN}. False negatives: {FN}.")

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score

    print(perf_measure(actuals, binary_predictions))

    accuracy = accuracy_score(actuals, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(f'Precision = {average_precision_score(actuals, predictions)}')
    print(f'Recall = {recall_score(actuals, predictions)}')
    print(f'F1 score = {f1_score(actuals, predictions)}')
    print(f'ROC-AUC score = {roc_auc_score(actuals, predictions)}') #0.524 that is predictive ability is not better than random guessing

    return predictions