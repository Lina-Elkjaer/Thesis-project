from sklearn.utils import class_weight
import optuna
from tfidf import tf_idf
from create_tensors import max_notes, tensors
from create_tensors_resampled import max_notes_resampled, tensors_resampled
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from LSTM import LSTM1
import math
from batches import batch


def tfidf_hyperparametersearch(predictor_dict_train, predictor_dict_test, df_admissions_train, df_admissions_test, y_train_df, y_test_df, resampled = False):
    
    def objective(trial):
        #Hyperparameterspaces to explore
        min_df = trial.suggest_int("min_df_trial", 0, 100, step = 10) 
        max_df = trial.suggest_float("max_df_trial", 0.50, 1.0, step = 0.1)
        max_features = trial.suggest_int("max_features_trial", 200, 600, step = 100)
        ngram_range_upper = trial.suggest_int("ngram_range_upper_trial", 1, 2)

        #Initialising
        df_tfidfvect_train, df_tfidfvect_test = tf_idf(predictor_dict_train["df_SFI_text"], predictor_dict_test["df_SFI_text"], min_df = min_df, max_df = max_df, max_features=max_features, ngram_range = (1, ngram_range_upper))

        print(max_features)

        if resampled == False:
            #Creating tensors
            maximum_notes_train = max_notes(df_tfidfvect_train, df_admissions_train) #(df_tfidfvect_train_small, df_admissions_train_small)

            maximum_notes_test = max_notes(df_tfidfvect_test, df_admissions_test) #(df_tfidfvect_test_small, df_admissions_test_small)

            maximum_notes = max(maximum_notes_train, maximum_notes_test)

            #train
            X_tfidfvect_train_tensor, y_tfidfvect_train_tensor, X_structured_train_tensor_tfidf, n_notes_train = tensors(df_tfidfvect_train, y_train_df, df_admissions_train, maximum_notes, predictor_dict_train) #(df_tfidfvect_train_small, y_train_df_small, df_admissions_train_small, maximum_notes, predictor_dict_train)

            #test
            X_tfidfvect_test_tensor, y_tfidfvect_test_tensor, X_structured_test_tensor_tfidf, n_notes_test = tensors(df_tfidfvect_test, y_test_df, df_admissions_test, maximum_notes, predictor_dict_test) #(df_tfidfvect_train_small, y_train_df_small, df_admissions_train_small, maximum_notes, predictor_dict_train)

        
        elif resampled == True:
            #Creating tensors
            maximum_notes_train = max_notes_resampled(df_tfidfvect_train) #(df_tfidfvect_train_small, df_admissions_train_small)

            maximum_notes_test = max_notes_resampled(df_tfidfvect_test) #(df_tfidfvect_test_small, df_admissions_test_small)

            maximum_notes = max(maximum_notes_train, maximum_notes_test)
            
            #train
            X_tfidfvect_train_tensor, y_tfidfvect_train_tensor, X_structured_train_tensor_tfidf, n_notes_train = tensors_resampled(df_tfidfvect_train, y_train_df, df_admissions_train, maximum_notes, predictor_dict_train) #(df_tfidfvect_train_small, y_train_df_small, df_admissions_train_small, maximum_notes, predictor_dict_train)

            #test
            X_tfidfvect_test_tensor, y_tfidfvect_test_tensor, X_structured_test_tensor_tfidf, n_notes_test = tensors_resampled(df_tfidfvect_test, y_test_df, df_admissions_test, maximum_notes, predictor_dict_test) #(df_tfidfvect_train_small, y_train_df_small, df_admissions_train_small, maximum_notes, predictor_dict_train)


        print(X_tfidfvect_train_tensor.shape)
        print(X_tfidfvect_test_tensor.shape)


        y_tfidfvect_train_tensor = torch.unsqueeze(y_tfidfvect_train_tensor, 1).cuda()
        y_tfidfvect_test_tensor = torch.unsqueeze(y_tfidfvect_test_tensor, 1).cuda()

        X_tfidfvect_train_tensor = X_tfidfvect_train_tensor.cuda()
        X_tfidfvect_test_tensor = X_tfidfvect_test_tensor.cuda()
        n_notes_train.cuda()
        n_notes_test.cuda()

        #y_tfidfvect_test_tensor.size()      

        #Initiazing hyperparameters of LSTM
        num_epochs = 10 #10 epochs
        learning_rate = 0.001 #0.001 lr

        input_size = X_tfidfvect_train_tensor.shape[2] #number of features
        hidden_size = 60 #number of features in hidden state
        num_layers = 1 #number of stacked lstm layers

        num_classes = 2 #number of output classes 
        seq_length = X_tfidfvect_train_tensor.shape[1] #maximum amount of notes
        batch_size = 32 #NB must be the same as batch_size further down

        print(X_tfidfvect_train_tensor.shape)
        print(X_tfidfvect_test_tensor.shape)
        
        
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

        #training loop
        for epoch in range(num_epochs):
            
            batch_size = 32
            batches_x = batch(X_tfidfvect_train_tensor, batch_size)
            batches_y = batch(y_tfidfvect_train_tensor, batch_size)
            batches_n_notes = batch(n_notes_train, batch_size)

            n_batches = math.ceil(len(X_tfidfvect_train_tensor)/batch_size)
            
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

                #print(outputs.shape)
                #print(batch_y_torch.shape)

                #obtaining loss function
                loss = criterion(outputs, batch_y_torch) 

                loss.backward() #calculates the loss of the loss function

                optimizer.step() #improve from loss, i.e backprop
            if epoch % 2 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
        

        
        ######## TESTING LSTM       
        batches_x = batch(X_tfidfvect_test_tensor, batch_size)
        batches_y = batch(y_tfidfvect_test_tensor, batch_size)
        batches_n_notes = batch(n_notes_test, batch_size)

        n_batches = math.ceil(len(X_tfidfvect_test_tensor)/batch_size)
            
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
            avg_test_loss = test_loss / len(X_tfidfvect_test_tensor)
            print('Average Test Loss: {:.4f}'.format(avg_test_loss))


        # Convert the predictions and actual values to numpy arrays
        predictions_temp = np.array(predictions_temp)
        predictions = []
        for i in predictions_temp:
            pred = np.argmax(i)
            predictions.append(pred)
            

        actuals = y_tfidfvect_test_tensor.cpu().numpy()


        #Evaluate based on ROC-AUC
        roc_auc = roc_auc_score(actuals, predictions)

        return roc_auc


    #Run study to find the tf-idf hyperparameters that maximize the ROC-AUC score of the XGBoost model
    study = optuna.create_study(direction = "maximize") 
    study.optimize(objective, n_trials = 50)  

    return study.best_params