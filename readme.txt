This file contains brief description of dashatest.py script.

General concept.

Script takes in a list of different embedding files and corresponding embedding sizes through argv, sequentially trains several BiDirectional LSTMs (each takes certain embedding type as an input), makes predictions on test.csv and then saves mean prediction of all LSTMs to specified location.

Details.

'train_path' -- Path to training data. Training data should be a csv file containing 8 columns, separated with commas. Columns include:
                'id', 'comment_text' and names of six classes.
                
'test_path' --Path to test data. Test data should be a csv file containin 2 columns, separated with commas. Columns include:
                'id', 'comment_text'.
                
'save_pred_path' -- Where to save final predictions.

'--embeddings' -- List of paths to embedding files. A BiDirectional LSTM will be trained for each embedding file in the list. Embedding files 
                  must be structured as: <word>, <embedding>.
                  
'--embeddings_sizes' -- List of ints corresponding to embeddings sizes. Note that length of this list must be the same as the length
                        of embeddings list.
'--max_embeddings' -- Ammount of words to use embeddings for. Some embedding files may contain up to 2 million embeddings. By specifying
                      this argument one can control the number of embeddings used.
                      
'--lemmatize' -- Wether to lemmatize comments.

'--batch_size' -- Size of batch used for training and prediction. Generally -- the bigger the batch size , the more stable the training
                  process is.

'--num_classes' -- Number of classes for classification. In this particular task num_classes=6.

'--num_epochs' -- Number of epochs for which every LSTM is trained.

'--verbosity' -- Wether to increase the verbosity.