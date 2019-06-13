```
FinalProject_Group8.zip
|
|--- CS_247_proposal.pdf
|--- data 
|     |--- yelp
|     |     |--- bow
|     |     |     |--- yelp_bow_reviews.p
|     |     |     |--- yelp_bow_labels.p
|     |     |
|     |     |--- doc2vec
|     |     |     |--- yelp_doc2vec_reviews.p
|     |     |     |--- yelp_doc2vec_labels.p
|     |     |
|     |     |--- reviews_labels.txt
|     |
|     |
|     |--- restaurant
|     |     |--- bow
|     |     |     |--- X_train_bow_rest (Restaurant train data bag of words representation)
|     |     |     |--- X_test_bow_rest (Restaurant test data bag of words representation)
|     |     |     |--- y_train_bow_rest (Restaurant train label bag of words representation)
|     |     |     |--- y_test_bow_rest (Restaurant test label bag of words representation)
|     |     |
|     |     |--- doc2vec
|     |     |     |--- X_train_doc2vec_rest (Restaurant train data doc2vec embedding)
|     |     |     |--- X_test_doc2vec_rest (Restaurant test data doc2vec embedding)
|     |     |     |--- y_train_doc2vec_rest (Restaurant train label doc2vec embedding)
|     |     |     |--- y_test_doc2vec_rest (Restaurant test label doc2vec embedding)
|     |     |
|     |     |--- train.xml
|     |     |--- test.xml
|     |     |--- train.json
|     |     |--- test.json
|     |     |--- train_data.txt
|     |     |--- test_data.txt
|     |
|     |--- Readme.txt 
|
|--- model
|     |---aspect_extraction
|     |     |--- feature_regression_mlp_bow_0.p
|     |     |--- feature_regression_mlp_bow_1.p
|     |     |--- feature_regression_mlp_bow_2.p
|     |     |--- feature_regression_mlp_bow_3.p
|     |     |--- feature_regression_mlp_bow_4.p
|     |     |--- feature_regression_mlp_0.p
|     |     |--- feature_regression_mlp_1.p
|     |     |--- feature_regression_mlp_2.p
|     |     |--- feature_regression_mlp_3.p
|     |     |--- feature_regression_mlp_4.p
|     |
|     |--- enwiki_dbow
|     |     |--- doc2vec.bin
|     |     |--- doc2vec.bin.syn0.npy
|     |     |--- doc2vec.bin.syn1neg.npy
|     |
|     |--- vectorizer.p
|
|
|--- aspect_extraction_bow.py 
|--- aspect_extraction_doc2vec.py
|--- baseline.py
|--- model1_bow_logistic.py
|--- model1_bow_mlp.py
|--- model1_doc2vec_logistic.py
|--- model1_doc2vec_mlp.py
|--- model2_bow.py
|--- model2_doc2vec.py
|--- models2_utils.py
|--- rest_bow.py
|--- rest_doc2vec.py
|--- rest_preprocess.py (preprocess Restaurant data)
|--- yelp_bow.py (preprocess Yelp data)
|--- yelp_doc2vec.py (preprocess Yelp data)
|--- yelp_preprocess.py
```

The google drive link of our data: https://drive.google.com/drive/folders/1Rzp888_zlH8Bh__Dzerf4wJr95Zc6qSl 
Before run, download and drag all dataset and model file into the 'data' and 'model' folder following the above hierarchy


The basic dataset you need is 'train.xml' and 'test.xml' for restaurant dataset, and 'reviews_label.txt' for yelp dataset.

Our workflow is
```
      Data preprocessing                         (rest_preprocess.py, yelp_preprocess.py) 
  --> Embedding                                  (yelp_bow.py, yelp_preprocess.py, 
                                                  rest_bow.py, rest_doc2vec.py) 
  --> Aspect polarity extraction module training (aspect_extraction_bow.py, aspect_extraction_doc2vec.py)
  --> Sentiment classification                   (model1_bow_logistic.py, model1_bow_mlp.py, 
                                                  model1_doc2vec_logistic.py, model1_doc2vec_mlp.py, 
                                                  model2_bow.py, model2_doc2vec.py)
```

If all dataset are loaded under the data folder, you can run each file shown above separately,
otherwise, load the basic datasets, and run code following the above workflow

To run any file above type the command ```'python filename.py'```, without single qoute.

run ```'python baseline.py'``` to show all base line evaluation results.




