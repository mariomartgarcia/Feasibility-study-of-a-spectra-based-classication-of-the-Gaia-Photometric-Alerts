# Feasibility study of a spectra-based classication of the Gaia Photometric Alerts

### Abstract

The photometric alerts obtained by the Gaia satellite are collected when a change in magnitude between two observations is detected. This alert is recorded to be later studied and classified, in other words, to know what has caused it (variable star, microlensing effects, transits...). The project focuses on the alerts that have been published and classified in order to study the feasibility of automating the process of classifying these alerts.

After selecting the alerts with the greatest representation, a web scraping process is carried out where the photometric spectra of each of the alerts participating in the study are obtained. Once we obtain a dataset formed by the photometric spectra and the classification of the alert, various supervised machine learning techniques are implemented. Given the large volume of data we work with, a balanced random subset of 4000 elements is selected to obtain the best hyperparameters and evaluate the performance of the following classifiers: Decision Trees, Support Vector Machines, Random Forests and Gradient Boosting Classifier. This process is repeated on the complete dataset using the hyperparameters obtained in the subset. Finally, the performance of different models for an Artificial Neural Network is constructed and evaluated.

The best model obtained is the Gradient Boosting Classifier which, with a maximum depth of 7 nodes, 200 estimators and a learning rate of 0.1, obtains an accuracy of 66.8\%. Although the results are not excessively good, we can affirm that the classification of Gaia photometric alerts according to the spectra is feasible.


- Keywords: "Gaia", "Photometry", "Classification", "Machine Learning", "Web Scraping".
