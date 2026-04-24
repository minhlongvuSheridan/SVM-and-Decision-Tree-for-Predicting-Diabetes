***Performance Evaluation of Decision Tree and SVM with Bagging to Detect Diabetes***

Group 6

Minh Long Vu

Rohan Patel

Prof. El Sayed Mahmoud

April 14, 2026


# Introduction

Diabetes has been a deadly disease. \[1\] shows that approximately 11% of global deaths in 2019 were attributed to diabetes. \[2\] shows that diabetes itself could lead to the development of other chronic diseases, such as Chronic Kidney Disease. Thus, there is a need for detecting diabetes as soon as possible to prevent it from developing worse. In the literature, many machine learning algorithms have been utilized to detect the diabetes. Among them, Support Vector Machine and Decision Tree are commonly used. However, traditional machine learning algorithms alone are not promising since their performance of them is poor. Some papers improve the traditional SVM by using K-means clustering \[3\] or CHAID decision tree \[4\]. Thus, this paper will attempt to improve the SVM and Decision Tree by using the bagging method. It asks a key research question “How effective is the Decision Tree and Support Vector Machine with Bagging method in detecting diabetes?”. From the key research question, there are two objectives that the paper tries to achieve:

1\. Evaluate each model independently

2\. Rank the algorithm and determine the better one for detecting diabetes

By conducting this research, the study could provide three things to the machine learning literature and the medical field. Firstly, the study extends the knowledge of the literature of how bagging method could improve the accuracy of machine learning in a particular field such as detecting diabetes. This is helpful for future study because the current literature doesn’t work much on the bagging method. Secondly, it provides a quantitative comparison between Support Vector Machine and Decision Tree after being improved by bagging method. Lastly, by determining which one performs better, the study could confidently suggest a strong and robust machine learning algorithm for detecting diabetes. The better the detection is, the more people will be saved in the future.

# Literature Review

### Support Vector Machine

\[5\] show that even though they have similar performance in detecting diabetes, the SVM is more efficient than traditional logistic regression because it requires no prior knowledge of data, such as distribution or interdependency. However, \[5\] also shows that depending on the Schemes of target variables, the AUC of both models could dramatically drop from around 83.5% to 73.2% regardless of Linear or RBF. Another study \[6\] argues that with a more feature-rich dataset, traditional SVM could significantly improve its accuracy, specifically 93.26%. \[6\] got this high accuracy 93% by using 16 variables which is double of the study \[3\] that just got 83.5%. However,  \[6\] doesn’t show how much the conventional SVM accuracy could achieve. Thus, the result might not be convincing since they use different datasets but it still indicates that accuracy may be increased by adding more features.

Taking another approach rather than using conventional SVM like \[5\] or \[6\], \[3\] modifies SVM by using K-means clustering for feature extraction and SVM for classifier. Overall,  SVM in \[3\] achieves an accuracy of 98.7% for the PIMA dataset. \[3\] also validate the result of \[6\] by showing the accuracy of conventional SVM in different datasets is consistently around 83% for 8 features. Another attempt that works on the same PIMA dataset is \[7\]. \[7\] show that the traditional SVM accuracy could be improved by a maximum 2% regardless of the percentage of data training if backward elimination is used as feature selection. However, this method of backward elimination could be really slow during the test phase. This might suggest that it is not worth modifying.

Instead of modifying the traditional SVM, \[8\] show that when SVM is used as a meta-learner for the stacking method, it achieves a higher accuracy of high accuracy which is 93.75% in the same PIMA dataset. \[8\] achieve this accuracy by only combining 4 different weak models that all have accuracies below 78%. One problem of \[8\] might be that it involves setting hyperparameters for different models which is a lot of work and hard to find the optimal

In summary, \[5\] and \[6\] show that traditional SVM could be a good alternative to statistical models for detecting diabetes. Some modifications have been made in attempting to improve the traditional SVM\[3\]\[7\]. The study \[8\] used an ensemble method similar to our study where we combine multiple weak models to create a strong and robust one. However, \[8\] uses a stacking technique that combines many different types of models, which could result in the overhead of setting hyperparameters. Furthermore, the SVM as a meta-learner might not reflect the actual capability of SVM as it can just learn from the other models' predictions instead of raw data.

### Decision Tree

As per the literature, Decision Trees (DT) can be well used to classify Pima Indian Diabetes because they are hierarchical and have natural interpretability. As demonstrated in \[8\], DT algorithms, including J48 give explicit diagnostic rules, which can be more easily understand by medical professionals compared to the other classification models. This can help the extraction of if-then rules that can then be easily tested against clinical knowledge. But \[8\] also points out that the effectiveness of a single DT can be sensitive to the particular breakdown of the training data, and that accuracy can be in the range of 75-78% based on the depth of the tree.

The other study \[9\] shows that the appropriateness of DT is improved by the capability to deal with the unusual feature of the Pima dataset, e.g., the combination of categorical-quality counts (e.g., number of pregnancies) and continuous values (e.g., BMI). In comparison to distance-based models, DTs do not involve the use of large amounts of data normalization or scaling and, therefore, are more effective in preliminary stages of data mining. There is also a further indication that the recursive partitioning quality of the algorithm is well successful in the capture of the non-linear relationships between physiological indicators and onset of diabetes.

In a more technical sense, \[9\] determines that the main issue of using a conventional DT on the Pima data is the overfitting that can lead to a large decrease in prediction accuracy when adopted to unknown test data. In response to this, \[9\] recommends pruning approaches and a minimum number of samples per leaf, to stabilize the model. The significance of feature engineering is supported in \[4\], indicating that a pruned DT can reach a good accuracy of about 82.4 with careful selection of significant and valuable features such as Glucose and BMI, compared to the average accuracy of around 41.7% of the base models that blindly use all feasible features.

Compared to the fact that the Pima data have been used so far with the help of the help of one tree, \[10\] proves that the ensemble methods based on the Decision tree are the most robust ones. This is demonstrated in \[10\] with figures showing that a single tree shows high variance though with ensemble methods such as the random forest or the gradient boosting which combines the decision of many trees, accuracy would rise to over 90 percent. Among the downsides mentioned in \[10\] is that when the model is one tree it is initially very interpretable, and because the model is now an ensemble it obtains a bit of predictive power but at the cost of some interpretability.

Overall, \[8\] and \[11\] define the Decision Tree as a more realistic and interpretable reference point in classifying diabetes, especially as a clinical reference with its transparency. Adjustments to pruning and feature choice \[9\]\[4\] are necessary to deal with the threats of overfitting that are present in the Pima dataset. Lastly, \[10\] demonstrates that ensemble techniques are by far the strongest application of DT logic, as it provides a better trade off between accuracy and reliability to diagnostic uses.

# Explain the Implementation

### Support Vector Machine

The general idea of the Support Vector Machine is to find the hyperplane that separates different classes in the feature space. A good SVM is one that can determine optimal support vectors to maximize the margin of the hyperplane. Maximizing the margin allows for improving the model’s generalization ability


***Performance Evaluation of Decision Tree and SVM with Bagging to Detect Diabetes***

Group 6

Minh Long Vu

Rohan Patel

Prof. El Sayed Mahmoud

April 14, 2026


# Introduction

Diabetes has been a deadly disease. \[1\] shows that approximately 11% of global deaths in 2019 were attributed to diabetes. \[2\] shows that diabetes itself could lead to the development of other chronic diseases, such as Chronic Kidney Disease. Thus, there is a need for detecting diabetes as soon as possible to prevent it from developing worse. In the literature, many machine learning algorithms have been utilized to detect the diabetes. Among them, Support Vector Machine and Decision Tree are commonly used. However, traditional machine learning algorithms alone are not promising since their performance of them is poor. Some papers improve the traditional SVM by using K-means clustering \[3\] or CHAID decision tree \[4\]. Thus, this paper will attempt to improve the SVM and Decision Tree by using the bagging method. It asks a key research question “How effective is the Decision Tree and Support Vector Machine with Bagging method in detecting diabetes?”. From the key research question, there are two objectives that the paper tries to achieve:

1\. Evaluate each model independently

2\. Rank the algorithm and determine the better one for detecting diabetes

By conducting this research, the study could provide three things to the machine learning literature and the medical field. Firstly, the study extends the knowledge of the literature of how bagging method could improve the accuracy of machine learning in a particular field such as detecting diabetes. This is helpful for future study because the current literature doesn’t work much on the bagging method. Secondly, it provides a quantitative comparison between Support Vector Machine and Decision Tree after being improved by bagging method. Lastly, by determining which one performs better, the study could confidently suggest a strong and robust machine learning algorithm for detecting diabetes. The better the detection is, the more people will be saved in the future.

# Literature Review

### Support Vector Machine

\[5\] show that even though they have similar performance in detecting diabetes, the SVM is more efficient than traditional logistic regression because it requires no prior knowledge of data, such as distribution or interdependency. However, \[5\] also shows that depending on the Schemes of target variables, the AUC of both models could dramatically drop from around 83.5% to 73.2% regardless of Linear or RBF. Another study \[6\] argues that with a more feature-rich dataset, traditional SVM could significantly improve its accuracy, specifically 93.26%. \[6\] got this high accuracy 93% by using 16 variables which is double of the study \[3\] that just got 83.5%. However,  \[6\] doesn’t show how much the conventional SVM accuracy could achieve. Thus, the result might not be convincing since they use different datasets but it still indicates that accuracy may be increased by adding more features.

Taking another approach rather than using conventional SVM like \[5\] or \[6\], \[3\] modifies SVM by using K-means clustering for feature extraction and SVM for classifier. Overall,  SVM in \[3\] achieves an accuracy of 98.7% for the PIMA dataset. \[3\] also validate the result of \[6\] by showing the accuracy of conventional SVM in different datasets is consistently around 83% for 8 features. Another attempt that works on the same PIMA dataset is \[7\]. \[7\] show that the traditional SVM accuracy could be improved by a maximum 2% regardless of the percentage of data training if backward elimination is used as feature selection. However, this method of backward elimination could be really slow during the test phase. This might suggest that it is not worth modifying.

Instead of modifying the traditional SVM, \[8\] show that when SVM is used as a meta-learner for the stacking method, it achieves a higher accuracy of high accuracy which is 93.75% in the same PIMA dataset. \[8\] achieve this accuracy by only combining 4 different weak models that all have accuracies below 78%. One problem of \[8\] might be that it involves setting hyperparameters for different models which is a lot of work and hard to find the optimal

In summary, \[5\] and \[6\] show that traditional SVM could be a good alternative to statistical models for detecting diabetes. Some modifications have been made in attempting to improve the traditional SVM\[3\]\[7\]. The study \[8\] used an ensemble method similar to our study where we combine multiple weak models to create a strong and robust one. However, \[8\] uses a stacking technique that combines many different types of models, which could result in the overhead of setting hyperparameters. Furthermore, the SVM as a meta-learner might not reflect the actual capability of SVM as it can just learn from the other models' predictions instead of raw data.

### Decision Tree

As per the literature, Decision Trees (DT) can be well used to classify Pima Indian Diabetes because they are hierarchical and have natural interpretability. As demonstrated in \[8\], DT algorithms, including J48 give explicit diagnostic rules, which can be more easily understand by medical professionals compared to the other classification models. This can help the extraction of if-then rules that can then be easily tested against clinical knowledge. But \[8\] also points out that the effectiveness of a single DT can be sensitive to the particular breakdown of the training data, and that accuracy can be in the range of 75-78% based on the depth of the tree.

The other study \[9\] shows that the appropriateness of DT is improved by the capability to deal with the unusual feature of the Pima dataset, e.g., the combination of categorical-quality counts (e.g., number of pregnancies) and continuous values (e.g., BMI). In comparison to distance-based models, DTs do not involve the use of large amounts of data normalization or scaling and, therefore, are more effective in preliminary stages of data mining. There is also a further indication that the recursive partitioning quality of the algorithm is well successful in the capture of the non-linear relationships between physiological indicators and onset of diabetes.

In a more technical sense, \[9\] determines that the main issue of using a conventional DT on the Pima data is the overfitting that can lead to a large decrease in prediction accuracy when adopted to unknown test data. In response to this, \[9\] recommends pruning approaches and a minimum number of samples per leaf, to stabilize the model. The significance of feature engineering is supported in \[4\], indicating that a pruned DT can reach a good accuracy of about 82.4 with careful selection of significant and valuable features such as Glucose and BMI, compared to the average accuracy of around 41.7% of the base models that blindly use all feasible features.

Compared to the fact that the Pima data have been used so far with the help of the help of one tree, \[10\] proves that the ensemble methods based on the Decision tree are the most robust ones. This is demonstrated in \[10\] with figures showing that a single tree shows high variance though with ensemble methods such as the random forest or the gradient boosting which combines the decision of many trees, accuracy would rise to over 90 percent. Among the downsides mentioned in \[10\] is that when the model is one tree it is initially very interpretable, and because the model is now an ensemble it obtains a bit of predictive power but at the cost of some interpretability.

Overall, \[8\] and \[11\] define the Decision Tree as a more realistic and interpretable reference point in classifying diabetes, especially as a clinical reference with its transparency. Adjustments to pruning and feature choice \[9\]\[4\] are necessary to deal with the threats of overfitting that are present in the Pima dataset. Lastly, \[10\] demonstrates that ensemble techniques are by far the strongest application of DT logic, as it provides a better trade off between accuracy and reliability to diagnostic uses.

# Explain the Implementation

### Support Vector Machine

The general idea of the Support Vector Machine is to find the hyperplane that separates different classes in the feature space. A good SVM is one that can determine optimal support vectors to maximize the margin of the hyperplane. Maximizing the margin allows for improving the model’s generalization ability

<img width="512" height="377" alt="image" src="https://github.com/user-attachments/assets/44d789d0-8091-4185-a18e-b22885d0cfc4" />

*<p align="center">Figure 1. Hyperplane in SVM</p>*

The usual linear SVM can only work with linearly separable data.  For the non-linearly separable input, it requires the kernel trick to transform it to a higher dimension where it is linearly separable.

<img width="975" height="226" alt="image" src="https://github.com/user-attachments/assets/71000e11-8d0f-4d13-8b73-f2829d24fd71" />

*<p align="center">Figure 2. Transform the input data into higher dimensions using Kernel tricks</p>*


For the unseen and unpredictable data, the study makes no assumptions about it. Thus, it will use the Kernel tricks (which is the _rbf_ option) to make sure that the model can work in any situation.

The SVM is chosen because it is widely used in the literature, specifically in detecting diabetes. It is complex enough to handle the complexity of the data, but also simple enough to be a good alternative for complicated statistical methods. Thus, SVM is the reasonable choice for

### Decision Tree

Instead of using the plain linear regression and logistic regression models, the advantage of using the decision tree is that we can split the data multiple times according to certain cutoff values in the features. Through splitting, different subsets of the dataset are created, with each instance belonging to one subset. The final subsets are called terminal or leaf nodes, and the intermediate subsets are called internal nodes or split nodes. To predict the outcome in each leaf node, the average outcome of the training data in this node is used.

<img width="975" height="481" alt="image" src="https://github.com/user-attachments/assets/f3b02452-64ef-4aba-8f27-059250d9ba9b" />


*<p align="center">Figure 3. Decision tree implementation \[12\]</p>*
### Bagging

Our study provides a simpler technique than stacking method \[7\] called “Bagging”. Bagging is one of the ensemble method where it combine multiple models to create a single model. Ensemble method used in \[7\] works with different models whereas Bagging used in this study works with same models


<img width="902" height="276" alt="image" src="https://github.com/user-attachments/assets/2abfc2d5-e615-46db-bd3f-35c4ac9ce8ea" />


*<p align="center">Figure 4. Bagging technique</p>*
The figure above show us how the bagging technique works. We train many models of the same algorithm and combine them together using major voting. Major voting is simply choose the outcome that has the most occurrences among n Prediction of n Model.

Bagging is used because it could help reduce the overfitting since each model is trained on different subset of the data. It could also significantly improve the accuracy of the algorithm\[13\]. Additionally, this technique only requires setting up the hyperparameters for only one time and could be more accurate in reflecting the ability of SVM in predicting raw data

# Evaluation (empirical)

### SVM

<img width="616" height="604" alt="image" src="https://github.com/user-attachments/assets/12431824-c52a-47b1-a95a-2034ad850e01" />


*<p align="center">Figure 5. Result Metrics for the SVM</p>*
Compared to the traditional accuracy of SVM in \[3\] with just 83.5, the SVM with the bagging method slightly increases the accuracy by roughly 1%.

In general, the metrics such as precision, recall, and F1-score work well for non-diabetes with 87%, 90%, and 88%, respectively. However, those metrics are significantly lower than for diabetes, with 79%, 73% and 76%, respectively.

### Decision Tree

<img width="698" height="667" alt="image" src="https://github.com/user-attachments/assets/22189dc8-49b9-4125-aef8-418a11acf73f" />


*<p align="center">Figure 6. Result Metrics for the Decision Tree</p>*
Compared to the performance of the conventional model in \[8\] with just an accuracy of 74.6%, the bagging method significantly increases the accuracy of the Decision Tree to 79,22%. The difference is 79,22%-74,6 = 4.62% which is four times larger than the difference of SVM.

In general, the metrics such as precision, recall, and F1-score work well for non-diabetes with equal scores of 84%. However, those metrics are significantly lower than for diabetes, with 69%, 60% and 69%.

### Comparison between SVM and Decision Tree

Overall, the Decision Tree experienced 4% increase in accuracy whereas SVM only received a slight 1% increase. However, the accuracy of SVM 84,4% is larger than the accuracy of the Decision Tree which only has 79%. Other metrics of SVM, such as precision, recall, and F1 score perform better than Decision Tree for both diabetes and non-diabetes.

# Conclusion

In this study, we have shown that the bagging method does increase the performance of the chosen algorithms Decision Tree and SVM in detecting diabetes. Especially, it improves the accuracy of the Decision Tree by 4%. This extends the knowledge of the literature on how Bagging could be used for machine learning in the medical field. Additionally, we compared two models and found that the SVM with bagging performs better in detecting diabetes. Thus, our study recommends the combination of SVM and Bagging for detecting diabetes. Our study is restricted to only diabetes, but the methodology of the study could be used for future studies, such as detecting Chronic Kidney Disease or Obesity

# References

\[1\] P. Saeedi et al., “Mortality attributable to diabetes in 20–79 years old adults, 2019 estimates: Results from the International Diabetes Federation Diabetes Atlas, 9th edition,” Diabetes Research and Clinical Practice, vol. 162, art. no. 108086, 2020, doi: 10.1016/j.diabres.2020.108086.

\[2\] D. Xie et al., “Global burden and influencing factors of chronic kidney disease due to type 2 diabetes in adults aged 20–59 years, 1990–2019,” Scientific Reports, vol. 13, no. 1, art. no. 20234, 2023, doi: 10.1038/s41598-023-47091-y.

\[3\] N. Arora, A. Singh, M. Z. N. Al-Dabagh, and S. K. Maitra, “A novel architecture for diabetes patients’ prediction using K-means clustering and SVM,” Mathematical Problems in Engineering, vol. 2022, pp. 1–9, 2022, doi: 10.1155/2022/4815521.

\[4\] R. Maimaitituerxun et al., “Predictive model for identifying mild cognitive impairment in patients with type 2 diabetes mellitus: A CHAID decision tree analysis,” Brain and Behavior, vol. 14, e3456, 2024, doi: 10.1002/brb3.3456.

\[5\] W. Yu, T. Liu, R. Valdez, M. Gwinn, and M. J. Khoury, “Application of support vector machine modeling for prediction of common diseases: The case of diabetes and pre-diabetes,” BMC Medical Informatics and Decision Making, vol. 10, 2010, doi: 10.1186/1472-6947-10-16.

\[6\] B. Rai, S. Sharma, M. Gupta, and M. Dinkar, “DDPIS: Diabetes disease prediction by improvising SVM,” International Journal of Reliable and Quality E-Healthcare, vol. 12, no. 2, pp. 1–11, 2023, doi: 10.4018/IJRQEH.318090.

\[7\] F. Maulidina et al., “Feature optimization using backward elimination and support vector machines (SVM) algorithm for diabetes classification,” Journal of Physics: Conference Series, vol. 1821, no. 1, art. no. 012006, 2021, doi: 10.1088/1742-6596/1821/1/012006.

\[8\] A. Khan et al., “Cardiovascular and diabetes diseases classification using ensemble stacking classifiers with SVM as a meta classifier,” Diagnostics, vol. 12, no. 11, p. 2595, 2022, doi: 10.3390/diagnostics12112595.

\[9\] J. Sadhasivam et al., “(Title unavailable in provided text),” Journal of Physics: Conference Series, vol. 1964, art. no. 062116, 2021, doi: 10.1088/1742-6596/1964/6/062116.

\[10\] D. Pei, C. Zhang, Y. Quan, and Q. Guo, “Identification of potential type II diabetes in a Chinese population with a sensitive decision tree approach,” Journal of Diabetes Research, vol. 2019, art. no. 4248218, 2019, doi: 10.1155/2019/4248218.

\[11\] Y.-Y. Song and Y. Lu, “Decision tree methods: Applications for classification and prediction,” Shanghai Archives of Psychiatry, vol. 27, no. 2, pp. 130–135, Apr. 2015, doi: 10.11919/j.issn.1002-0829.215044.

\[12\] Salman, “Understanding Decision Trees: The Beginner’s Guide to Machine Learning,” Medium, Nov. 10, 2025. \[Online\]. Available: https://medium.com/@salmanraju1809/understanding-decision-trees-the-beginners-guide-to-machine-learning-d81bbcbb7757

\[13\] H. Zhao, W. Liu, Y. Wang, and L. Wu, “Comparative analysis of algorithmic approaches in ensemble learning: bagging vs. boosting,” _Scientific Reports_, vol. 15, art. no. 34218, 2025, doi: 10.1038/s41598-025-15971-0.
*<p align="center">Figure 1. Hyperplane in SVM</p>*


