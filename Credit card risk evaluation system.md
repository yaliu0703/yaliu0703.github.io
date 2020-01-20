## Decision support system (DSS) for evaluating the risk of Home Equity Line of Credit (HELOC) applications

**Project description:** 

This decision support system is designed to help bank managers predict risk performance of a credit card aplicant. In this project, I trained the predictive model with dataset first and then visualized the system with Streamlit. Bank managers not only can get decision support but also explaination from this system.



### 1. About Dataset

I use a real-world financial dataset provided by FICO with 14060 rows and 23 variables. The predictor variables are all quantitative or categorical, and come from anonymized credit bureau data. The target variable to predict is a binary variable called RiskPerformance. The value “Bad” indicates that a consumer was 90 days past due or worse at least once over a period of 24 months from when the credit account was opened. The value “Good” indicates that they have made their payments without ever being more than 90 days overdue. More details about dataset can be found [here](https://community.fico.com/s/explainable-machine-learning-challenge).

### 2. Predictive Model Training

#### 2.1 Prepare data

##### 2.1.1 Data Scaling

The dataset Homeline of Credit contains features highly varying in magnitudes, units and range. But since, most of the machine learning algorithms use Eucledian distance between two data points in their computations, this is a problem.

If left alone, these algorithms only take in the magnitude of features neglecting the units. The features with high magnitudes will weigh in a lot more in the distance calculations than features with low magnitudes.To suppress this effect, I need to control all features on the same level of magnitudes. This can be achieved by scaling.

Here I use the most common method StandardScaler(). Standardisation replaces the values by their Z scores.

This redistributes the features with their mean = 0 and standard deviation =1 . sklearn.preprocessing.StandardScaler helps me implementing standardisation in python.

##### 2.1.2 How to deal with special values (-7,-8,-9) in the dataset

From the data dictionary file, we know that ‘-7’ represents condition not Met (e.g. No Inquiries, No Delinquencies), -9 represents no Bureau Record or No Investigation and -8 represents no Usable/Valid Accounts Trades or Inquiries. There are a bunch of young people that do not have previous credit record before they apply for loan or credit cards. So we cannot arbitarily drop those data. Instead, we should remind the bank that though these people might be predicted as ‘Good’, it might be unknown risk to allow those people to apply for large number of loan.

#### 2.2 Model Comparison

The core method of our model selection is using GridSearchCV function to find out the model with highest CV score.

We first tried on single model include Decision tree, Logistic regression, KNN and SVM, and fine-tuned our model by trying different hyper-parameters until we found out the best one. Among all the single models, comparing both their CV scores and accuracy, SVM(rbf) has the best performance.

In order to further improve our accuracy, we then applied the aggregation method to our model training, which includes Boosting, Random Forest and Bagging models. Since SVM(rbf) performs really well in our single model training, we decided to use SVM(rbf) with the best hyper-parameters we found in the previous step as the base estimator of our bagging model. It is not surprising that all the three models have high CV scores and accuracy comparing to the single model method. Among them, the bagging model has the best performance. So we decided to choose the bagging model as our risk evaluation model, which has approximately 71.86% accuracy.

The following table shows the best estimator, CV score and accuracy of each model we have trained.







Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

```javascript
if (isAwesome){
  return true
}
```

### 2. Assess assumptions on which statistical inference will be based

```javascript
if (isAwesome){
  return true
}
```

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
