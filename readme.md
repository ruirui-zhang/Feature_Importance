

Feature\_Importance

March 11, 2023

**1 Automatic Feature Importances & selection**

Feature importances are an important topic in machine learning because they help us understand

which features (or variables) are the most important in predicting the target variable. Feature

importance is the process of determining which features contribute most to the accuracy of a

machine learning model. By understanding the importance of each feature, we can identify which

variables have the most impact on the model’s predictions and make better decisions about feature

selection, model building, and problem-solving.

• Data-based feature importances strategies

• Model-based feature importance strategies

• Automatic comparing strategy for feature importances

• Automatic feature selection algorithm

[2]: **import numpy as np**

**import pandas as pd**

**import shap**

**import matplotlib.pyplot as plt**

**from sklearn.base import** clone

**from sklearn.preprocessing import** StandardScaler

**from sklearn.decomposition import** PCA

**from sklearn.metrics import** r2\_score, mean\_squared\_error, log\_loss

**from sklearn.model\_selection import** train\_test\_split

**from sklearn.ensemble import** RandomForestRegressor, RandomForestClassifier

**1.1 Datasets**

**1.1.1 Wine Quality Dataset**

The Wine Quality Dataset involves predicting the quality of white wines on a scale given chemical

measures of each wine.

It is a multi-class classiﬁcation problem, but could also be framed as a regression problem. The

number of observations for each class is not balanced. There are 4,898 observations with 11 input

variables and one output variable. The variable names are as follows:

• Fixed acidity.

• Volatile acidity.

• Citric acid.

• Residual sugar.

1





• Chlorides.

• Free sulfur dioxide.

• Total sulfur dioxide.

• Density.

• pH.

• Sulphates.

• Alcohol.

• Quality (score between 0 and 10).

[3]: wine\_data = pd.read\_csv('winequality-white.csv', delimiter=';')

wine\_data.head()

[3]:

fixed acidity volatile acidity citric acid residual sugar chlorides

\

0

1

2

3

4

7\.0

6\.3

8\.1

7\.2

7\.2

0\.27

0\.30

0\.28

0\.23

0\.23

0\.36

0\.34

0\.40

0\.32

0\.32

20\.7

1\.6

6\.9

8\.5

8\.5

0\.045

0\.049

0\.050

0\.058

0\.058

free sulfur dioxide total sulfur dioxide density

pH sulphates

\

0

1

2

3

4

45\.0

14\.0

30\.0

47\.0

47\.0

170\.0

132\.0

97\.0

186\.0

186\.0

1\.0010 3.00

0\.45

0\.49

0\.44

0\.40

0\.40

0\.9940 3.30

0\.9951 3.26

0\.9956 3.19

0\.9956 3.19

alcohol quality

0

1

2

3

4

8\.8

9\.5

6

6

6

6

6

10\.1

9\.9

9\.9

**1.2 Data-based Feature Importances Strategies**

The simplest technique to identify important regression features is to rank them by their correlation

coeꢀcient; the feature with the largest coeꢀcient is taken to be the most important.

**1.3 Correlation**

[5]: **def** top\_rank\_corr\_based(df, target, n=**None**, ascending=**False**, method='spearman'):

"""

Calculate first / last N correlation with target

This kind of importance is called single-feature relevance importance

But suffers in the presence of codependent features.

Groups of features with similar relationships to the response variable␣

↪

receive the same or similar ranks,

even though just one should be considered important.

2





pearson : standard correlation coefficient

kendall : Kendall Tau correlation coefficient

spearman : Spearman rank correlation

:return:

"""

**if not** n:

n = len(df.columns)

feas = list(abs(df.corr(method=method)[target]).

sort\_values(ascending=ascending).index[1:n+1])

vals = list(abs(df.corr(method=method)[target]).

sort\_values(ascending=ascending))[1:n+1]

**return** feas, vals

↪

↪

[89]: **def** list\_to\_pd(fea,val):

**return** pd.DataFrame({'Feature':fea,'Feature\_importance':val})

**def** plot\_feat\_imp(df):

ordered\_df = df.sort\_values(by='Feature\_importance',ascending=**True**).

↪

reset\_index(drop=**True**)

fig, ax = plt.subplots(figsize=(8,6))

my\_range = range(1, len(df.index) + 1)

ax.hlines(y=my\_range, xmin=0, xmax=ordered\_df['Feature\_importance'],

color='blue')

**for** i,val **in** enumerate(ordered\_df.Feature\_importance):

**if** np.sign(val)==1:

ax.annotate(f"**{**val**:**0.3f**}**", (val + 0.002, i+1+0.1), size=11,␣

↪

↪

annotation\_clip=**False**)

**else**:

ax.annotate(f"**{**val**:**0.3f**}**", (val - 0.002, i+1+0.1), size=11,␣

annotation\_clip=**False**)

ax.scatter(ordered\_df['Feature\_importance'], my\_range, marker = 'o')

ax.set\_yticks(ordered\_df.index+1)

ax.set\_yticklabels(labels = ordered\_df.Feature.values,fontsize=12)

ax.set\_xlabel('Feature Importance',fontsize=12)

**if** np.sign(min(ordered\_df.Feature\_importance))==1:

ax.set\_xlim(min(ordered\_df.Feature\_importance)-min(ordered\_df.

Feature\_importance),max(ordered\_df.Feature\_importance)+max(ordered\_df.

Feature\_importance)\*0.5)

↪

↪

↪

**else**:

ax.set\_xlim(min(ordered\_df.Feature\_importance)+min(ordered\_df.

Feature\_importance),max(ordered\_df.Feature\_importance)+max(ordered\_df.

Feature\_importance)\*0.5)

↪

**return** plt

3





[90]: feas, vals = top\_rank\_corr\_based(wine\_data,'quality',method='spearman')

df = list\_to\_pd(feas,vals)

plot\_feat\_imp(df).show()

**1.4 PCA**

Another possibility is to use principle component analysis (PCA), which operates on just the X

explanatory matrix. PCA transforms data into a new space characterized by eigenvectors and

identiﬁes features that explain the most variance in the new space. If the ﬁrst principal component

covers a large percentage of the variance, the “loads” associated with that component can indicate

importance of features in the original X space.

[91]: **def** top\_rank\_PCA\_based(df, target, n=**None**, ascending=**False**):

**if not** n:

n = len(df.columns)

scaler = StandardScaler()

feas = [col **for** col **in** df.columns **if** col != target]

X = scaler.fit\_transform(df.loc[:, feas])

pca = PCA(n\_components=0.9)

pca.fit(X)

featimp = {feas[i]:abs(pca.components\_[0])[i] **for** i **in** range(len(feas))}

feas = sorted(featimp, key=featimp.get, reverse=**True**)[:n]

vals = [featimp[fea] **for** fea **in** feas]

**return** feas, vals

4





[92]: feas, vals = top\_rank\_PCA\_based(wine\_data,'quality')

df = list\_to\_pd(feas,vals)

plot\_feat\_imp(df).show()

**1.5 Minimal-redundancy-maximal-relevance (mRMR)**

This algorithm tends to select a subset of features having the most correlation with the class

(output) and the least correlation between themselves. It ranks features according to the minimal-

redundancy-maximal-relevance criterion which is based on mutual information

• Relevance: The coeꢀcient between the individual feature and the target

• Redundancy: The coeꢀcient between each individual feature.

[93]: **def** mRMR(df, target, mode='spearman', n=**None**, info=**False**):

**if not** n:

n = len(df.columns)

mrmr = dict()

\# use different mode to calculate correaltion

feas, imps = top\_rank(df, target, method=mode)

corr = dict([(fea, imp) **for** imp, fea **in** zip(imps, feas)])

selected\_feat = [feas[0]]

**for** i **in** range(len(feas)):

5





rest\_feat = [col **for** col **in** feas **if** col **not in** selected\_feat]

**if not** len(rest\_feat):

**break**

candi\_mrmr = []

**for** fi **in** rest\_feat:

redundancy = 0

relevance = corr[fi]

**for** fj **in** selected\_feat:

feas, imps = top\_rank(df.drop(columns=target), fj, method=mode)

corr\_fj = dict([(fea, imp) **for** imp, fea **in** zip(imps, feas)])

redundancy += corr\_fj[fi]

redundancy /= len(selected\_feat)

candi\_mrmr.append(relevance - redundancy)

max\_feature = rest\_feat[np.argmax(candi\_mrmr)]

mrmr[max\_feature] = np.max(candi\_mrmr)

**if** info:

print(f'**{**i+1**}** iteration, selected **{**max\_feature**}** feature with mRMR␣

value = **{**mrmr[max\_feature]**:**.3f**}**')

selected\_feat.append(max\_feature)

feat\_imp = pd.Series(mrmr.values(), index=mrmr.keys()).

sort\_values(ascending=**False**)

↪

↪

**return** feat\_imp.index[:n],feat\_imp.values[:n]

[94]: feas, vals = mRMR(wine\_data,'quality')

df = list\_to\_pd(feas,vals)

plot\_feat\_imp(df).show()

6





**1.6 Model based feature importance**

Model based feature importance involves ﬁtting a model such as random forest to all the features

and ﬁnding the relation between feature and the response variable. The importances obtained will

be dependant on the model chosen and the accuracy of our model ﬁt. Some of the methods that

we will try involve:

• Permutation importances

• Drop column importances

**1.7 Permutation Importance**

It works by shuﬄing the values of a single feature in the dataset and measuring the resulting

decrease in the model’s performance metric, such as accuracy, precision, or recall.

The permutation feature importance technique calculates the importance of a feature by comparing

the original model performance with the performance of the model after shuﬄing the values of that

feature. A feature is considered more important if shuﬄing its values leads to a larger decrease in

the model’s performance.

Permutation feature importance is a model-agnostic technique and can be applied to any type of

model, including linear regression, logistic regression, decision trees, and neural networks. It is a

popular method for feature selection, as it can identify the most relevant features in a dataset and

help to reduce the dimensionality of the input space, leading to simpler and more interpretable

models.

[98]: **def** rf\_model(x\_train, y\_train):

rf = RandomForestRegressor(n\_estimators=30,

min\_samples\_leaf=80,

max\_features=0.5,

max\_depth=10,

oob\_score=**True**,

n\_jobs=-1)

rf.fit(x\_train, y\_train)

**return** rf

[99]: # Model-based importance strategies

**def** permutation\_importance(X\_train, y\_train, X\_valid, y\_valid):

model = rf\_model(X\_train, y\_train)

baseline = r2\_score(y\_valid, model.predict(X\_valid))

imp = []

**for** col **in** X\_valid.columns:

save = X\_valid[col].copy()

X\_valid[col] = np.random.permutation(X\_valid[col])

m = r2\_score(y\_valid, model.predict(X\_valid))

X\_valid[col] = save

7





imp.append(baseline - m)

feat\_imp = pd.Series(imp, index=X\_valid.columns).

↪

sort\_values(ascending=**False**)

**return** feat\_imp.index,feat\_imp.values

[100]: **def** train\_val\_split(df, ratio):

train, val = train\_test\_split(df, train\_size=ratio, shuffle=**True**)

**return** train, val

**def** split\_target(df, target):

Y = df[target].values

X = df.drop(columns=[target])

**return** X, Y

X,Y = split\_target(wine\_data,'quality')

X\_train, X\_val = train\_val\_split(X, 0.7)

Y\_train, Y\_val = train\_val\_split(Y, 0.7)

feas, vals = permutation\_importance(X\_train,Y\_train,X\_val,Y\_val)

df = list\_to\_pd(feas,10\*vals)

plot\_feat\_imp(df).show()

8





**1.8 Drop Column Importances**

it’s a technique similar to permutation importance, but with the new model trained with one

column dropped to determine whether there is a diﬀerence in performance

[103]: **def** dropcol\_importances(X\_train, y\_train, X\_valid, y\_valid):

model = rf\_model(X\_train, y\_train)

baseline = model.oob\_score\_

imp = []

**for** col **in** X\_train.columns:

X\_train\_ = X\_train.drop(col, axis=1)

X\_valid\_ = X\_valid.drop(col, axis=1)

model\_ = clone(model)

model\_.fit(X\_train\_, y\_train)

m = model\_.oob\_score\_

imp.append(baseline - m)

feat\_imp = pd.Series(imp, index=X\_valid.columns).

↪

sort\_values(ascending=**False**)

**return** feat\_imp.index,feat\_imp.values

[105]: feas, vals = dropcol\_importances(X\_train,Y\_train,X\_val,Y\_val)

df = list\_to\_pd(feas,10\*vals)

plot\_feat\_imp(df).show()

9





**1.9 SHAP importance**

Shapley Additive Explanations (SHAP) feature importance is a method used to explain the output

of a machine learning model by attributing the importance of each input feature to the prediction.

It is based on game theory, speciﬁcally the Shapley value, which is a concept used to allocate

rewards fairly among players in a cooperative game. In the context of machine learning, each

feature can be seen as a player in a game, and the Shapley value measures the contribution of each

feature to the model output.

The SHAP feature importance method calculates the average contribution of each feature to the

model output across all possible permutations of features. By doing so, it provides a more accu-

rate and reliable measure of feature importance than other methods, such as permutation feature

importance or coeꢀcient values in linear models.

The SHAP feature importance method is model-agnostic, meaning it can be used with any machine

learning model, and it provides both global and local feature importance values. Global feature

importance measures the importance of a feature across the entire dataset, while local feature

importance measures the importance of a feature for a speciﬁc instance or prediction.

[124]: **def** shap\_importances(x\_train, y\_train, x\_val, y\_val):

rf = rf\_model(x\_train, y\_train)

shap\_values = shap.TreeExplainer(rf, data=x\_train).shap\_values(X=x\_val,␣

↪

y=y\_val, check\_additivity=**False**)

imp = np.mean(np.abs(shap\_values), axis=0)

**return** x\_val.columns,imp

[125]: feas, vals = shap\_importances(X\_train,Y\_train,X\_val,Y\_val)

df = list\_to\_pd(feas,10\*vals)

plot\_feat\_imp(df).show()

10





**1.10 Compare diﬀerent Strategies**

The function below compares diﬀerent strategies and draw a R2 score vs num of variables chosen

plot.

[143]: # Comparing strategies

**def** Top\_k\_loss(x\_train, y\_train, x\_val, y\_val, feat, imp, k=10,␣

↪

metric=r2\_score):

model = rf\_model(x\_train, y\_train)

loss\_list = []

n\_imp = pd.Series(imp, index=feat).sort\_values(ascending=**False**)[:k]

**for** i **in** range(1, k+1):

model\_ = clone(model)

features = n\_imp.index[:i]

model\_.fit(x\_train.loc[:, features], y\_train)

pred = model\_.predict(x\_val.loc[:, features])

loss = metric(y\_val, pred)

loss\_list.append(loss)

**return** loss\_list

**def** compare\_Top\_k(data, target, k=10):

train, val = train\_val\_split(data, 0.8)

x\_train, y\_train = split\_target(train, target)

11





x\_val, y\_val = split\_target(val, target)

feat\_spearman, imp\_spearman = top\_rank\_corr\_based(data, target,␣

method='spearman')

loss\_spearman = Top\_k\_loss(x\_train, y\_train, x\_val, y\_val, feat\_spearman,␣

imp\_spearman, k=k)

feat\_pearson, imp\_pearson = top\_rank\_corr\_based(data, target,␣

method='pearson')

loss\_pearson = Top\_k\_loss(x\_train, y\_train, x\_val, y\_val, feat\_pearson,␣

imp\_pearson, k=k)

feat\_kendall, imp\_kendall = top\_rank\_corr\_based(data, target,␣

method='kendall')

loss\_kendall = Top\_k\_loss(x\_train, y\_train, x\_val, y\_val, feat\_kendall,␣

imp\_kendall, k=k)

↪

↪

↪

↪

↪

↪

feat\_pca, imp\_pca = top\_rank\_PCA\_based(data, target)

loss\_pca = Top\_k\_loss(x\_train, y\_train, x\_val, y\_val, feat\_pca, imp\_pca)

feat\_perm, imp\_perm= permutation\_importance(x\_train, y\_train, x\_val, y\_val)

loss\_perm = Top\_k\_loss(x\_train, y\_train, x\_val, y\_val, feat\_perm, imp\_perm,␣

k=k)

feat\_drop, imp\_drop = dropcol\_importances(x\_train, y\_train, x\_val, y\_val)

loss\_drop = Top\_k\_loss(x\_train, y\_train, x\_val, y\_val, feat\_drop, imp\_drop,␣

k=k)

feat\_shap, imp\_shap = shap\_importances(x\_train, y\_train, x\_val, y\_val)

loss\_shap = Top\_k\_loss(x\_train, y\_train, x\_val, y\_val, feat\_shap, imp\_shap,␣

k=k)

feat\_mrmr, imp\_mrmr= mRMR(data, target)

loss\_mrmr = Top\_k\_loss(x\_train, y\_train, x\_val, y\_val, feat\_mrmr, imp\_mrmr,␣

k=k)

↪

↪

↪

↪

fig = plt.figure(figsize=(15,15))

ax = plt.axes()

ax.grid(**False**)

x, markers = range(1, k+1), ['o', '8', 's', 'p', '+', '\*', 'h', 'v']

plt.plot(x, loss\_spearman, '#BA5645', marker=markers[0], label='Spearman')

plt.plot(x, loss\_pearson, '#BA8949', marker=markers[1], label='Pearson')

plt.plot(x, loss\_kendall, '#8DBA49', marker=markers[2], label='Kendall')

plt.plot(x, loss\_pca, '#49A7BA', marker=markers[3], label='PCA')

plt.plot(x, loss\_perm, '#6E49BA', marker=markers[4], label='Permutation')

plt.plot(x, loss\_drop, '#BA49A0', marker=markers[5], label='Drop Column')

plt.plot(x, loss\_shap, '#878784', marker=markers[6], label='Shap')

plt.plot(x, loss\_mrmr, '#000000', marker=markers[7], label='mRMR')

handles, labels = ax.get\_legend\_handles\_labels()

ax.legend(handles, labels)

ax.set\_ylabel('R2 score', fontsize=10)

ax.set\_xlabel('Top K selected features', fontsize=10)

plt.show()

12





[144]: compare\_Top\_k(wine\_data, 'quality', 10)

[ ]:

[ ]:

13

