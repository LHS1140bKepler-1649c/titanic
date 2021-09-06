import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score
from sklearn import svm
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import activations

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

def probability(a, b):
    return a / (a + b)

def splitBySex(sex=np.array(dftrain['sex']), age=np.array(dftrain['age']), survival = np.array(y_train)):
    males_age = list()
    females_age = list()
    males_survival = list()
    females_survival = list()
    for i in range(len(sex)):
        if sex[i] == 'male':
            males_age.append(age[i])
            males_survival.append(survival[i])
        elif sex[i] == 'female':
            females_age.append(age[i])
            females_survival.append(survival[i])
    return np.array(males_age), np.array(males_survival), np.array(females_age), np.array(females_survival)

def survivalProbabilitesByAge(age=np.array(dftrain['age']), survival=np.array(y_train), _save = False):

    survived = list()
    not_survived = list()
    for i in range(len(age)):
        if survival[i] == 1:
            survived.append(age[i])
        else:
            not_survived.append(age[i])
    
    survived = sorted(survived)
    not_survived = sorted(not_survived)

    c_s = Counter(survived)
    c_n = Counter(not_survived)
    c_all = Counter(age)
    
    probabilites = list()
    ordered_ages = list()
    for key, value in c_s.items():
        b = c_n[key]
        probabilites.append(probability(value, b))
        ordered_ages.append(key)
    plt.scatter(ordered_ages, probabilites)
    if _save == True:
        plt.savefig('survivalProbabilitesByAge.jpg')
    plt.show()

    return probabilites, ordered_ages

def survivalProbabilitiesBySex(sex=np.array(dftrain['sex']), survival = np.array(y_train), _save=True):
    survived = list()
    not_survived = list()
    for i in range(len(sex)):
        if survival[i] == 1:
            survived.append(sex[i])
        else:
            not_survived.append(sex[i])
    survived = sorted(survived)
    not_survived = sorted(not_survived)
    c_s = Counter(survived)
    c_n = Counter(not_survived)
    c_all = Counter(sex)
    probabilites = list()
    ordered_sex = list()
    for key, value in c_s.items():
        b = c_n[key]
        probabilites.append(probability(value, b))
        ordered_sex.append(key)
    plt.bar(ordered_sex, probabilites)
    if _save == True:
        plt.savefig('survivalProbabilitesBySex.jpg')
    plt.show()

def predict_probability(dftest=dfeval, y_test=y_eval):
    survived = list()
    for sex in dftest['sex']:
        if sex == 'female':
            survived.append(1)
        elif sex == 'male':
            survived.append(0)
    rate_of_success = 0
    for i in range(len(survived)):
        if survived[i] == y_eval[i]:
            rate_of_success += 1
    rate_of_success = rate_of_success / len(survived)
    return rate_of_success

def preprocessData(df):
    new_df = df.drop(['deck', 'parch', 'embark_town', 'fare'], axis=1)
    encoder = OrdinalEncoder(categories=['Third', 'Second', 'First'])
    cat = pd.Categorical(new_df['class'], 
                     categories=['unknown', 'Third', 
                                 'Second', 'First'], 
                     ordered=True)
    cat.fillna('unknown')
    labels, unique = pd.factorize(cat, sort=True)
    new_df['class'] = labels
    new_df['sex'] = [1 if sex == 'male' else 0 for sex in new_df['sex']]
    new_df['alone'] = [1 if alone == 'y' else 0 for alone in new_df['alone']]
    return new_df

def preprocessDataAgeInterval(df):
    intervals = [x for x in range(0, 110, 10)]
    ages = list()
    new_df = df.drop(['deck', 'parch', 'embark_town', 'fare'], axis=1)
    for age in new_df['age']:
        for i in range(len(intervals) - 1):
            if age >= intervals[i] and age < intervals[i + 1]:
                ages.append(intervals[i])
    encoder = OrdinalEncoder(categories=['Third', 'Second', 'First'])
    cat = pd.Categorical(new_df['class'], 
                     categories=['unknown', 'Third', 
                                 'Second', 'First'], 
                     ordered=True)
    cat.fillna('unknown')
    labels, unique = pd.factorize(cat, sort=True)
    new_df['class'] = labels
    new_df['sex'] = [1 if sex == 'male' else 0 for sex in new_df['sex']]
    new_df['alone'] = [1 if alone == 'y' else 0 for alone in new_df['alone']]
    new_df['age'] = ages
    return new_df

def train(train, y):
    nb = MultinomialNB()
    nb.fit(train.to_numpy(), y.to_numpy())
    MultinomialNB()
    return nb

def trainBernoulli(train, y):
    nb = BernoulliNB()
    nb.fit(train.to_numpy(), y.to_numpy())
    BernoulliNB()
    return nb

def trainSVM(train, y):
    s = svm.SVC(kernel='rbf')
    s.fit(train, y)
    svm.SVC(kernel='rbf')
    return s

def trainSGD(train, y):
    sgd = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
    sgd.fit(train, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                ('sgdclassifier', SGDClassifier())])
    return sgd

def predict(model, test):
    return model.predict(test.to_numpy())

def dnn_model(x_train, y_train):
    x_train=np.asarray(x_train).astype('float32')
    y_train=np.asarray(y_train).astype('float32')
    x_val = x_train[-50:]
    y_val = y_train[-50:]
    x_train = x_train[:-50]
    y_train = y_train[:-50]
    inputs = keras.Input(shape=(5,))
    x = layers.Dense(25, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(50, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=8,
        epochs=70,
        validation_data=(x_val, y_val),
    )
    return model, history

def dnn_model2(x_train, y_train):
    x_train=np.asarray(x_train).astype('float32')
    y_train=np.asarray(y_train).astype('float32')
    x_val = x_train[-50:]
    y_val = y_train[-50:]
    x_train = x_train[:-50]
    y_train = y_train[:-50]
    inputs = keras.Input(shape=(5,))
    x = layers.Dense(25, activation="relu")(inputs)
    x = layers.Dense(50, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(25, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=8,
        epochs=100,
        validation_data=(x_val, y_val),
    )
    return model, history

def predict_dnn(model, test_data):
    predicted = model.predict(np.asarray(test_data).astype('float32'), batch_size=8)
    predicts = list()
    for elem in predicted:
        predicts.append(int(np.argmax(elem)))
    return predicts



if __name__ == '__main__':

    train_df_interval = preprocessDataAgeInterval(dftrain)
    test_df_interval = preprocessDataAgeInterval(dfeval)
    model_interval_m = train(train_df_interval, y_train)
    prediction_interval_m = predict(model_interval_m, test_df_interval)
    model_interval_b = trainBernoulli(train_df_interval, y_train)
    prediction_interval_b = predict(model_interval_b, test_df_interval)
    model_svm_interval = trainSVM(train_df_interval, y_train)
    prediction_svm_interval = predict(model_svm_interval, test_df_interval)

    train_df = preprocessData(dftrain)
    test_df = preprocessData(dfeval)
    model_m = train(train_df, y_train)
    prediction_m = predict(model_m, test_df)
    model_b = trainBernoulli(train_df, y_train)
    prediction_b = predict(model_b, test_df)
    model_svm = trainSVM(train_df, y_train)
    prediction_svm = predict(model_svm, test_df)
    model_sgd = trainSVM(train_df, y_train)
    prediction_sgd = predict(model_sgd, test_df)

    model, history = dnn_model2(train_df, y_train)
    predicted_dnn = predict_dnn(model, test_df)

    print(f'Accuracy interval with Bernoulli: {accuracy_score(y_eval, prediction_interval_b)}')
    print(f'Accuracy interval with multinomial: {accuracy_score(y_eval, prediction_interval_m)}')
    print(f'Accuracy Bernoulli: {accuracy_score(y_eval, prediction_b)}')
    print(f'Accuracy multinomial: {accuracy_score(y_eval, prediction_m)}')
    print(f'Accuracy interval with SVM: {accuracy_score(y_eval, prediction_svm_interval)}')
    print(f'Accuracy SVM: {accuracy_score(y_eval, prediction_svm)}')
    print(f'Accuracy SGD: {accuracy_score(y_eval, prediction_sgd)}')
    print(f'Accuracy DNN: {accuracy_score(y_eval, predicted_dnn)}')