# Importing required modules
from DataPreparation import prepare_training_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def build_baseline():
    '''
    Building the baseline model

    Returns:
        Model: SKlearn logistic regression model object
    '''
    # Preparing data
    train_data, val_data = prepare_training_data(cleaning = True)
    train_text = [row['text'] for row in train_data]
    train_label = [row['target'] for row in train_data]
    val_text = [row['text'] for row in val_data]
    val_label = [row['target'] for row in val_data]

    # Creating word embedding
    vectorizer = TfidfVectorizer(lowercase = True, stop_words = 'english', token_pattern = r'(?u)\b\w+\b|\,|\.|\;|\:')
    train_matrix = vectorizer.fit_transform(train_text)
    val_matrix = vectorizer.transform(val_text)

    # Hyperparameter searching
    model = LogisticRegression(random_state = 2020)
    grid_value = {'C' : [0.01, 0.1, 1, 3, 5]}
    grid_search_model = GridSearchCV(model, param_grid = grid_value, scoring = 'neg_log_loss', cv = 5)
    grid_search_model.fit(train_matrix, train_label)

    # Model preparaing
    model = LogisticRegression(**grid_search_model.best_params_, random_state = 2020)
    model.fit(train_matrix, train_label)

    # Validating
    val_pred = model.predict(val_matrix)
    acc = round(accuracy_score(val_label, val_pred), 2)
    print(f'Baseline model provide validation accuracy : {acc}')

    return model