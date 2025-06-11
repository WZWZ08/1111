import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# 1. 数据加载
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 2. 文本预处理和特征提取
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X_train = tfidf.fit_transform(train_data['text'])
y_train = train_data['label']
X_test = tfidf.transform(test_data['text'])

# 3. 模型训练和评估
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': LinearSVC(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

for name, model in models.items():
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('clf', model)
    ])
    scores = cross_val_score(pipeline, train_data['text'], train_data['label'], cv=5, scoring='f1_macro')
    print(f"{name} - F1 Score: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 4. 选择最佳模型进行预测
best_model = LinearSVC()
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)

# 5. 生成提交文件
submission = pd.DataFrame({'id': test_data['id'], 'label': predictions})
submission.to_csv('submission.csv', index=False)