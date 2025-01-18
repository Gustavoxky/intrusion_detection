import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import numpy as np
import cupy as cp

dataset_path = "../data/KDDTrain+.csv"

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

print("Carregando o dataset...")
data = pd.read_csv(dataset_path, names=columns)

data["label"] = data["label"].apply(lambda x: 0 if x == 21 else 1)

categorical_columns = ["protocol_type", "service", "flag"]

data = pd.get_dummies(data, columns=categorical_columns)

X = data.drop("label", axis=1)
y = data["label"]

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

print("Normalizando os dados...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

print("Aplicando SMOTE para balancear as classes...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

batch_size = 10000
n_batches = (len(X_train_balanced) + batch_size - 1) // batch_size 

print("Treinando o modelo com XGBoost em lotes...")
xgb_model = xgb.XGBClassifier(
    tree_method='hist',
    device='cuda', 
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)

for i in range(n_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(X_train_balanced))
    print(f"Treinando lote {i + 1}/{n_batches}...")
    
    X_batch = cp.array(X_train_balanced[start:end])
    y_batch = cp.array(y_train_balanced[start:end])
    
    if i == 0:
        xgb_model.fit(X_batch, y_batch)
    else:
        xgb_model.fit(X_batch, y_batch, xgb_model=xgb_model)

X_test_gpu = cp.array(X_test)
y_pred = xgb_model.predict(X_test_gpu)

y_pred_cpu = cp.asnumpy(y_pred)

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred_cpu))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_cpu))

feature_importance = xgb_model.feature_importances_
important_features = np.argsort(feature_importance)[-10:][::-1]
print("Top 10 Features Importantes:")
for idx in important_features:
    print(f"Feature {idx}: {feature_importance[idx]:.4f}")

print("Salvando o modelo...")
joblib.dump(xgb_model, "../model/xgb_model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")
print("Modelo treinado e salvo com sucesso!")
