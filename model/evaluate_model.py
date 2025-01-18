import joblib
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import xgboost as xgb
import cupy as cp

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

dataset_test_path = "../data/KDDTest+.csv"

data = pd.read_csv(dataset_test_path, names=columns)

print("Rótulos únicos no dataset original de teste:")
print(data["label"].value_counts())

data["label"] = data["label"].apply(lambda x: 0 if x == 21 else 1)

print("Distribuição de classes após transformação:")
print(data["label"].value_counts())

categorical_columns = ["protocol_type", "service", "flag"]

data = pd.get_dummies(data, columns=categorical_columns)

X_test = data.drop("label", axis=1)
y_test = data["label"]

X_train_path = "../data/KDDTrain+.csv"
X_train = pd.read_csv(X_train_path, names=columns)

X_train["label"] = X_train["label"].apply(lambda x: 0 if x == 21 else 1)
X_train = pd.get_dummies(X_train, columns=categorical_columns)
X_train_features = X_train.drop("label", axis=1)

X_test = X_test.reindex(columns=X_train_features.columns, fill_value=0)

X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0)

xgb_model = joblib.load("../model/xgb_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

X_test_scaled = scaler.transform(X_test)

X_test_scaled_gpu = cp.array(X_test_scaled)

y_pred_proba_gpu = xgb_model.predict_proba(X_test_scaled_gpu)[:, 1]

threshold = 0.3
y_pred = (cp.asnumpy(y_pred_proba_gpu) > threshold).astype(int)

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

feature_importance = xgb_model.feature_importances_
important_features = sorted(
    enumerate(feature_importance), key=lambda x: x[1], reverse=True
)[:10]
print("Top 10 Features Importantes:")
for idx, importance in important_features:
    print(f"Feature {idx}: {importance:.4f}")
