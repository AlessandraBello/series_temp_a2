from model.preprocessor import Differencer, LogDiffTransform
import numpy as np

# Teste simples das transformações
series = np.array([10.0, 12.0, 15.0, 20.0, 25.0])

log_transformer = Differencer()
log_series = log_transformer.transform(series)
recovered_series = log_transformer.inverse_transform(log_series)
print("Original Series:", series)
print("Log Transformed Series:", log_series)
print("Recovered Series from Log:", recovered_series)

logdiff_transformer = LogDiffTransform()
logdiff_series = logdiff_transformer.transform(series)
recovered_logdiff_series = logdiff_transformer.inverse_transform(logdiff_series)
print("Log-Difference Transformed Series:", logdiff_series)
print("Recovered Series from Log-Difference:", recovered_logdiff_series)