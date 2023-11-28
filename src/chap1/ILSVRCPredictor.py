import numpy as np
# 出力結果からラベルを予測する後処理クラス
class ILSVRCPredictor():
    def __init__(self, class_index) -> None:
        self.class_index = class_index
    
    def predict_max(self, out):
        mx_id = np.argmax(out.detach().numpy())
        predict_label_name = self.class_index[str(mx_id)][1]

        return predict_label_name