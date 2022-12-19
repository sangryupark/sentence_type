def label_to_num(data, label):
    label_dict = {}
    if label == "유형":
        label_dict = {
            "사실형": 0,
            "추론형": 1,
            "대화형": 2,
            "예측형": 3,
        }
    elif label == "극성":
        label_dict = {"긍정": 0, "부정": 1, "미정": 2}
    elif label == "확실성":
        label_dict = {
            "확실": 0,
            "불확실": 1,
        }
    elif label == "시제":
        label_dict = {
            "과거": 0,
            "현재": 1,
            "미래": 2,
        }
    return data.map(label_dict)
